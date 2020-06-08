import tensorflow as tf
from self_awareness import networks

import gpflow
from gpflow.test_util import notebook_niter, is_continuous_integration
import gpflow.multioutput.kernels as mk

import tensorflow.contrib.distributions as tfd

from scipy.cluster.vq import kmeans2
import numpy as np

float_type = gpflow.settings.float_type

ITERATIONS = notebook_niter(1000)

class Model():

    def __init__(self, args, is_training=True):


        # Store the arguments
        self.args = args

        # args.rnn_size contains the dimension of the hidden state of the LSTM

        # TODO: (resolve) Do we need to use a fixed seq_length?
        # Input data contains sequence of (x,y) points
        self.input_data = tf.placeholder(tf.float32, [args.batch_size, args.target_image_size[0], args.target_image_size[1], 1])
        self.pred_data = tf.placeholder(tf.float32, [None, args.target_image_size[0], args.target_image_size[1], 1])

        # target data contains sequences of (x,y) points as well
        self.target_data = tf.placeholder(tf.float32, [args.batch_size, 7])
        self.trans_target, self.rot_target = tf.split(self.target_data, [3, 4], axis=1)

        if args.is_normalization:
            with tf.variable_scope('param'):
                self.norm_mean = tf.Variable(args.norm_mean, dtype=tf.float32, trainable=False, name="norm_mean")
                self.norm_std = tf.Variable(args.norm_std, dtype=tf.float32, trainable=False, name="norm_std")

            target_trans, target_rot = tf.split(self.target_data, [3, 4], axis=1)
            target_trans_centered = target_trans - tf.tile(tf.reshape(self.norm_mean, [1, 3]), [tf.shape(target_trans)[0], 1])
            target_trans_normed = target_trans_centered / tf.tile(tf.reshape(self.norm_std, [1, 3]), [tf.shape(target_trans)[0], 1])
            target_normed = tf.concat([target_trans_normed, target_rot], axis=1)
        else:
            target_normed = self.target_data

        network = networks.catalogue[args.network](args)
        output, trans_feat, rot_feat = network.build(self.input_data, is_training)
        _, rot_pred = tf.split(output, [3, 4], axis=1)

        pose_feat = tf.concat([trans_feat, rot_feat], axis=1)

        trans_target, rot_target = tf.split(target_normed, [3, 4], axis=1)

        f_X = tf.cast(trans_feat, dtype=float_type)
        Y = tf.cast(trans_target, dtype=float_type)

        '''Gaussian Process for translation regression'''
        with tf.variable_scope('gp'):
            kernel = mk.SharedIndependentMok(gpflow.kernels.RBF(args.feat_dim, ARD=False, name="rbf_ard"), args.output_dim)
            # kernel = mk.SeparateIndependentMok([gpflow.kernels.RBF(128, ARD=True, name="rbf_ard"+str(i)) for i in range(3)])
            q_mu = np.zeros((args.batch_size, args.output_dim)).reshape(args.batch_size * args.output_dim, 1)
            q_sqrt = np.eye(args.batch_size * args.output_dim).reshape(1, args.batch_size * args.output_dim, args.batch_size * args.output_dim)
            # feature = gpflow.features.InducingPoints(np.zeros((args.batch_size, 128)))

            self.gp_model = gpflow.models.SVGP(X=f_X, Y=Y, kern=kernel, likelihood=gpflow.likelihoods.Gaussian(name="lik"), Z=np.zeros((args.batch_size, args.feat_dim)), q_mu=q_mu, q_sqrt=q_sqrt, name="svgp")
            #self.gp_model = gpflow.models.SVGP(X=f_X, Y=Y, kern=kernel, Z=np.zeros((args.batch_size, 128), dtype=float_type), likelihood=gpflow.likelihoods.Gaussian(), num_latent=3)

        if is_training:
            with tf.variable_scope('adam'):
                cnn_tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vggnet_localization/regressor')
                gp_tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gp')

                # Learning rate
                self.lr = tf.Variable(args.learning_rate, trainable=False, name="learning_rate")
                # Global step
                self.global_step = tf.Variable(0, trainable=False, name="global_step")
                self.lamda_weights = tf.Variable(args.lamda_weights, trainable=False, name="category_weights", dtype=float_type)

                self.trans_loss = -self.gp_model.likelihood_tensor / args.batch_size

                '''Rotation loss'''
                rot_loss_1 = tf.reduce_mean(tf.square(rot_pred - rot_target), axis=1)
                rot_loss_2 = tf.reduce_mean(tf.square(rot_pred + rot_target), axis=1)

                tmp = tf.stack([rot_loss_1, rot_loss_2], axis=1)
                tmp = tf.reduce_min(tmp, axis=1)
                self.rot_loss = tf.cast(tf.reduce_mean(tmp), dtype=float_type)

                self.total_loss = self.trans_loss + self.lamda_weights * self.rot_loss

                gp_optimizer = tf.train.AdamOptimizer(self.lr)
                gp_grad_vars = gp_optimizer.compute_gradients(loss=self.total_loss, var_list=gp_tvars)

                cnn_optimizer = tf.train.AdamOptimizer(self.lr * 0.1)
                cnn_grad_vars = cnn_optimizer.compute_gradients(loss=self.total_loss, var_list=cnn_tvars)

                self.train_op =tf.group(gp_optimizer.apply_gradients(gp_grad_vars, global_step=self.global_step), cnn_optimizer.apply_gradients(cnn_grad_vars))

                #self.train_op = gp_optimizer.apply_gradients(gp_grad_vars, global_step=self.global_step)

        else:
            pred_output, pred_trans_feat, pred_rot_feat = network.build(self.pred_data, is_training)
            pred_feat = tf.concat([pred_trans_feat, pred_rot_feat], axis=1)

            c_mean, c_var = self.gp_model._build_predict(tf.cast(pred_trans_feat, dtype=float_type), full_cov=False, full_output_cov=False)
            y_mean, y_var = self.gp_model.likelihood.predict_mean_and_var(c_mean, c_var)

            trans_pred = tf.cast(y_mean, dtype=tf.float32)
            _, rot_pred = tf.split(pred_output, [3, 4], axis=1)

            dist = tfd.Normal(loc=tf.reshape(c_mean, [1,3]), scale=tf.reshape(c_var*1000., [1, 3]))
            samples = tf.cast(tf.reshape(dist.sample([100]), [100, 3]), dtype=tf.float32)

            # samples = []
            # for i in range(100):
            #     samples.append(tf.random.normal(shape=[1, 3], mean=tf.reshape(tf.cast(c_mean, dtype=tf.float32), [1,3]), stddev=tf.reshape(tf.cast(c_var, dtype=tf.float32)*1000., [1, 3])))
            #
            # samples = tf.reshape(tf.stack(samples, axis=0), [100, 3])

            self.distribution_mean = tf.cast(c_mean, dtype=tf.float32)
            self.distribution_cov = tf.cast(c_var, dtype=tf.float32)

            if args.is_normalization:
                target_trans_unscaled = trans_pred * tf.tile(tf.reshape(self.norm_std, [1, 3]), [tf.shape(trans_pred)[0], 1])
                target_trans_uncentered = target_trans_unscaled + tf.tile(tf.reshape(self.norm_mean, [1, 3]), [tf.shape(trans_pred)[0], 1])

                samples_unscaled = samples * tf.tile(tf.reshape(self.norm_std, [1, 3]), [tf.shape(samples)[0], 1])
                samples_uncentered = samples_unscaled + tf.tile(tf.reshape(self.norm_mean, [1, 3]), [tf.shape(samples)[0], 1])
                self.samples = samples_uncentered

                self.trans_prediction = target_trans_uncentered
                self.rot_prediction = rot_pred
            else:
                self.trans_prediction = trans_pred
                self.rot_prediction = rot_pred
                self.samples = samples





        # self.trans_prediction2 = trans_pred
        #


        # if is_training:
        #
        #     self.trans_loss = tf.losses.mean_squared_error(labels=trans_target, predictions=trans_pred)
        #     self.rot_loss0 = tf.losses.mean_squared_error(labels=rot_target, predictions=rot_pred)
        #
        #     rot_loss_1 = tf.reduce_mean(tf.square(rot_pred - rot_target), axis=1)
        #     rot_loss_2 = tf.reduce_mean(tf.square(rot_pred + rot_target), axis=1)
        #
        #     tmp = tf.stack([rot_loss_1, rot_loss_2], axis=1)
        #     tmp = tf.reduce_min(tmp, axis=1)
        #     self.rot_loss = tf.reduce_mean(tmp)
        #
        #     # Learning rate
        #     self.lr = tf.Variable(args.learning_rate, trainable=False, name="learning_rate")
        #     self.lamda_weights = tf.Variable(args.lamda_weights, trainable=False, name="category_weights")
        #
        #     self.total_loss = self.trans_loss + self.rot_loss * self.lamda_weights
        #
        #     tvars = tf.trainable_variables()
        #
        #     gradients = tf.gradients(self.total_loss, tvars)
        #     # Clip the gradients if they are larger than the value given in args
        #     grads, _ = tf.clip_by_global_norm(gradients, args.grad_clip)
        #     optimizer = tf.train.AdamOptimizer(self.lr)
        #     self.train_op = optimizer.apply_gradients(zip(grads, tvars))



    def evalute(self, sess):

        with tf.variable_scope('metrics/'):
            with tf.variable_scope('validation', reuse=True):

                self.trans_error, self.trans_update_op = tf.metrics.root_mean_squared_error(labels=self.trans_target, predictions=self.trans_pred, name='trans_error')
                self.rot_error, self.rot_update_op = tf.metrics.root_mean_squared_error(labels=self.rot_target, predictions=self.rot_pred, name='rot_error')

                error_vars = tf.contrib.framework.get_local_variables(scope='metrics/{}'.format('validation'))
                self.reset_op = tf.variables_initializer(var_list=error_vars)







