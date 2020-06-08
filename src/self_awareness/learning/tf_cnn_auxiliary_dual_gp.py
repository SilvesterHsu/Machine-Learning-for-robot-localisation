import tensorflow as tf
from src.self_awareness import networks

import numpy as np

import gpflow
from gpflow.test_util import notebook_niter, is_continuous_integration
import gpflow.multioutput.kernels as mk

import tensorflow.contrib.distributions as tfd

float_type = gpflow.settings.float_type


class Model():

    def __init__(self, args, is_training=True):

        # Store the arguments
        self.args = args

        with tf.variable_scope("param"):
            self.norm_mean = tf.Variable(args.norm_mean, dtype=tf.float32, trainable=False, name="norm_mean")
            self.norm_std = tf.Variable(args.norm_std, dtype=tf.float32, trainable=False, name="norm_std")

        if not is_training:
            batch_size = 1
        else:
            batch_size = args.batch_size

        # input_data_t_1 refers to data at time <t-1>
        self.input_data = tf.placeholder(tf.float32, [batch_size, args.target_image_size[0], args.target_image_size[1], 1])
        # target data
        self.target_data = tf.placeholder(tf.float32, [batch_size, 7])

        target_data_normed = self.normalize(self.target_data, self.norm_mean, self.norm_std)

        # dense_feat_net = networks.catalogue[args.network](args, name="dense_feat")
        # dense_feat = dense_feat_net.build(self.input_data, is_training, opt="base")

        dense_feat, _ = networks.resnet.resnet_v1_50(self.input_data, global_pool=False, num_classes=None,
                                                      is_training=is_training, reuse=tf.AUTO_REUSE, scope="dense_feat")

        # context stack for global pose learning
        global_context = networks.catalogue[args.network](args, name="context_stack_global")
        global_context_feat = global_context.build(dense_feat, is_training, opt="context")

        # regreesor for global pose learning
        global_regressor = networks.catalogue[args.network](args, name="regressor_global")
        global_output, trans_feat, rot_feat = global_regressor.build(global_context_feat, is_training, opt="regressor")

        _, rot_pred = tf.split(global_output, [3, 4], axis=1)

        pose_feat = tf.concat([trans_feat, rot_feat], axis=1)

        trans_target, rot_target = tf.split(target_data_normed, [3, 4], axis=1)

        f_X_t = tf.cast(trans_feat, dtype=float_type)
        Y_t = tf.cast(trans_target, dtype=float_type)

        f_X_r = tf.cast(trans_feat, dtype=float_type)
        Y_r = tf.cast(trans_target, dtype=float_type)

        '''Gaussian Process for translation regression'''
        with tf.variable_scope('gp'):
            # GP for translation learning
            kernel_t = mk.SharedIndependentMok(gpflow.kernels.RBF(args.feat_dim, ARD=False, name="rbf_ard"), args.output_dim)
            q_mu_t = np.zeros((args.batch_size, args.output_dim)).reshape(args.batch_size * args.output_dim, 1)
            q_sqrt_t = np.eye(args.batch_size * args.output_dim).reshape(1, args.batch_size * args.output_dim, args.batch_size * args.output_dim)

            self.gp_model_t = gpflow.models.SVGP(X=f_X_t, Y=Y_t, kern=kernel_t,likelihood=gpflow.likelihoods.Gaussian(name="lik"),
                                                Z=np.zeros((args.batch_size, args.feat_dim)), q_mu=q_mu_t, q_sqrt=q_sqrt_t, name="svgp")


            # GP for rotation learning
            kernel_r = mk.SharedIndependentMok(gpflow.kernels.RBF(args.feat_dim, ARD=False, name="rbf_ard"), args.output_dim)
            q_mu_r = np.zeros((args.batch_size, args.output_dim)).reshape(args.batch_size * args.output_dim, 1)
            q_sqrt_r = np.eye(args.batch_size * args.output_dim).reshape(1, args.batch_size * args.output_dim, args.batch_size * args.output_dim)

            self.gp_model_r = gpflow.models.SVGP(X=f_X_r, Y=Y_r, kern=kernel_r, likelihood=gpflow.likelihoods.Gaussian(name="lik"),
                                               Z=np.zeros((args.batch_size, args.feat_dim)), q_mu=q_mu_r, q_sqrt=q_sqrt_r, name="svgp")

        if is_training:
            with tf.variable_scope('adam'):
                cnn_tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='regressor_global')
                gp_tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gp')

                # Learning rate
                self.lr = tf.Variable(args.learning_rate, trainable=False, name="learning_rate")
                # Global step
                self.global_step = tf.Variable(0, trainable=False, name="global_step")
                self.lamda_weights = tf.Variable(args.lamda_weights, trainable=False, name="category_weights", dtype=float_type)

                self.trans_loss = -self.gp_model_t.likelihood_tensor / args.batch_size

                '''Rotation loss'''

                self.rot_loss = -self.gp_model_r.likelihood_tensor / args.batch_size

                self.total_loss = self.trans_loss + self.rot_loss

                gp_optimizer = tf.train.AdamOptimizer(self.lr)
                gp_grad_vars = gp_optimizer.compute_gradients(loss=self.total_loss, var_list=gp_tvars)

                cnn_optimizer = tf.train.AdamOptimizer(self.lr * 0.1)
                cnn_grad_vars = cnn_optimizer.compute_gradients(loss=self.total_loss, var_list=cnn_tvars)

                self.train_op = tf.group(gp_optimizer.apply_gradients(gp_grad_vars, global_step=self.global_step),
                                         cnn_optimizer.apply_gradients(cnn_grad_vars))

        else:

            c_mean_t, c_var_t = self.gp_model_t._build_predict(tf.cast(trans_feat, dtype=float_type), full_cov=False, full_output_cov=False)
            y_mean_t, y_var_t = self.gp_model_t.likelihood.predict_mean_and_var(c_mean_t, c_var_t)

            trans_pred = tf.cast(y_mean_t, dtype=tf.float32)

            c_mean_r, c_var_r = self.gp_model_r._build_predict(tf.cast(rot_feat, dtype=float_type), full_cov=False, full_output_cov=False)
            y_mean_r, y_var_r = self.gp_model_r.likelihood.predict_mean_and_var(c_mean_r, c_var_r)

            rot_pred = tf.cast(y_mean_r, dtype=tf.float32)

            dist = tfd.Normal(loc=tf.reshape(c_mean_t, [1, 3]), scale=tf.reshape(c_var_t * 1000., [1, 3]))
            samples = tf.cast(tf.reshape(dist.sample([100]), [100, 3]), dtype=tf.float32)

            self.distribution_mean = tf.cast(c_mean_t, dtype=tf.float32)
            self.distribution_cov = tf.cast(c_var_t, dtype=tf.float32)

            trans_pred_demormed = self.denomalize_navie(trans_pred, self.norm_mean, self.norm_std)
            samples_demormed = self.denomalize_navie(samples, self.norm_mean, self.norm_std)

            self.trans_prediction = trans_pred_demormed
            self.rot_prediction = rot_pred
            self.samples = samples_demormed


    @staticmethod
    def translational_rotational_loss(pred=None, gt=None, lamda=None):

        trans_pred, rot_pred = tf.split(pred, [3, 4], axis=1)
        trans_gt, rot_gt = tf.split(gt, [3, 4], axis=1)

        trans_loss = tf.losses.mean_squared_error(labels=trans_gt, predictions=trans_pred)

        rot_loss = 1. - tf.reduce_mean(tf.square(tf.reduce_sum(tf.multiply(rot_pred, rot_gt), axis=-1)))

        loss = trans_loss + lamda * rot_loss

        return loss, trans_loss, rot_loss

    @staticmethod
    def rotational_loss(pred=None, gt=None):
        rot_loss_1 = tf.reduce_mean(tf.square(pred - gt), axis=1)
        rot_loss_2 = tf.reduce_mean(tf.square(pred + gt), axis=1)

        tmp = tf.stack([rot_loss_1, rot_loss_2], axis=1)
        tmp = tf.reduce_min(tmp, axis=1)
        rot_loss = tf.reduce_mean(tmp)

        return rot_loss

    @staticmethod
    def normalize(target, norm_mean, norm_std):
        target_trans, target_rot = tf.split(target, [3, 4], axis=1)
        target_trans_centered = target_trans - tf.tile(tf.reshape(norm_mean, [1, 3]),
                                                                     [tf.shape(target_trans)[0], 1])
        target_trans_normed = target_trans_centered / tf.tile(tf.reshape(norm_std, [1, 3]),
                                                                            [tf.shape(target_trans)[0], 1])
        target_normed = tf.concat([target_trans_normed, target_rot], axis=1)

        return target_normed

    @staticmethod
    def denomalize(normed_target, norm_mean, norm_std):
        normed_target_trans, normed_target_rot = tf.split(normed_target, [3, 4], axis=1)
        target_trans_unscaled = normed_target_trans * tf.tile(tf.reshape(norm_std, [1, 3]), [tf.shape(normed_target_trans)[0], 1])
        target_trans_uncentered = target_trans_unscaled + tf.tile(tf.reshape(norm_mean, [1, 3]),
                                                                  [tf.shape(normed_target_trans)[0], 1])

        target = tf.concat([target_trans_uncentered, normed_target_rot], axis=1)
        return target

    @staticmethod
    def denomalize_navie(normed_target, norm_mean, norm_std):

        target_trans_unscaled = normed_target * tf.tile(tf.reshape(norm_std, [1, 3]),
                                                              [tf.shape(normed_target)[0], 1])
        target_uncentered = target_trans_unscaled + tf.tile(tf.reshape(norm_mean, [1, 3]),
                                                                  [tf.shape(normed_target)[0], 1])

        return target_uncentered

    def get_relative_pose(self, Q_a, Q_b):

        M_a = self.quanternion2matrix(Q_a)
        M_b = self.quanternion2matrix(Q_b)

        [M_a, M_b] = tf.cond(tf.logical_and(tf.reduce_any(tf.is_nan(M_a)), tf.reduce_any(tf.is_nan(M_b))), lambda: [tf.eye(4, batch_shape=[M_a.shape[0]]), tf.eye(4, batch_shape=[M_b.shape[0]])], lambda: [M_a, M_b])

        try:
            M_delta = tf.map_fn(lambda x: tf.linalg.matmul(tf.linalg.inv(x[0]), x[1]), (M_a, M_b), dtype=tf.float32)
        except ValueError:
            print("matrix is not invertiable")
            M_delta = tf.eye(4, batch_shape=M_a.shape[0])


        Q_delta = self.matrix2quternion(M_delta)

        # self.M_a = M_a
        # self.M_b = M_b
        # self.M_delta = M_delta
        # self.Q_delta = Q_delta

        return Q_delta


    def quanternion2matrix(self, q):

        with tf.variable_scope('geometry'):

            tx, ty, tz, qx, qy, qz, qw = tf.split(q, [1, 1, 1, 1, 1, 1, 1], axis=-1)


            M11 = 1.0 - 2 * (tf.square(qy) + tf.square(qz))
            M12 = 2. * qx * qy - 2. * qw * qz
            M13 = 2. * qw * qy + 2. * qx * qz
            M14 = tx

            M21 = 2. * qx * qy + 2. * qw * qz
            M22 = 1. - 2. * (tf.square(qx) + tf.square(qz))
            M23 = -2. * qw * qx + 2. * qy * qz
            M24 = ty

            M31 = -2. * qw * qy + 2. * qx * qz
            M32 = 2. * qw * qx + 2. * qy * qz
            M33 = 1. - 2. * (tf.square(qx) + tf.square(qy))
            M34 = tz

            M41 = tf.zeros_like(M11)
            M42 = tf.zeros_like(M11)
            M43 = tf.zeros_like(M11)
            M44 = tf.ones_like(M11)

            M11 = tf.expand_dims(M11, axis=-1)
            M12 = tf.expand_dims(M12, axis=-1)
            M13 = tf.expand_dims(M13, axis=-1)
            M14 = tf.expand_dims(M14, axis=-1)

            M21 = tf.expand_dims(M21, axis=-1)
            M22 = tf.expand_dims(M22, axis=-1)
            M23 = tf.expand_dims(M23, axis=-1)
            M24 = tf.expand_dims(M24, axis=-1)

            M31 = tf.expand_dims(M31, axis=-1)
            M32 = tf.expand_dims(M32, axis=-1)
            M33 = tf.expand_dims(M33, axis=-1)
            M34 = tf.expand_dims(M34, axis=-1)

            M41 = tf.expand_dims(M41, axis=-1)
            M42 = tf.expand_dims(M42, axis=-1)
            M43 = tf.expand_dims(M43, axis=-1)
            M44 = tf.expand_dims(M44, axis=-1)

            M_l1 = tf.concat([M11, M12, M13, M14], axis=2)
            M_l2 = tf.concat([M21, M22, M23, M24], axis=2)
            M_l3 = tf.concat([M31, M32, M33, M34], axis=2)
            M_l4 = tf.concat([M41, M42, M43, M44], axis=2)

            M = tf.concat([M_l1, M_l2, M_l3, M_l4], axis=1)

        return M

    def matrix2quternion(self, M):
        tx = tf.reshape(M[:, 0, 3], [-1, 1])
        ty = tf.reshape(M[:, 1, 3], [-1, 1])
        tz = tf.reshape(M[:, 2, 3], [-1, 1])
        qw = tf.reshape(0.5 * tf.sqrt(M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2] + M[:, 3, 3]), [-1, 1])
        qx = tf.reshape(M[:, 2, 1] - M[:, 1, 2], [-1, 1]) / (4. * qw)
        qy = tf.reshape(M[:, 0, 2] - M[:, 2, 0], [-1, 1]) / (4. * qw)
        qz = tf.reshape(M[:, 1, 0] - M[:, 0, 1], [-1, 1]) / (4. * qw)

        q = tf.concat([tx, ty, tz, qx, qy, qz, qw], axis=-1)
        return q









