import tensorflow as tf
from src.self_awareness import networks


class Model():

    def __init__(self, args, is_training=True):

        # Store the arguments
        self.args = args

        # args.rnn_size contains the dimension of the hidden state of the LSTM

        # TODO: (resolve) Do we need to use a fixed seq_length?
        # Input data contains sequence of (x,y) points
        self.input_data = tf.placeholder(tf.float32, [None, args.target_image_size[0], args.target_image_size[1], 1])
        # target data contains sequences of (x,y) points as well
        self.target_data = tf.placeholder(tf.float32, [None, 7])
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
        output, _, _ = network.build(self.input_data, is_training)

        trans_pred, rot_pred = tf.split(output, [3, 4], axis=1)
        trans_target, rot_target = tf.split(target_normed, [3, 4], axis=1)

        self.trans_prediction2 = trans_pred

        if args.is_normalization:
            target_trans_unscaled = trans_pred * tf.tile(tf.reshape(self.norm_std, [1, 3]), [tf.shape(trans_pred)[0], 1])
            target_trans_uncentered = target_trans_unscaled + tf.tile(tf.reshape(self.norm_mean, [1, 3]), [tf.shape(trans_pred)[0], 1])

            self.trans_prediction = target_trans_uncentered
            self.rot_prediction = rot_pred
        else:
            self.trans_prediction = trans_pred
            self.rot_prediction = rot_pred

        if is_training:

            self.trans_loss = tf.losses.mean_squared_error(labels=trans_target, predictions=trans_pred)
            self.rot_loss0 = tf.losses.mean_squared_error(labels=rot_target, predictions=rot_pred)

            rot_loss_1 = tf.reduce_mean(tf.square(rot_pred - rot_target), axis=1)
            rot_loss_2 = tf.reduce_mean(tf.square(rot_pred + rot_target), axis=1)

            tmp = tf.stack([rot_loss_1, rot_loss_2], axis=1)
            tmp = tf.reduce_min(tmp, axis=1)
            self.rot_loss = tf.reduce_mean(tmp)

            # Learning rate
            self.lr = tf.Variable(args.learning_rate, trainable=False, name="learning_rate")
            # Global step
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            # lamda weight
            self.lamda_weights = tf.Variable(args.lamda_weights, trainable=False, name="category_weights")

            with tf.variable_scope('param'):
                self.lamda_weights = tf.clip_by_value(tf.Variable(args.lamda_weights, trainable=True, name="category_weights"), -5.0, 5.0)
                self.lamda_weights_sigmoid = tf.math.sigmoid(self.lamda_weights)

            self.total_loss = self.trans_loss + self.rot_loss * self.lamda_weights

            tvars = tf.trainable_variables()
            optimizer = tf.train.AdamOptimizer(self.lr)
            grad_vars = optimizer.compute_gradients(loss=self.total_loss, var_list=tvars)

            # cnn_tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vggnet_localization')
            # param_tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='param')
            #
            # param_optimizer = tf.train.AdamOptimizer(self.lr*10.0)
            # param_grad_vars = param_optimizer.compute_gradients(loss=-self.total_loss, var_list=param_tvars)
            #
            # cnn_optimizer = tf.train.AdamOptimizer(self.lr)
            # cnn_grad_vars = cnn_optimizer.compute_gradients(loss=self.total_loss, var_list=cnn_tvars)

            self.train_op = optimizer.apply_gradients(grad_vars)

            # gradients = tf.gradients(self.total_loss, tvars)
            # # Clip the gradients if they are larger than the value given in args
            # grads, _ = tf.clip_by_global_norm(gradients, args.grad_clip)
            # optimizer = tf.train.AdamOptimizer(self.lr)
            # self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)



    def evalute(self, sess):

        with tf.variable_scope('metrics/'):
            with tf.variable_scope('validation', reuse=True):
                self.trans_error, self.trans_update_op = tf.metrics.root_mean_squared_error(labels=self.trans_target, predictions=self.trans_pred, name='trans_error')
                self.rot_error, self.rot_update_op = tf.metrics.root_mean_squared_error(labels=self.rot_target, predictions=self.rot_pred, name='rot_error')

                error_vars = tf.contrib.framework.get_local_variables(scope='metrics/{}'.format('validation'))
                self.reset_op = tf.variables_initializer(var_list=error_vars)







