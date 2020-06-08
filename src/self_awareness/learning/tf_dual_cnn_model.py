import tensorflow as tf
from src.self_awareness import networks


class Model():

    def __init__(self, args, is_training=True):

        # Store the arguments
        self.args = args

        # args.rnn_size contains the dimension of the hidden state of the LSTM

        # TODO: (resolve) Do we need to use a fixed seq_length?
        # input_data_t_1 refers to data at time <t-1>
        self.input_data_t_1 = tf.placeholder(tf.float32, [args.batch_size, args.target_image_size[0], args.target_image_size[1], 1])
        self.input_data_t = tf.placeholder(tf.float32, [args.batch_size, args.target_image_size[0], args.target_image_size[1], 1])
        # target data (global poses)
        self.target_data_t_1 = tf.placeholder(tf.float32, [args.batch_size, 7])
        self.global_target = self.target_data_t = tf.placeholder(tf.float32, [args.batch_size, 7])

        self.global_trans_target, self.global_rot_target = tf.split(self.global_target, [3, 4], axis=1)

        self.relative_target = self.get_relative_pose(self.target_data_t_1, self.target_data_t)
        self.relative_trans_target, self.relative_rot_target = tf.split(self.relative_target, [3, 4], axis=1)

        if args.is_normalization:
            self.norm_mean = tf.Variable(args.norm_mean, dtype=tf.float32, trainable=False, name="norm_mean")
            self.norm_std = tf.Variable(args.norm_std, dtype=tf.float32, trainable=False, name="norm_std")

            global_target_trans, global_target_rot = tf.split(self.global_target, [3, 4], axis=1)
            global_target_trans_centered = global_target_trans - tf.tile(tf.reshape(self.norm_mean, [1, 3]), [tf.shape(global_target_trans)[0], 1])
            global_target_trans_normed = global_target_trans_centered / tf.tile(tf.reshape(self.norm_std, [1, 3]), [tf.shape(global_target_trans)[0], 1])
            global_target_normed = tf.concat([global_target_trans_normed, global_target_rot], axis=1)
        else:
            global_target_normed = self.global_target_t

        # extract vgg feature at time <t>
        network_vgg_t = networks.catalogue[args.network](args, name="vgg_net_t")
        vgg_feat_t = network_vgg_t.build(self.input_data_t, is_training, opt="base")

        # extrat vgg feature at time <t+1
        network_vgg_t_1 = networks.catalogue[args.network](args, name="vgg_net_t_1")
        vgg_feat_t_1 = network_vgg_t_1.build(self.input_data_t_1, is_training, opt="base")

        # concat temporal features for relative pose estimation
        vgg_feat_relative = tf.concat([vgg_feat_t, vgg_feat_t_1], axis=-1)

        # context stack for global pose learning
        global_context = networks.catalogue[args.network](args, name="context_stack_global")
        global_context_feat = global_context.build(vgg_feat_t, is_training, opt="context")

        # context stack for relative pose learning
        relative_context = networks.catalogue[args.network](args, name="context_stack_relative")
        relative_context_feat = relative_context.build(vgg_feat_relative, is_training, opt="context")

        # regreesor for global pose learning
        global_regressor = networks.catalogue[args.network](args, name="regressor_global")
        global_output, _, _ = global_regressor.build(global_context_feat, is_training, opt="regressor")

        # regressor for relative pose learning
        relative_regressor = networks.catalogue[args.network](args, name="regressor_relative")
        relative_output, _, _ = relative_regressor.build(relative_context_feat, is_training, opt="regressor")

        global_trans_pred, global_rot_pred = tf.split(global_output, [3, 4], axis=1)
        global_trans_target, global_rot_target = tf.split(global_target_normed, [3, 4], axis=1)

        self.relative_trans_pred, self.relative_rot_pred = tf.split(relative_output, [3, 4], axis=1)

        if is_training:

            # Global pose loss
            self.global_trans_loss = tf.losses.mean_squared_error(labels=global_trans_target, predictions=global_trans_pred)

            global_rot_loss_1 = tf.reduce_mean(tf.square(global_rot_pred - global_rot_target), axis=1)
            global_rot_loss_2 = tf.reduce_mean(tf.square(global_rot_pred + global_rot_target), axis=1)

            global_tmp = tf.stack([global_rot_loss_1, global_rot_loss_2], axis=1)
            global_tmp = tf.reduce_min(global_tmp, axis=1)
            self.global_rot_loss = tf.reduce_mean(global_tmp)

            # Relative pose loss
            self.relative_trans_loss = tf.losses.mean_squared_error(labels=self.relative_trans_target, predictions=self.relative_trans_pred)

            relative_rot_loss_1 = tf.reduce_mean(tf.square(self.relative_rot_pred - self.relative_rot_target), axis=1)
            relative_rot_loss_2 = tf.reduce_mean(tf.square(self.relative_rot_pred + self.relative_rot_target), axis=1)

            relative_tmp = tf.stack([relative_rot_loss_1, relative_rot_loss_2], axis=1)
            relative_tmp = tf.reduce_min(relative_tmp, axis=1)
            self.relative_rot_loss = tf.reduce_mean(relative_tmp)

            # Learning rate
            self.lr = tf.Variable(args.learning_rate, trainable=False, name="learning_rate")
            # Global step
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            # lamda weight
            self.lamda_weights = tf.Variable(args.lamda_weights, trainable=False, name="category_weights")

            self.global_loss = self.global_trans_loss + self.global_rot_loss * self.lamda_weights
            self.relative_loss = self.relative_trans_loss + self.relative_rot_loss * self.lamda_weights
            self.total_loss = self.global_loss + self.relative_loss

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

        else:
            pass


    def get_relative_pose(self, Q_a, Q_b):

        # M_a = self.quanternion2matrix(Q_a)
        # M_b = self.quanternion2matrix(Q_b)
        #
        # self.M_a = M_a
        # self.M_b = M_b

        # M_delta = tf.map_fn(lambda x: tf.linalg.matmul(tf.linalg.inv(x[0]), x[1]), (M_a, M_b), dtype=tf.float32)
        #
        # self.M_delta = M_delta
        # self.Q_delta = self.matrix2quternion(self.M_delta)

        # M_a = tf.unstack(M_a, axis=0)
        # M_b = tf.unstack(M_b, axis=0)
        #
        # M_delta = []
        #
        # for m_a, m_b in zip(M_a, M_b):
        #
        #     m_delta = tf.linalg.matmul(tf.linalg.inv(m_a), m_b)
        #
        #     M_delta.append(m_delta)
        #
        # self.M_delta = tf.stack(M_delta, axis=0)
        # self.Q_delta = self.matrix2quternion(self.M_delta)


        Q_a = tf.unstack(Q_a, axis=0)
        Q_b = tf.unstack(Q_b, axis=0)

        Q_delta = []
        M_a = []
        M_b = []
        M_delta = []

        for q_a, q_b in zip(Q_a, Q_b):

            m_a = self.quanternion2matrix(q_a)
            m_b = self.quanternion2matrix(q_b)

            m_delta = tf.linalg.matmul(tf.linalg.inv(m_a), m_b)

            q_delta = self.matrix2quternion(m_delta)

            Q_delta.append(q_delta)
            M_delta.append(m_delta)
            M_a.append(m_a)
            M_b.append(m_b)

        Q_delta = tf.stack(Q_delta, axis=0)
        M_delta = tf.stack(M_delta, axis=0)
        M_a = tf.stack(M_a, axis=0)
        M_b = tf.stack(M_b, axis=0)

        self.M_a = M_a
        self.M_b = M_b
        self.M_delta = M_delta
        self.Q_delta = Q_delta

        return self.Q_delta


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

            M_l1 = tf.concat([M11, M12, M13, M14], axis=1)
            M_l2 = tf.concat([M21, M22, M23, M24], axis=1)
            M_l3 = tf.concat([M31, M32, M33, M34], axis=1)
            M_l4 = tf.concat([M41, M42, M43, M44], axis=1)

            M = tf.concat([M_l1, M_l2, M_l3, M_l4], axis=0)

        return M

    def matrix2quternion(self, M):
        # tx = tf.reshape(M[:, 0, 3], [-1, 1])
        # ty = tf.reshape(M[:, 1, 3], [-1, 1])
        # tz = tf.reshape(M[:, 2, 3], [-1, 1])
        # qw = tf.reshape(0.5 * tf.sqrt(M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2] + M[:, 3, 3]), [-1, 1])
        # qx = tf.reshape(M[:, 2, 1] - M[:, 1, 2], [-1, 1]) / (4. * qw)
        # qy = tf.reshape(M[:, 0, 2] - M[:, 2, 0], [-1, 1]) / (4. * qw)
        # qz = tf.reshape(M[:, 1, 0] - M[:, 0, 1], [-1, 1]) / (4. * qw)

        tx = tf.reshape(M[0, 3], [1])
        ty = tf.reshape(M[1, 3], [1])
        tz = tf.reshape(M[2, 3], [1])
        qw = tf.reshape(0.5 * tf.sqrt(M[0, 0] + M[1, 1] + M[2, 2] + M[3, 3]), [1])
        qx = tf.reshape(M[2, 1] - M[1, 2], [1]) / (4. * qw)
        qy = tf.reshape(M[0, 2] - M[2, 0], [1]) / (4. * qw)
        qz = tf.reshape(M[1, 0] - M[0, 1], [1]) / (4. * qw)

        q = tf.concat([tx, ty, tz, qx, qy, qz, qw], axis=-1)
        return q



    def evalute(self, sess):

        with tf.variable_scope('metrics/'):
            with tf.variable_scope('validation', reuse=True):
                self.trans_error, self.trans_update_op = tf.metrics.root_mean_squared_error(labels=self.trans_target, predictions=self.trans_pred, name='trans_error')
                self.rot_error, self.rot_update_op = tf.metrics.root_mean_squared_error(labels=self.rot_target, predictions=self.rot_pred, name='rot_error')

                error_vars = tf.contrib.framework.get_local_variables(scope='metrics/{}'.format('validation'))
                self.reset_op = tf.variables_initializer(var_list=error_vars)







