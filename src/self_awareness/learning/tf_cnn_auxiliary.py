import tensorflow as tf
from src.self_awareness import networks

#from tensorflow.contrib import slim
#from tensorflow.contrib.slim.python.slim.nets import resnet_v1

class Model():

    def __init__(self, args, is_training=True):

        # Store the arguments
        self.args = args

        self.norm_mean = tf.Variable(args.norm_mean, dtype=tf.float32, trainable=False, name="norm_mean")
        self.norm_std = tf.Variable(args.norm_std, dtype=tf.float32, trainable=False, name="norm_std")

        if not is_training:
            args.batch_size = 1

        # TODO: (resolve) Do we need to use a fixed seq_length?
        # input_data_t_1 refers to data at time <t-1>
        self.input_data_t0 = tf.placeholder(tf.float32, [args.batch_size, args.target_image_size[0], args.target_image_size[1], 1])
        self.input_data_t1 = tf.placeholder(tf.float32, [args.batch_size, args.target_image_size[0], args.target_image_size[1], 1])
        # target data (global poses)
        self.target_data_t0 = tf.placeholder(tf.float32, [args.batch_size, 7])
        self.target_data_t1 = tf.placeholder(tf.float32, [args.batch_size, 7])

        # self.global_trans_target, self.global_rot_target = tf.split(self.global_target, [3, 4], axis=1)

        self.relative_target = self.get_relative_pose(self.target_data_t0, self.target_data_t1)
        # self.relative_trans_target, self.relative_rot_target = tf.split(self.relative_target, [3, 4], axis=1)

        target_data_t0_normed = self.normalize(self.target_data_t0, self.norm_mean, self.norm_std)
        target_data_t1_normed = self.normalize(self.target_data_t1, self.norm_mean, self.norm_std)
        relative_target_normed = self.get_relative_pose(target_data_t0_normed, target_data_t1_normed)

        #with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        # extract dense feature at time <t0>
        dense_feat0, _ = networks.resnet.resnet_v1_50(self.input_data_t0, global_pool=False, num_classes=None,
                                                         is_training=is_training, reuse=tf.AUTO_REUSE, scope="dense_feat")

        # extrat dense feature at time <t1>
        dense_feat1, _ = networks.resnet.resnet_v1_50(self.input_data_t1, global_pool=False, num_classes=None,
                                                         is_training=is_training, reuse=tf.AUTO_REUSE, scope="dense_feat")

        # dense_feat0, _ = networks.densenet.densenet121(self.input_data_t0, num_classes=None,
        #                                               is_training=is_training, is_global_pool=False, reuse=tf.AUTO_REUSE, scope="dense_feat")
        #
        # dense_feat1, _ = networks.densenet.densenet121(self.input_data_t1, num_classes=None,
        #                                               is_training=is_training, is_global_pool=False, reuse=True, scope="dense_feat")


        # dense_feat_net = networks.catalogue[args.network](args, name="dense_feat")
        # dense_feat0 = dense_feat_net.build(self.input_data_t0, is_training, opt="base")
        # dense_feat1 = dense_feat_net.build(self.input_data_t1, is_training, opt="base")

        # concat temporal features for relative pose estimation
        dense_feat_relative = tf.concat([dense_feat0, dense_feat1], axis=-1)

        # context stack for global pose learning
        global_context = networks.catalogue[args.network](args, name="context_stack_global")
        global_context_feat0 = global_context.build(dense_feat0, is_training, opt="context")
        global_context_feat1 = global_context.build(dense_feat1, is_training, opt="context")

        # context stack for relative pose learning
        relative_context = networks.catalogue[args.network](args, name="context_stack_relative")
        relative_context_feat = relative_context.build(dense_feat_relative, is_training, opt="context")

        # regreesor for global pose learning
        global_regressor = networks.catalogue[args.network](args, name="regressor_global")
        global_output0, _, _ = global_regressor.build(global_context_feat0, is_training, opt="regressor")
        global_output1, _, _ = global_regressor.build(global_context_feat1, is_training, opt="regressor")

        # ############################## EXP ###########################################
        # global_output0 = target_data_t0_normed
        # global_output1 = target_data_t1_normed
        # ############################################################################

        relative_consistence = self.get_relative_pose(global_output0, global_output1)

        # regressor for relative pose learning
        relative_regressor = networks.catalogue[args.network](args, name="regressor_relative")
        relative_output, _, _ = relative_regressor.build(relative_context_feat, is_training, opt="regressor")

        if is_training:
            # Learning rate
            self.lr = tf.Variable(args.learning_rate, trainable=False, name="learning_rate")
            # Global step
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            # lamda weight
            self.lamda_weights = tf.Variable(args.lamda_weights, trainable=False, name="category_weights")

            # Global pose loss
            self.global_loss, self.global_trans_loss, self.global_rot_loss = \
                self.translational_rotational_loss(pred=global_output1, gt=target_data_t1_normed, lamda=self.lamda_weights)

            # Relative pose loss
            self.relative_loss, self.relative_trans_loss, self.relative_rot_loss = \
                self.translational_rotational_loss(pred=relative_output, gt=relative_target_normed, lamda=self.lamda_weights)

            # Geometry consistence loss
            self.geometry_consistent_loss, self.consis_trans_loss, self.consis_rot_loss = \
                self.translational_rotational_loss(pred=relative_consistence, gt=relative_target_normed, lamda=self.lamda_weights)


            self.total_loss = self.global_loss + 0.1 * self.geometry_consistent_loss # + self.relative_loss

            tvars = tf.trainable_variables()
            optimizer = tf.train.AdamOptimizer(self.lr)
            grad_vars = optimizer.compute_gradients(loss=self.total_loss, var_list=tvars)

            self.train_op = optimizer.apply_gradients(grad_vars, global_step=self.global_step)

        else:

            self.trans_target, self.rot_target = tf.split(self.target_data_t1, [3, 4], axis=1)
            global_output1_demormed = self.denomalize(global_output1, self.norm_mean, self.norm_std)
            self.trans_prediction, self.rot_prediction = tf.split(global_output1_demormed, [3, 4], axis=1)

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

        mask = tf.cast(tf.math.less_equal(tf.math.abs(qw), 10e-6 * tf.ones_like(qw)), tf.int32)
        qw = tf.cond(tf.math.equal(tf.math.count_nonzero(mask), mask.shape[0]*mask.shape[1]), lambda: qw, lambda: qw+10e-6*tf.ones_like(qw))

        qx = tf.reshape(M[:, 2, 1] - M[:, 1, 2], [-1, 1]) / (4. * qw)
        qy = tf.reshape(M[:, 0, 2] - M[:, 2, 0], [-1, 1]) / (4. * qw)
        qz = tf.reshape(M[:, 1, 0] - M[:, 0, 1], [-1, 1]) / (4. * qw)

        q = tf.concat([tx, ty, tz, qx, qy, qz, qw], axis=-1)
        return q



class Model2():

    def __init__(self, args, is_training=True):

        # Store the arguments
        self.args = args

        self.norm_mean = tf.Variable(args.norm_mean, dtype=tf.float16, trainable=False, name="norm_mean")
        self.norm_std = tf.Variable(args.norm_std, dtype=tf.float16, trainable=False, name="norm_std")

        if not is_training:
            args.batch_size = 1

        # TODO: (resolve) Do we need to use a fixed seq_length?
        # input_data_t_1 refers to data at time <t-1>
        self.input_data_t0 = tf.placeholder(tf.float16, [args.batch_size, args.target_image_size[0], args.target_image_size[1], 1])
        self.input_data_t1 = tf.placeholder(tf.float16, [args.batch_size, args.target_image_size[0], args.target_image_size[1], 1])
        # target data (global poses)
        self.target_data_t0 = tf.placeholder(tf.float16, [args.batch_size, 7])
        self.target_data_t1 = tf.placeholder(tf.float16, [args.batch_size, 7])

        # self.global_trans_target, self.global_rot_target = tf.split(self.global_target, [3, 4], axis=1)

        self.relative_target = self.get_relative_pose(self.target_data_t0, self.target_data_t1)
        # self.relative_trans_target, self.relative_rot_target = tf.split(self.relative_target, [3, 4], axis=1)

        target_data_t0_normed = self.normalize(self.target_data_t0, self.norm_mean, self.norm_std)
        target_data_t1_normed = self.normalize(self.target_data_t1, self.norm_mean, self.norm_std)
        relative_target_normed = self.get_relative_pose(target_data_t0_normed, target_data_t1_normed)

        # extract dense feature at time <t0>
        dense_feat0, _ = networks.resnet.resnet_v1_50(self.input_data_t0, global_pool=False, num_classes=None,
                                                         is_training=is_training, reuse=tf.AUTO_REUSE, scope="dense_feat")

        # extrat dense feature at time <t1>
        dense_feat1, _ = networks.resnet.resnet_v1_50(self.input_data_t1, global_pool=False, num_classes=None,
                                                         is_training=is_training, reuse=tf.AUTO_REUSE, scope="dense_feat")

        # dense_feat0, _ = networks.densenet.densenet121(self.input_data_t0, num_classes=None,
        #                                               is_training=is_training, is_global_pool=False, reuse=tf.AUTO_REUSE, scope="dense_feat")
        #
        # dense_feat1, _ = networks.densenet.densenet121(self.input_data_t1, num_classes=None,
        #                                               is_training=is_training, is_global_pool=False, reuse=True, scope="dense_feat")


        # dense_feat_net = networks.catalogue[args.network](args, name="dense_feat")
        # dense_feat0 = dense_feat_net.build(self.input_data_t0, is_training, opt="base")
        # dense_feat1 = dense_feat_net.build(self.input_data_t1, is_training, opt="base")

        # concat temporal features for relative pose estimation
        dense_feat_relative = tf.concat([dense_feat0, dense_feat1], axis=-1)

        # context stack for global pose learning
        global_context = networks.catalogue[args.network](args, name="context_stack_global")
        global_context_feat0 = global_context.build(dense_feat0, is_training, opt="context")
        global_context_feat1 = global_context.build(dense_feat1, is_training, opt="context")

        # context stack for relative pose learning
        relative_context = networks.catalogue[args.network](args, name="context_stack_relative")
        relative_context_feat = relative_context.build(dense_feat_relative, is_training, opt="context")

        # regreesor for global pose learning
        global_regressor = networks.catalogue[args.network](args, name="regressor_global")
        global_output0, _, _ = global_regressor.build(global_context_feat0, is_training, opt="regressor")
        global_output1, _, _ = global_regressor.build(global_context_feat1, is_training, opt="regressor")

        # ############################## EXP ###########################################
        # global_output0 = target_data_t0_normed
        # global_output1 = target_data_t1_normed
        # ############################################################################

        relative_consistence = self.get_relative_pose(global_output0, global_output1)

        # regressor for relative pose learning
        relative_regressor = networks.catalogue[args.network](args, name="regressor_relative")
        relative_output, _, _ = relative_regressor.build(relative_context_feat, is_training, opt="regressor")

        if is_training:
            # Learning rate
            self.lr = tf.Variable(args.learning_rate, trainable=False, name="learning_rate")
            # Global step
            self.global_step = tf.Variable(0, trainable=False, name="global_step")
            # lamda weight
            self.lamda_weights = tf.Variable(args.lamda_weights, trainable=False, name="category_weights")

            # Global pose loss
            self.global_loss, self.global_trans_loss, self.global_rot_loss = \
                self.translational_rotational_loss(pred=global_output1, gt=target_data_t1_normed, lamda=self.lamda_weights)

            # Relative pose loss
            self.relative_loss, self.relative_trans_loss, self.relative_rot_loss = \
                self.translational_rotational_loss(pred=relative_output, gt=relative_target_normed, lamda=self.lamda_weights)

            # Geometry consistence loss
            self.geometry_consistent_loss, self.consis_trans_loss, self.consis_rot_loss = \
                self.translational_rotational_loss(pred=relative_consistence, gt=relative_target_normed, lamda=self.lamda_weights)


            self.total_loss = self.global_loss + 0.1 * self.geometry_consistent_loss # + self.relative_loss

            tvars = tf.trainable_variables()
            optimizer = tf.train.AdamOptimizer(self.lr)
            grad_vars = optimizer.compute_gradients(loss=self.total_loss, var_list=tvars)

            self.train_op = optimizer.apply_gradients(grad_vars, global_step=self.global_step)

        else:

            self.trans_target, self.rot_target = tf.split(self.target_data_t1, [3, 4], axis=1)
            global_output1_demormed = self.denomalize(global_output1, self.norm_mean, self.norm_std)
            self.trans_prediction, self.rot_prediction = tf.split(global_output1_demormed, [3, 4], axis=1)

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

    def get_relative_pose(self, Q_a, Q_b):

        M_a = self.quanternion2matrix(Q_a)
        M_b = self.quanternion2matrix(Q_b)

        [M_a, M_b] = tf.cond(tf.logical_and(tf.reduce_any(tf.is_nan(M_a)), tf.reduce_any(tf.is_nan(M_b))), lambda: [tf.eye(4, batch_shape=[M_a.shape[0]], dtype=tf.float16), tf.eye(4, batch_shape=[M_b.shape[0]], dtype=tf.float16)], lambda: [M_a, M_b])

        try:
            M_delta = tf.map_fn(lambda x: tf.linalg.matmul(tf.linalg.inv(x[0]), x[1]), (M_a, M_b), dtype=tf.float16)
        except ValueError:
            print("matrix is not invertiable")
            M_delta = tf.eye(4, batch_shape=M_a.shape[0], dtype=tf.float16)


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

        mask = tf.cast(tf.math.less_equal(tf.math.abs(qw), 10e-6 * tf.ones_like(qw)), tf.int32)
        qw = tf.cond(tf.math.equal(tf.math.count_nonzero(mask), mask.shape[0]*mask.shape[1]), lambda: qw, lambda: qw+10e-6*tf.ones_like(qw))

        qx = tf.reshape(M[:, 2, 1] - M[:, 1, 2], [-1, 1]) / (4. * qw)
        qy = tf.reshape(M[:, 0, 2] - M[:, 2, 0], [-1, 1]) / (4. * qw)
        qz = tf.reshape(M[:, 1, 0] - M[:, 0, 1], [-1, 1]) / (4. * qw)

        q = tf.concat([tx, ty, tz, qx, qy, qz, qw], axis=-1)
        return q





