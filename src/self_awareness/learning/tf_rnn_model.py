import tensorflow as tf
from src.self_awareness import networks
import numpy as np
import math
from tensorflow.contrib.layers import fully_connected



class Model():

    def __init__(self, args, is_training=True):

        # infer mode
        if not is_training:
            # Infer one position at a time
            args.batch_size = 1
            args.max_seq_length = 1
            args.keep_prob = 1.0

        # Store the arguments
        self.args = args

        with tf.variable_scope("seq2seq"):

            if args.model == 'rnn':
                cells = [tf.contrib.rnn.RNNCell(args.rnn_size) for i in range(args.num_layers)]
            elif args.model == 'gru':
                cells = [tf.contrib.rnn.GRUCell(args.rnn_size) for i in range(args.num_layers)]
            else: # LSTM as the default
                t_cells = [tf.contrib.rnn.LSTMCell(args.rnn_size, initializer=tf.contrib.layers.xavier_initializer(), state_is_tuple=True) for i in range(args.num_layers)]
                r_cells = [tf.contrib.rnn.LSTMCell(args.rnn_size, initializer=tf.contrib.layers.xavier_initializer(), state_is_tuple=True) for i in range(args.num_layers)]

            # Multi-layer RNN construction, if more than one layer
            trans_cells = tf.contrib.rnn.MultiRNNCell(t_cells, state_is_tuple=True)
            rot_cells = tf.contrib.rnn.MultiRNNCell(r_cells, state_is_tuple=True)

        # Store the recurrent unit
        self.trans_cells = trans_cells
        self.rot_cells = rot_cells

        # Input data contains sequence of (x,y) points
        self.input_data = tf.placeholder(tf.float32, [None, args.max_seq_length, args.target_image_size[0], args.target_image_size[1], 1])
        # target data contains sequences of (x,y) points as well
        self.target_data = tf.placeholder(tf.float32, [None, args.max_seq_length, 7])

        if args.is_normalization:
            self.norm_mean = tf.Variable(args.norm_mean, dtype=tf.float32, trainable=False, name="norm_mean")
            self.norm_std = tf.Variable(args.norm_std, dtype=tf.float32, trainable=False, name="norm_std")

            target_trans, target_rot = tf.split(self.target_data, [3, 4], axis=2)
            target_trans_centered = target_trans - tf.tile(tf.reshape(self.norm_mean, [1, 1, 3]), [tf.shape(target_trans)[0], args.max_seq_length, 1])
            target_trans_normed = target_trans_centered / tf.tile(tf.reshape(self.norm_std, [1, 1, 3]), [tf.shape(target_trans)[0], args.max_seq_length, 1])
            target_normed = tf.concat([target_trans_normed, target_rot], axis=2)
        else:
            target_normed = self.target_data


        if is_training:
            self.seq_len = tf.placeholder(tf.int64, [None])
            self.index = tf.placeholder(tf.int64, [None])

        # Learning rate
        self.lr = tf.Variable(args.learning_rate, trainable=False, name="learning_rate")

        # Global step
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        # Initial cell state of the LSTM (initialised with zeros)
        self.initial_state = (trans_cells.zero_state(batch_size=args.batch_size, dtype=tf.float32), rot_cells.zero_state(batch_size=args.batch_size, dtype=tf.float32))

        # Split inputs according to sequences.
        inputs = tf.unstack(self.input_data, axis=1)

        # Feature embedding through CNN
        # choose CNN
        network = networks.catalogue[args.network](args)

        inputs_trans = []
        inputs_rot = []
        for x in inputs:
            _, feat_t, feat_r = network.build(x, is_training)
            inputs_trans.append(feat_t)
            inputs_rot.append(feat_r)

        with tf.variable_scope("seq2seq"):
        # Feed the embedded input data, the initial state of the LSTM cell, the recurrent unit to the seq2seq decoder
            trans_outputs, last_trans_state = tf.contrib.legacy_seq2seq.rnn_decoder(inputs_trans, self.initial_state[0], trans_cells, loop_function=None, scope="seq2seq_trans")
            rot_outputs, last_rot_state = tf.contrib.legacy_seq2seq.rnn_decoder(inputs_rot, self.initial_state[1], rot_cells, loop_function=None, scope="seq2seq_rot")

            # Concatenate the outputs from the RNN decoder and reshape it to args.rnn_size
            trans_outputs = tf.reshape(tf.concat(trans_outputs, axis=1), [-1, args.rnn_size])
            trans_pred = fully_connected(trans_outputs, 3, activation_fn=None, scope='logits_t')
            rot_outputs = tf.reshape(tf.concat(rot_outputs, axis=1), [-1, args.rnn_size])
            rot_pred = fully_connected(rot_outputs, 4, activation_fn=None, scope='logits_r')


        # Store the final LSTM cell state after the input data has been feeded
        self.final_state = (last_trans_state, last_rot_state)

        # reshape target data so that it aligns with predictions
        flat_target_data = tf.reshape(target_normed, [-1, 7])
        trans_pred = tf.gather(trans_pred, self.index)

        if args.is_normalization:
            target_trans_unscaled = trans_pred * tf.tile(tf.reshape(self.norm_std, [1, 3]), [tf.shape(trans_pred)[0], 1])
            target_trans_uncentered = target_trans_unscaled + tf.tile(tf.reshape(self.norm_mean, [1, 3]), [tf.shape(trans_pred)[0], 1])

            self.trans_prediction = target_trans_uncentered
            self.rot_prediction = rot_pred
        else:
            self.trans_prediction = trans_pred
            self.rot_prediction = rot_pred

        rot_pred = tf.gather(rot_pred, self.index)
        flat_target_data = tf.gather(flat_target_data, self.index)
        trans_target, rot_target = tf.split(flat_target_data, [3, 4], axis=1)

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
            self.lamda_weights = tf.Variable(args.lamda_weights, trainable=False, name="category_weights")

            self.total_loss = self.trans_loss + self.rot_loss * self.lamda_weights

            ### tvars = tf.trainable_variables()
            rnn_tvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seq2seq')

            gradients = tf.gradients(self.total_loss, rnn_tvars)
            # Clip the gradients if they are larger than the value given in args
            grads, _ = tf.clip_by_global_norm(gradients, args.grad_clip)
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.apply_gradients(zip(grads, rnn_tvars), global_step=self.global_step)

