import tensorflow as tf
import argparse
import os
import time
import pickle
from src.self_awareness.networks import utils
from src.self_awareness.tf_rnn_model import Model
import numpy as np


parser = argparse.ArgumentParser()

'''Training Parameters'''
parser.add_argument('--rnn_size', type=int, default=128, help='size of RNN hidden state')
parser.add_argument('--max_seq_length', type=int, default=5, help='Maximum RNN sequence length')
parser.add_argument('--num_layers', type=int, default=1, help='number of layers in the RNN')
parser.add_argument('--model', type=str, default='lstm', help='rnn, gru, or lstm')
parser.add_argument('--initializer', type=str, default='xavier', help='initializer for RNN weights: uniform, xavier, svd')
parser.add_argument('--batch_size', type=int, default=128, help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--grad_clip', type=float, default=100., help='clip gradients at this value')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--learning_rate_clip', type=float, default=0.00000001, help='learning rate clip')
parser.add_argument('--decay_rate', type=float, default=.9, help='decay rate for rmsprop')
parser.add_argument('--weight_decay', type=float, default=.0, help='decay rate for rmsprop')
parser.add_argument('--batch_norm_decay', type=float, default=.999, help='decay rate for rmsprop')
parser.add_argument('--keep_prob', type=float, default=1.0, help='dropout keep probability')
parser.add_argument('--lamda_weights', type=float, default=.1, help='lamda weight')
parser.add_argument('--data_argumentation', type=bool, default=True, help='whether do data argument')
parser.add_argument('--is_normalization', type=bool, default=True, help='whether do data nomalization')
parser.add_argument('--target_image_size', default=[300, 300], nargs=2, type=int, help='Input images will be resized to this for data argumentation.')

'''Configure'''
parser.add_argument('--network', type=str, default='vggnet_localization')
parser.add_argument('--model_dir', type=str, default='/home/kevin/models/global_localization', help='rnn, gru, or lstm')
parser.add_argument('--train_dataset', type=str, default = ['/home/kevin/data/michigan_gt/2012_01_08', '/home/kevin/data/michigan_gt/2012_01_15', '/home/kevin/data/michigan_gt/2012_01_22'])
parser.add_argument('--seed', default=1337, type=int)
parser.add_argument('--save_every', type=int, default=1000, help='save frequency')
parser.add_argument('--display', type=int, default=10, help='display frequency')

args = parser.parse_args()

def run():

    # Create the data loader object.
    data_loader = utils.SequenceDataLoader(args.batch_size, dataset_dirs=args.train_dataset, max_seq_length=args.max_seq_length, is_argumentation=args.data_argumentation, target_image_size=args.target_image_size)
    [args.norm_mean, args.norm_std] = [data_loader.norm_mean, data_loader.norm_std]
    # Create a Tensorflow Model
    model = Model(args)

    # Initialize a TensorFlow session
    with tf.Session() as sess:
        # Add all the variables to the list of variables to be saved
        saver = tf.train.Saver(tf.global_variables())

        # Initialize all the variables in the graph
        sess.run(tf.global_variables_initializer())

        '''Summaries'''
        train_writer = tf.summary.FileWriter(os.path.join(args.model_dir, 'log'), sess.graph)
        tf.summary.scalar('total_loss', model.total_loss)
        tf.summary.scalar('trans_loss', model.trans_loss)
        tf.summary.scalar('rot_loss', model.rot_loss)
        tf.summary.scalar('rot_loss0', model.rot_loss0)
        all_summaries = tf.summary.merge_all()

        if os.path.isfile(os.path.join(args.model_dir, 'cnn', 'config.pkl')):
            ##saver2 = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vggnet_localization'))
            ##ckpt = tf.train.get_checkpoint_state(os.path.join(args.model_dir, 'cnn'))
            # Get the checkpoint state to load the model from
            saver = tf.train.Saver(tf.global_variables())
            ckpt = tf.train.get_checkpoint_state(args.model_dir)
            print('loading model: ', ckpt.model_checkpoint_path)
            # Restore the model at the checpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            # Save the arguments int the config file
            with open(os.path.join(args.model_dir, 'config.pkl'), 'wb') as f:
                pickle.dump(args, f)

        # Assign the global step
        sess.run(tf.assign(model.global_step, 0))

        '''Training Loop'''
        for e in range(args.num_epochs):

            # Get the initial cell state of the LSTM
            state = sess.run(model.initial_state)
            is_feed_state = [True for _ in range(args.batch_size)]
            rnn_zeros_state = np.zeros((1, args.rnn_size))

            # Assign the learning rate (decayed acc. to the epoch number)
            sess.run(tf.assign(model.lr, max(args.learning_rate_clip, args.learning_rate * (args.decay_rate ** e))))
            # shuffle_data
            data_loader.shuffle_data(mode='train')

            # For each batch in this epoch
            train_loss = 0.
            b = 0

            while not data_loader.is_epoch_done:

                # Tic
                start = time.time()
                # Get the source and target data of the current batch
                # x has the source data, y has the target data
                x, y, seq_len, new_is_feed_state = data_loader.next_rnn_batch()

                # extract the seq index to achieve dynamic seq2seq model
                index = []
                for i in range(0, args.batch_size):
                    index.append(
                        range(i * args.max_seq_length, i * args.max_seq_length + seq_len[i]))  # self.seq_len[i] - 1

                index = [bb for aa in index for bb in aa]

                # update states
                for i in range(args.batch_size):
                    if not is_feed_state[i]:
                        for layeri in range(args.num_layers):
                            state[0][layeri][0][i] = rnn_zeros_state
                            state[0][layeri][1][i] = rnn_zeros_state
                            state[1][layeri][0][i] = rnn_zeros_state
                            state[1][layeri][1][i] = rnn_zeros_state

                # Feed the source, target data and the initial LSTM state to the model
                feed = {model.input_data: x, model.target_data: y, model.seq_len: seq_len, model.index: index, model.initial_state: state}
                # Fetch the loss of the model on this batch, the final LSTM state from the session
                batch_total_loss, batch_trans_loss, batch_rot_loss, state, summaries, global_step,  _ = sess.run([model.total_loss, model.trans_loss, model.rot_loss, model.final_state, all_summaries, model.global_step, model.train_op], feed)

                is_feed_state = new_is_feed_state

                train_writer.add_summary(summaries, global_step)

                # Toc
                end = time.time()
                # Print epoch, batch, loss and time taken
                train_loss += batch_total_loss

                if b % args.display == 0:
                     print(
                        "{}/{} (epoch {}), train_loss = {:.8f}, time/batch = {:.3f}, learning rate = {:.6f}"
                        .format(
                        global_step,
                        args.num_epochs * data_loader.num_batches,
                        e,
                        train_loss / (b+1),
                        end - start,
                        sess.run(model.lr)))

                '''Save Model'''
                # Save the model if the current epoch and batch number match the frequency
                if (e * data_loader.num_batches + b) % args.save_every == 0 and ((e * data_loader.num_batches + b) > 0):
                    checkpoint_path = os.path.join(args.model_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))

                    # Save the arguments int the config file
                    with open(os.path.join(args.model_dir, 'config.pkl'), 'wb') as f:
                        pickle.dump(args, f)
                    saver.save(sess, os.path.join(args.model_dir, 'model.ckpt'))

                b += 1


if __name__ == '__main__':
    run()

