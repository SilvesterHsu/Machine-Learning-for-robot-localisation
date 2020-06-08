import tensorflow as tf
import argparse
import os
import time
import pickle
from src.self_awareness.networks import utils
from src.self_awareness.tf_model import Model


parser = argparse.ArgumentParser()

'''Training Parameters'''
parser.add_argument('--batch_size', type=int, default=16, help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
parser.add_argument('--grad_clip', type=float, default=100., help='clip gradients at this value')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('--learning_rate_clip', type=float, default=0.0000001, help='learning rate clip')
parser.add_argument('--decay_rate', type=float, default=.95, help='decay rate for rmsprop')
parser.add_argument('--weight_decay', type=float, default=.0, help='decay rate for rmsprop')
parser.add_argument('--batch_norm_decay', type=float, default=.999, help='decay rate for rmsprop')
parser.add_argument('--keep_prob', type=float, default=1.0, help='dropout keep probability')
parser.add_argument('--lamda_weights', type=float, default=-3., help='lamda weight')
parser.add_argument('--data_argumentation', type=bool, default=True, help='whether do data argument')
parser.add_argument('--is_normalization', type=bool, default=True, help='whether do data nomalization')
parser.add_argument('--target_image_size', default=[300, 300], nargs=2, type=int, help='Input images will be resized to this for data argumentation.')

'''Configure'''
parser.add_argument('--network', type=str, default='vggnet_localization')
parser.add_argument('--model_dir', type=str, default='/home/kevin/models/global_localization', help='rnn, gru, or lstm')
parser.add_argument('--train_dataset', type=str, default = ['/home/kevin/data/michigan_gt/2012_01_08', '/home/kevin/data/michigan_gt/2012_01_15', '/home/kevin/data/michigan_gt/2012_01_22'])
parser.add_argument('--seed', default=1337, type=int)
parser.add_argument('--save_every', type=int, default=10000, help='save frequency')
parser.add_argument('--display', type=int, default=10, help='display frequency')

args = parser.parse_args()

def run():

    # Create the data loader object.
    data_loader = utils.DataLoader(args.batch_size, dataset_dirs=args.train_dataset, is_argumentation=args.data_argumentation, target_image_size=args.target_image_size)
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
        tf.summary.scalar('lamda_sigmoid', model.lamda_weights_sigmoid)
        all_summaries = tf.summary.merge_all()

        if os.path.isfile(os.path.join(args.model_dir, 'config.pkl')):
            '''Train from saved model'''
            # Get the checkpoint state to load the model from
            ckpt_file = os.path.join(args.model_dir, 'model-53.ckpt-48000' )
            print('loading model: ', ckpt_file)
            saver = tf.train.Saver()
            # Restore the model at the checpoint
            saver.restore(sess, ckpt_file)
            print('model restored.')


        # Assign the global step
        sess.run(tf.assign(model.global_step, 0))


        '''Training Loop'''
        for e in range(args.num_epochs):
            # Assign the learning rate (decayed acc. to the epoch number)
            sess.run(tf.assign(model.lr, max(args.learning_rate_clip, args.learning_rate * (args.decay_rate ** e))))
            # shuffle_data
            data_loader.shuffle_data(mode='train')

            # For each batch in this epoch
            train_loss = 0.

            for b in range(data_loader.num_batches):

                # Tic
                start = time.time()
                # Get the source and target data of the current batch
                # x has the source data, y has the target data
                x, y = data_loader.next_batch(b)

                feed = {model.input_data: x, model.target_data: y}
                # Fetch the loss of the model on this batch, the final LSTM state from the session
                batch_total_loss, batch_trans_loss, batch_rot_loss, global_step, summaries, _ = sess.run([model.total_loss, model.trans_loss, model.rot_loss, model.global_step, all_summaries, model.train_op], feed)

                train_writer.add_summary(summaries, e*data_loader.num_batches+b)

                # Toc
                end = time.time()
                # Print epoch, batch, loss and time taken
                train_loss += batch_total_loss

                if b % args.display == 0:
                     print(
                        "{}/{} (epoch {}), train_loss = {}, time/batch = {:.3f}, learning rate = {:.9f}"
                        .format(
                        e * data_loader.num_batches + b,
                        args.num_epochs * data_loader.num_batches,
                        e,
                        train_loss / (b+1),
                        end - start,
                        sess.run(model.lr)))

                '''Save Model'''
                # Save the model if the current epoch and batch number match the frequency
                if (e * data_loader.num_batches + b + 1) % args.save_every == 0 and ((e * data_loader.num_batches + b) > 0):
                    checkpoint_path = os.path.join(args.model_dir, 'model-{}.ckpt'.format(e))
                    saver = tf.train.Saver()
                    saver.save(sess, checkpoint_path, global_step=global_step)
                    print("model saved to {}".format(checkpoint_path))

                    # Save the arguments int the config file
                    with open(os.path.join(args.model_dir, 'config.pkl'), 'wb') as f:
                        pickle.dump(args, f)


if __name__ == '__main__':
    run()

