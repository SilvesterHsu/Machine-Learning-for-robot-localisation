#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import argparse
import os, math
import time
import pickle

from src.self_awareness.networks import utils
from src.self_awareness.tf_model import Model
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='size of mini batch')
    parser.add_argument('--model_dir', type=str, default='/home/kevin/models/global_localization', help='model directory')
    parser.add_argument('--test_dataset', type=str, default=['/home/kevin/data/michigan_gt/2012_01_22'])
    args = parser.parse_args()
    test(args)


def test(args):

    # load saved model
    model_dir = args.model_dir
    with open(os.path.join(model_dir, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)

    model = Model(saved_args, is_training=False)

    # Create the data loader object. This object would preprocess the data in terms of
    data_loader = utils.DataLoader(args.batch_size, dataset_dirs=args.test_dataset, is_argumentation=saved_args.data_argumentation, target_image_size=saved_args.target_image_size)
    data_loader.shuffle_data(mode='evaluate')

    # create the graph
    sess = tf.InteractiveSession()
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state(args.model_dir)
    print('loading model: ', ckpt.model_checkpoint_path)
    # Restore the model at the checpoint
    saver.restore(sess, ckpt.model_checkpoint_path)
    #saver.restore(sess, '/home/kevin/models/global_localization/model.ckpt-70000')

    # For each batch
    trans_errors = []
    rot_errors = []

    total_trans_error = 0.
    total_rot_error = 0.

    count = 0.

    trans_preds = []
    trans_gts = []

    rot_preds = []
    rot_gts = []

    for b in range(data_loader.num_batches):

        # Tic
        start = time.time()
        # Get the source and target data of the current batch
        # x has the source data, y has the target data
        x, y = data_loader.next_batch(b, mode='evaluation')
        feed = {model.input_data: x, model.target_data: y}
        trans_pred, rot_pred, trans_gt, rot_gt = sess.run([model.trans_pred, model.rot_pred, model.trans_target, model.rot_target], feed)
        # Toc
        end = time.time()

        count += 1

        trans_preds.append(trans_pred[0])
        rot_preds.append(rot_pred[0])
        trans_gts.append(trans_gt[0])
        rot_gts.append(rot_gt[0])

        trans_error = np.sum((trans_pred[0] - trans_gt[0]) ** 2) ** 0.5
        rot_error = np.arccos(np.dot(rot_pred[0], rot_gt[0])) / math.pi * 180

        trans_errors.append(trans_error)
        rot_errors.append(rot_error)

        total_trans_error += trans_error
        total_rot_error += rot_error

        saved_args.display = 1
        if b % saved_args.display == 0 and b > 0:
            print(
                "{}/{}, translation error = {:.3f}, rotation error = {:.3f}, time/batch = {:.3f}"
                .format(
                 b,
                 data_loader.num_batches,
                total_trans_error / count,
                total_rot_error / count,
                end - start))

    plt.hist(trans_errors, bins='auto')
    plt.title("Translation errors")
    plt.show()

    plt.hist(rot_errors, bins='auto')
    plt.title("Rotation errors")
    plt.show()

    median_trans_errors = np.median(trans_errors)
    median_rot_errors = np.median(rot_errors)

    print("median translation error = {:.3f}, median rotation error = {:.3f}".format(median_trans_errors, median_rot_errors))

if __name__ == '__main__':
    main()
