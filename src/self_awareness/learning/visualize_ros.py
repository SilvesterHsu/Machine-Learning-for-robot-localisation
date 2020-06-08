#!/usr/bin/env python

import numpy as np
import tensorflow as tf
import argparse
import os
import time
import pickle

from src.self_awareness.networks import utils
from src.self_awareness.learning.tf_cnn_auxiliary import Model
import matplotlib.pyplot as plt
import random

import roslib
import rospy
import tf as tf_ros
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
import math


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='size of mini batch')
    parser.add_argument('--model_dir', type=str, default='/home/kevin/models/global_localization', help='model directory')
    parser.add_argument('--test_dataset', type=str, default=['/home/kevin/data/michigan_gt/2012_05_11'] )
    parser.add_argument('--map_dataset', type=str, default='/home/kevin/data/michigan_gt/map')
    args = parser.parse_args()
    test(args)


def test(args):

    # load saved model
    model_dir = args.model_dir
    with open(os.path.join(model_dir, 'config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)

    model = Model(saved_args, is_training=False)

    # Create the data loader object. This object would preprocess the data in terms of
    data_loader = utils.DataLoader(args.batch_size, dataset_dirs=args.test_dataset, is_argumentation = saved_args.data_argumentation, target_image_size = saved_args.target_image_size)
    data_loader.shuffle_data(mode='evaluate')

    sess = tf.InteractiveSession()
    saver = tf.train.Saver(tf.global_variables())
    ckpt_file = os.path.join(args.model_dir, 'model-114.ckpt-460000')
    print('loading model: ', ckpt_file)
    # Restore the model at the checpoint
    saver.restore(sess, ckpt_file)
    #saver.restore(sess, '/home/kevin/models/global_localization/model.ckpt-70000')

    # For each batch
    trans_errors = []
    rot_errors = []
    pose_map = []

    total_trans_error = 0.
    total_rot_error = 0.

    count = 0.

    is_save_map = False
    is_read_map = True

    trans_preds = []
    trans_gts = []

    rot_preds = []
    rot_gts = []

    br = tf_ros.TransformBroadcaster()

    GT_POSE_TOPIC = '/gt_pose'
    BIRDVIEW_TOPIC_PUB = '/bird_view'
    MAP_TOPIC_PUB = '/pose_map'
    gt_pose_pub = rospy.Publisher(GT_POSE_TOPIC, Odometry, queue_size=1)
    bird_view_pub = rospy.Publisher(BIRDVIEW_TOPIC_PUB, Image, queue_size=1)
    map_pub = rospy.Publisher(MAP_TOPIC_PUB, Path, queue_size=1)

    if is_read_map:
        pose_map = np.loadtxt(os.path.join(args.map_dataset, 'map.txt'))

        traj = Path()

        for i in range(pose_map.shape[0]):
            pose = pose_map[i, :]
            t_i = PoseStamped()
            [t_i.pose.position.x, t_i.pose.position.y, t_i.pose.position.z] = pose[:3]
            [t_i.pose.orientation.x, t_i.pose.orientation.y, t_i.pose.orientation.z, t_i.pose.orientation.w] = pose[-4:]
            traj.poses.append(t_i)

        traj.header.stamp = rospy.Time.now()
        traj.header.frame_id = 'world'

        map_pub.publish(traj)
        print("published!")


    for b in range(0, data_loader.num_batches, 10):

        # Tic
        start = time.time()
        # Get the source and target data of the current batch
        # x has the source data, y has the target data
        x, y = data_loader.next_batch(b, mode='evaluate')
        feed = {model.input_data_t1: x, model.target_data_t1: y}
        trans_pred, rot_pred, trans_gt, rot_gt = sess.run([model.trans_prediction, model.rot_prediction, model.trans_target, model.rot_target], feed)

        # Toc
        end = time.time()

        br.sendTransform((trans_pred[0][0], trans_pred[0][1], trans_pred[0][2]),
                         (rot_pred[0][0], rot_pred[0][1], rot_pred[0][2], rot_pred[0][3]), rospy.Time.now(),
                         "estimation", "world")

        [px_gt, py_gt, pz_gt] = [trans_gt[0][0], trans_gt[0][1], trans_gt[0][2]]
        [qx_gt, qy_gt, qz_gt, qw_gt] = [rot_gt[0][0], rot_gt[0][1], rot_gt[0][2], rot_gt[0][3]]

        br.sendTransform((px_gt, py_gt, pz_gt),
                         (qx_gt, qy_gt, qz_gt, qw_gt),
                         rospy.Time.now(), "gt", "world")

        timestamp = rospy.Time.now()

        gt_msg = Odometry()
        gt_msg.header.frame_id = 'world'
        gt_msg.header.stamp = timestamp
        gt_msg.child_frame_id = 'base_link'
        gt_msg.pose.pose.position.x = px_gt
        gt_msg.pose.pose.position.y = py_gt
        gt_msg.pose.pose.position.z = pz_gt
        [gt_msg.pose.pose.orientation.x, gt_msg.pose.pose.orientation.y, gt_msg.pose.pose.orientation.z, gt_msg.pose.pose.orientation.w] = [qx_gt, qy_gt, qz_gt, qw_gt]

        if b % 30 == 0 and is_save_map:
            pose_map.append([px_gt, py_gt, pz_gt] + [qx_gt, qy_gt, qz_gt, qw_gt])

        if b % 10000 == 0 and is_read_map:
            traj.header.stamp = rospy.Time.now()
            traj.header.frame_id = 'world'
            map_pub.publish(traj)

        bridge = CvBridge()

        bird_view_img_msg = bridge.cv2_to_imgmsg(np.asarray(x[0], dtype=np.float32), encoding="passthrough")
        stamp_now = rospy.Time.now()
        bird_view_img_msg.header.stamp = stamp_now

        bird_view_pub.publish(bird_view_img_msg)

        rospy.sleep(.1)

        count += 1

        trans_preds.append(trans_pred[0])
        rot_preds.append(rot_pred[0])
        trans_gts.append(trans_gt[0])
        rot_gts.append(rot_gt[0])

        trans_error = np.sum((trans_pred[0] - trans_gt[0])**2)**0.5
        rot_error_1 = np.arccos(np.dot(rot_pred[0], rot_gt[0]))/math.pi*180
        rot_error_2 = np.arccos(np.dot(rot_pred[0], -rot_gt[0])) / math.pi * 180
        rot_error = min(rot_error_1, rot_error_2)

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


    if len(pose_map):
        np.savetxt(os.path.join(args.map_dataset, 'map.txt'), np.asarray(pose_map, dtype=np.float32))
        print("map is saved!")

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
    rospy.init_node('global_localization_tf_broadcaster_cnn')
    main()
    rospy.spin()
