#!/usr/bin/python3.5

# -*- coding: utf-8 -*-
# @Author: Kevin Sun
# @Date:   2018-06-09 18:20:13
# @Last Modified by:   Kevin Sun
# @Last Modified time: 2017-05-10 19:10:52

import rospy
import message_filters
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs import point_cloud2
from tf import transformations as tf_tran

from src.self_awareness.toolbox import lidar_projection

import numpy as np
import argparse
import os
import time
import pickle

from self_localization.srv import *

from src.self_awareness.learning.tf_cnn_auxiliary_gp import Model
import rospy
import tf as tf_ros
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from std_msgs.msg import Bool

import math
import cv2
import copy


class NN_DataSaver:

    def __init__(self):

        parser = argparse.ArgumentParser()
        parser.add_argument('--saving_dir', type=str, default='/home/kevin/data/michigan_gt/tmp',
                            help='dataset directory')
        args = parser.parse_args()

        if not os.path.isdir(args.saving_dir):
            os.mkdir(args.saving_dir)


        self.img_save_dir = os.path.join(args.saving_dir, 'images')
        self.pose_save_dir = os.path.join(args.saving_dir, 'poses')
        index_dir = args.saving_dir + '/index.txt'

        if os.path.isdir(self.img_save_dir) == False:
            os.mkdir(self.img_save_dir)

        if os.path.isdir(self.pose_save_dir) == False:
            os.mkdir(self.pose_save_dir)

        self.text_file = open(index_dir, "a")


        # define topics
        VELODYNE_QUEUE_TOPIC = '/velodyne_queue'
        BIRDVIEW_TOPIC_PUB = '/bird_view_image'

        self.br = tf_ros.TransformBroadcaster()

        GT_POSE_TOPIC = '/gt_pose'
        MAP_TOPIC_PUB = '/pose_map'
        PARTICLES_PUB = '/particles'
        NN_LOCALIZASION_PUB = '/nn_pose'

        rospy.init_node('nn_data_saver', anonymous=True)

        # define publisher, subscriber
        self.bird_view_pub = rospy.Publisher(BIRDVIEW_TOPIC_PUB, Image, queue_size=1)

        self.gt_pose_pub = rospy.Publisher(GT_POSE_TOPIC, Odometry, queue_size=1)

        self.nn_data_saver_srv = rospy.Service('nn_data_saver', NNDataSaver, self.do_save)


        print("Ready to localize the robot!")

        rospy.spin()

    def convert_pc2msg_to_numpy(self, velodyne_points, mode='XYZ'):
        pcl_array = []

        if mode == 'XYZ':
            for point in point_cloud2.read_points(velodyne_points, skip_nans=True, field_names=("x", "y", "z")):
                pcl_array.append([point[0], point[1], point[2]])
        else:
            for point in point_cloud2.read_points(velodyne_points, skip_nans=True,
                                                  field_names=("x", "y", "z", "intensity", "ring")):
                pcl_array.append([point[0], point[1], point[2], point[3], point[4]])

        return np.asarray(pcl_array)

    def convert_numpy_to_pc2msg(self, header, vel):

        pointFiledx = PointField('x', 0, 7, 1)
        pointFiledy = PointField('y', 4, 7, 1)
        pointFieldz = PointField('z', 8, 7, 1)
        pointFieldi = PointField('intensity', 12, 7, 1)
        pointFiled = [pointFiledx, pointFiledy, pointFieldz]

        header.stamp = rospy.Time.now()
        vel_msg = point_cloud2.create_cloud(header, pointFiled, vel.tolist())

        return vel_msg


    def convet_odom_msg_to_matrix(self, odom_msg):

        [px, py, pz, q_x, q_y, _qz, q_w] = [odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z, odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w]
        T = tf_tran.quaternion_matrix([q_x, q_y, _qz, q_w])
        T[0:3, 3] = [px, py, pz]

        return T


    def do_save(self, req):

        start_time = time.time()

        vel_msg = req.vels
        odom_msg = req.pose

        pc_np = self.convert_pc2msg_to_numpy(vel_msg)
        odom = self.convet_odom_msg_to_matrix(odom_msg)

        bird_view_img = lidar_projection.birds_eye_point_cloud(pc_np, side_range=(-50, 50), fwd_range=(-50, 50), res=0.25, min_height=-6, max_height=-1)
        print("bird-view image shape: ", bird_view_img.shape)

        is_publish = False
        is_save = True


        if is_save:
            time_stamp = req.vels.header.stamp

            position = np.array(tf_tran.translation_from_matrix(odom), dtype=np.single)
            euler = np.array(tf_tran.euler_from_matrix(odom, 'rxyz'), dtype=np.single)

            cv2.imwrite(os.path.join(self.img_save_dir, str(time_stamp) + ".png"), bird_view_img)
            np.savetxt(os.path.join(self.pose_save_dir, str(time_stamp) + ".txt"), np.concatenate((position, euler), axis=0))

            self.text_file.write("%s\n" % str(time_stamp))
            print('saving ' + str(time_stamp))

        if is_publish:

            bridge = CvBridge()

            bird_view_img_msg = bridge.cv2_to_imgmsg(bird_view_img, encoding="passthrough")
            stamp_now = rospy.Time.now()
            bird_view_img_msg.header.stamp = stamp_now

            self.bird_view_pub.publish(bird_view_img_msg)

            self.gt_pose_pub.publish(odom_msg)


        return NNDataSaverResponse(Bool(True))


if __name__ == '__main__':
    dr = NN_DataSaver()
