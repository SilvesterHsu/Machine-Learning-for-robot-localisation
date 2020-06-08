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

from toolbox import lidar_projection

import numpy as np
import tensorflow as tf
import argparse
import os
import time
import pickle

from tf_cnn_auxiliary_gp import Model
import rospy
import tf as tf_ros
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
import math
import cv2
import copy




class Self_Awareness:

    def __init__(self):

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch_size', type=int, default=1, help='size of mini batch')
        args = parser.parse_args()

        # define topics
        VELODYNE_TOPIC = '/points2'
        ODOM_TOPIC = '/fuser'
        VELODYNE_QUEUE_TOPIC = '/velodyne_queue'
        BIRDVIEW_TOPIC_PUB = '/bird_view_image'

        self.br = tf_ros.TransformBroadcaster()

        GT_POSE_TOPIC = '/gt_pose'
        MAP_TOPIC_PUB = '/pose_map'
        PARTICLES_PUB = '/particles'
        NN_LOCALIZASION_PUB = '/nn_pose'

        self.QUEUE_NUM = 10

        self.frame_id = 0

        rospy.init_node('detection_recognition', anonymous=True)

        # define publisher, subscriber
        self.vel_sub = message_filters.Subscriber(VELODYNE_TOPIC, PointCloud2, queue_size=10)
        self.odom_sub = message_filters.Subscriber(ODOM_TOPIC, Odometry, queue_size=10)
        self.velodyne_queue_pub = rospy.Publisher(VELODYNE_QUEUE_TOPIC, PointCloud2, queue_size=1)
        self.bird_view_pub = rospy.Publisher(BIRDVIEW_TOPIC_PUB, Image, queue_size=1)

        self.gt_pose_pub = rospy.Publisher(GT_POSE_TOPIC, Odometry, queue_size=1)
        self.map_pub = rospy.Publisher(MAP_TOPIC_PUB, Path, queue_size=1)
        self.particles_pub = rospy.Publisher(PARTICLES_PUB, PoseArray, queue_size=1)
        self.nn_pose_pub = rospy.Publisher(NN_LOCALIZASION_PUB, Odometry, queue_size=1)

        self.veldynes = []
        self.odoms = []

        date = 'tm' #sys.argv[1]

        ts = message_filters.ApproximateTimeSynchronizer([self.vel_sub, self.odom_sub], 1, 1)
        ts.registerCallback(self.do_lidar_odom_queue)

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

    def get_relative_transform(self, source, target):

        deltaT = target.dot(np.linalg.inv(source))

        return deltaT

    def update_velodyne_queue(self, vel, odom):

        if len(self.veldynes) >= self.QUEUE_NUM:
            self.veldynes.pop(0)

        if vel.shape[0] > 0:
            self.veldynes.append(vel)

        vels = self.veldynes[0]
        for i in range(1, len(self.veldynes)):
            vels = np.vstack([vels, self.veldynes[i]])

        vels = np.hstack([vels, np.ones([vels.shape[0], 1])])
        vels = np.matmul(np.linalg.inv(odom), vels.transpose())

        vels = vels.transpose()[:, :3]

        return vels


    def convet_odom_msg_to_matrix(self, odom_msg):

        [px, py, pz, q_x, q_y, _qz, q_w] = [odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z, odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w]
        T = tf_tran.quaternion_matrix([q_x, q_y, _qz, q_w])
        T[0:3, 3] = [px, py, pz]

        return T


    def do_lidar_odom_queue(self, vel_msg, odom_msg):

        start_time = time.time()

        self.frame_id += 1

        pc_np = self.convert_pc2msg_to_numpy(vel_msg)
        odom = self.convet_odom_msg_to_matrix(odom_msg)

        # convert global mapped point cloud to local velodyne frame
        vels = self.update_velodyne_queue(pc_np, odom)
        print("point cloud shape: ", vels.shape)

        bird_view_img = lidar_projection.birds_eye_point_cloud(vels, side_range=(-50, 50), fwd_range=(-50, 50), res=0.25, min_height=-6, max_height=-1)
        print("bird-view image shape: ", bird_view_img.shape)

        #if self.frame_id % 2 == 0:
        is_publish = True

        header = vel_msg.header
        header.frame_id = 'fuser_base_link'

        vel_queue_msg = self.convert_numpy_to_pc2msg(header, vels)

        self.velodyne_queue_pub.publish(vel_queue_msg)

        if is_publish:

            bridge = CvBridge()

            bird_view_img_msg = bridge.cv2_to_imgmsg(bird_view_img, encoding="passthrough")
            stamp_now = rospy.Time.now()
            bird_view_img_msg.header.stamp = stamp_now

            self.bird_view_pub.publish(bird_view_img_msg)


        print("[inference_node]: runing time = " + str(time.time() - start_time))


if __name__ == '__main__':
    dr = Self_Awareness()
