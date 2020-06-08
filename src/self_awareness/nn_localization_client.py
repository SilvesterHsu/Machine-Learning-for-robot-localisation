#!/usr/bin/python3.5

# -*- coding: utf-8 -*-
# @Author: Kevin Sun
# @Date:   2018-06-09 18:20:13
# @Last Modified by:   Kevin Sun
# @Last Modified time: 2017-05-10 19:10:52

import rospy
import message_filters
from sensor_msgs.msg import PointCloud2

from self_localization.srv import *


def callback(vels_msg):

    try:
        localize = rospy.ServiceProxy('nn_localization', NNPose)
        res = localize(rospy.Time.now(), vels_msg)
        print(res.pose.pose.pose)
    except ValueError:
        print("Service call failed")


def main():
    rospy.init_node('nn_localization_client', anonymous=True)

    rospy.wait_for_service('nn_localization')

    rospy.Subscriber('/velodyne_queue', PointCloud2, callback)

    rospy.spin()


if __name__ == "__main__":
    main()