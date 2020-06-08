# !/usr/bin/python
#
# Convert the velodyne_hits binary files to a rosbag
#
# To call:
#
#   python vel_to_rosbag.py velodyne_hits.bin vel.bag
#

import rosbag, rospy
from std_msgs.msg import Header

import sys
import numpy as np
import struct

from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Odometry
import scipy.interpolate
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from tf import transformations as tf_tran

def read_odom(odom_dir):

    odom_np = np.loadtxt(odom_dir, delimiter=",")

    odom = dict()

    for i in range(odom_np.shape[0]):
        keyi = str(int(odom_np[i, 0]/1e4))
        odomi = odom_np[i, 1:]

        odom[keyi] = odomi

    print("the odometry data is loaded")

    return odom

def read_ground_truth(gt_dir):

    gt_np = np.loadtxt(gt_dir, delimiter=",")

    gt = dict()

    for i in range(gt_np.shape[0]):
        keyi = str(int(gt_np[i, 0]/1e4))
        gti = gt_np[i, 1:]

        gt[keyi] = gti

    print("the ground truth data is loaded")

    return gt

def read_imu(imu_dir):

    imu_np = np.loadtxt(imu_dir, delimiter = ",")

    imu = dict()

    for i in range(imu_np.shape[0]):
        keyi = str(int(imu_np[i, 0]/1e4))
        imui = imu_np[i, 1:]

        imu[keyi] = imui

    print("the imu data is loaded")

    return imu


def convert(x_s, y_s, z_s):

    scaling = 0.005 # 5 mm
    offset = -100.0

    x = x_s * scaling + offset
    y = y_s * scaling + offset
    z = z_s * scaling + offset

    return x, y, z

def array2pointcloud2(cloud, stamp):

    header = Header()

    pointFiledx = PointField('x', 0, 7, 1)
    pointFiledy = PointField('y', 4, 7, 1)
    pointFieldz = PointField('z', 8, 7, 1)
    pointFieldi = PointField('intensity', 12, 7, 1)
    pointFieldr = PointField('ring', 16, 7, 1)
    pointFiled = [pointFiledx, pointFiledy, pointFieldz, pointFieldi, pointFieldr]

    header.stamp = stamp
    header.frame_id = 'velodyne'
    pc2_msg = point_cloud2.create_cloud(header, pointFiled, cloud)

    return pc2_msg

def verify_magic(s):

    magic = 44444

    m = struct.unpack('<HHHH', s)

    return len(m)>=3 and m[0] == magic and m[1] == magic and m[2] == magic and m[3] == magic


def ssc_to_homo(ssc):
    # Convert 6-DOF ssc coordinate transformation to 4x4 homogeneous matrix
    # transformation
    sr = np.sin(ssc[3])
    cr = np.cos(ssc[3])

    sp = np.sin(ssc[4])
    cp = np.cos(ssc[4])

    sh = np.sin(ssc[5])
    ch = np.cos(ssc[5])

    H = np.zeros((4, 4))

    H[0, 0] = ch*cp
    H[0, 1] = -sh*cr + ch*sp*sr
    H[0, 2] = sh*sr + ch*sp*cr
    H[1, 0] = sh*cp
    H[1, 1] = ch*cr + sh*sp*sr
    H[1, 2] = -ch*sr + sh*sp*cr
    H[2, 0] = -sp
    H[2, 1] = cp*sr
    H[2, 2] = cp*cr

    H[0, 3] = ssc[0]
    H[1, 3] = ssc[1]
    H[2, 3] = ssc[2]

    H[3, 3] = 1

    return H

def get_gt_pose(gt, timestamp):
    func = scipy.interpolate.interp1d(gt[:, 0], gt[:, 1:], kind='nearest', axis=0)
    return func(timestamp)

def main(args):

    #if len(sys.argv) < 2:
    #    print 'Please specify velodyne hits file'
    #    return 1

    #if len(sys.argv) < 3:
    #    print 'Please specify odom file'
    #    return 1

    #if len(sys.argv) < 4:
    #    print 'Please specify imu file'
    #    return 1
    #if len(sys.argv) < 5:
    #    print 'Please specify output rosbag file'
    #    return 1

    date = sys.argv[1]
    velodyne_hits_dir = '/home/kevin/DATA2/raw_data/' + date + '/velodyne_hits.bin'
    odom_file_dir = '/home/kevin/DATA2/raw_data/' + date + '/odometry_mu_100hz.csv'
    gt_file_dir = '/home/kevin/DATA2/raw_data/' + date + '/groundtruth_'+date+'.csv'
    output_dir = '/home/kevin/DATA/Michigan/tmp/'

    f_bin = open(velodyne_hits_dir, "rb")

    #odom = read_odom(odom_file_dir)
    #gt1 = read_ground_truth(gt_file_dir)
    print("read odometry")
    odom = pd.read_csv(odom_file_dir).values

    print('Reading ground truth')
    gt = pd.read_csv(gt_file_dir).values

    count = 0
    split_num = 5000

    is_save_odom = True
    is_save_gt = True


    while True:

        if count % split_num is 0:
            bag = rosbag.Bag(output_dir + str(int(count / split_num)) + '.bag', 'w')

        count += 1
        print(count)

        # Read all hits
        data = []

        for iii in range(233):

            magic = f_bin.read(8)
            if magic == '': # eof
                break

            if not verify_magic(magic):
                print("Could not verify magic")

            num_hits = struct.unpack('<I', f_bin.read(4))[0]
            utime = struct.unpack('<Q', f_bin.read(8))[0]

            f_bin.read(4) # padding

            for ii in range(num_hits):

                x = struct.unpack('<H', f_bin.read(2))[0]
                y = struct.unpack('<H', f_bin.read(2))[0]
                z = struct.unpack('<H', f_bin.read(2))[0]
                i = struct.unpack('B', f_bin.read(1))[0]
                l = struct.unpack('B', f_bin.read(1))[0]

                x, y, z = convert(x, y, z)

                data.append([x, y, z, float(i), float(l)])

        if len(data) == 0:
            print("task finished!")
            break
        # Now write out to rosbag

        if count == 1:
            last_utime = utime
            delta_t = 0.1375

            if is_save_odom:
                deltaT_odom = ssc_to_homo([0,0,0,0,0,0])
            if is_save_gt:
                deltaT_gt = ssc_to_homo([0,0,0,0,0,0])
        else:
            print(utime)
            delta_t = (utime - last_utime) / 1e6


        ################################################################################################################
        # velodyne
        timestamp = rospy.Time.from_sec(utime/1e6)

        vel_msg = array2pointcloud2(data, timestamp)

        # write to bag
        bag.write('/velodyne_points', vel_msg, t=timestamp)
        ################################################################################################################

        # odometry
        key = str(int(utime/1e4))


        if is_save_odom:

            try:
                odom_eular = get_gt_pose(odom, utime)
            except:
                print("error")
                if count == 1:
                    count -= 1
                continue

            T_odom = ssc_to_homo(odom_eular)

            if count > 1:
                deltaT_odom = np.dot(tf_tran.inverse_matrix(last_T_odom), T_odom)

            quaternion_odom = tf_tran.quaternion_from_matrix(T_odom)
            angular_odom = np.array(tf_tran.euler_from_matrix(deltaT_odom, 'rxyz')) / delta_t
            linear_odom = tf_tran.translation_from_matrix(deltaT_odom) / delta_t


            odom_msg = Odometry()
            odom_msg.header.frame_id = 'odom'
            odom_msg.header.stamp = timestamp
            odom_msg.child_frame_id = 'base_link'
            odom_msg.pose.pose.position.x = odom_eular[0]
            odom_msg.pose.pose.position.y = odom_eular[1]
            odom_msg.pose.pose.position.z = odom_eular[2]
            [odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w] = quaternion_odom
            [odom_msg.twist.twist.linear.x, odom_msg.twist.twist.linear.y, odom_msg.twist.twist.linear.z] = linear_odom
            [odom_msg.twist.twist.angular.x, odom_msg.twist.twist.angular.y, odom_msg.twist.twist.angular.z] = angular_odom
            # write to bag
            bag.write('/robot_odom', odom_msg, t=timestamp)

            last_T_odom = T_odom


        ################################################################################################################
        # ground truth
        if is_save_gt:

            try:
                pose_eular = get_gt_pose(gt, utime)
            except:
                print("error")
                if count == 1:
                    count -= 1
                continue

            T_gt = ssc_to_homo(pose_eular)

            if count > 1:
                deltaT_gt = np.dot(tf_tran.inverse_matrix(last_T_gt), T_gt)

            quaternion_gt = tf_tran.quaternion_from_matrix(T_gt)
            angular_gt = np.array(tf_tran.euler_from_matrix(deltaT_gt, 'rxyz')) / delta_t
            linear_gt = tf_tran.translation_from_matrix(deltaT_gt) / delta_t

            gt_msg = Odometry()
            gt_msg.header.frame_id = 'world'
            gt_msg.header.stamp = timestamp
            gt_msg.child_frame_id = 'base_link'
            gt_msg.pose.pose.position.x = pose_eular[0]
            gt_msg.pose.pose.position.y = pose_eular[1]
            gt_msg.pose.pose.position.z = pose_eular[2]
            [gt_msg.pose.pose.orientation.x, gt_msg.pose.pose.orientation.y, gt_msg.pose.pose.orientation.z,
            gt_msg.pose.pose.orientation.w] = quaternion_gt
            [gt_msg.twist.twist.linear.x, gt_msg.twist.twist.linear.y, gt_msg.twist.twist.linear.z] = linear_gt
            [gt_msg.twist.twist.angular.x, gt_msg.twist.twist.angular.y, gt_msg.twist.twist.angular.z] = angular_gt
            # write to bag
            bag.write('/robot_gt', gt_msg, t=timestamp)

            last_T_gt = T_gt




        print(int(utime), " done!")

        last_utime = utime



    f_bin.close()
    bag.close()

    return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
