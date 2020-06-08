import numpy as np
import tensorflow as tf
import argparse
import os
import time
import pickle

from src.self_awareness.networks import utils
from src.self_awareness.tf_cnn_auxiliary_gp import Model
import matplotlib.pyplot as plt
import random

import roslib
import rospy
import tf as tf_ros
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
import math
import cv2
import copy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='size of mini batch')
    parser.add_argument('--model_dir', type=str, default='/home/kevin/models/global_localization/gp', help='model directory')
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

    # Initialize a TensorFlow session
    with model.gp_model.enquire_session() as sess:

        '''Initialization'''
        # Initialize the gp model
        # model.gp_model.initialize(session=sess)
        model.gp_model.compile(sess)

        # Initialize the variables
        tf_nn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dense_feat')
        tf_nn_vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='context_stack_global')
        tf_nn_vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='regressor_global')
        tf_vars = tf_nn_vars + tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='adam')
        tf_vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='param')
        sess.run(tf.variables_initializer(var_list=tf_vars))
        # Get the tensors to save (model variables)
        tf_save_vars = copy.copy(tf_nn_vars)
        tf_save_vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gp')
        tf_save_vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='param')

        '''Evaluate from saved model'''
        # Get the checkpoint state to load the model from
        ckpt_file = os.path.join(args.model_dir, 'model-179.ckpt-80000')
        print('loading model: ', ckpt_file)
        saver = tf.train.Saver(tf_save_vars)
        # Restore the model at the checpoint
        saver.restore(sess, ckpt_file)
        print('model restored.')

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
        PARTICLES_PUB = '/particles'
        NN_LOCALIZASION_PUB = '/nn_pose'
        gt_pose_pub = rospy.Publisher(GT_POSE_TOPIC, Odometry, queue_size=1)
        bird_view_pub = rospy.Publisher(BIRDVIEW_TOPIC_PUB, Image, queue_size=1)
        map_pub = rospy.Publisher(MAP_TOPIC_PUB, Path, queue_size=1)
        particles_pub = rospy.Publisher(PARTICLES_PUB, PoseArray, queue_size=1)
        nn_pose_pub = rospy.Publisher(NN_LOCALIZASION_PUB, Odometry, queue_size=1)

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
            feed = {model.input_data: x}
            trans_pred, rot_pred, trans_mean, trans_cov, samples = sess.run([model.trans_prediction, model.rot_prediction, model.distribution_mean, model.distribution_cov, model.samples], feed)
            # Toc
            end = time.time()

            particles = PoseArray()
            particles.header.stamp = rospy.Time.now()
            particles.header.frame_id = 'world'
            for s in samples:
                pose = Pose()
                [pose.position.x, pose.position.y, pose.position.z] = s
                [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w] = rot_pred[0]
                particles.poses.append(pose)

            particles_pub.publish(particles)


            trans_gt = np.asarray(y)[:, :3]
            rot_gt = np.asarray(y)[:, -4:]

            [px_pred, py_pred, pz_pred] = [trans_pred[0][0], trans_pred[0][1], trans_pred[0][2]]
            [qx_pred, qy_pred, qz_pred, qw_pred] = [rot_pred[0][0], rot_pred[0][1], rot_pred[0][2], rot_pred[0][3]]

            br.sendTransform((px_pred, py_pred, pz_pred),
                             (qx_pred, qy_pred, qz_pred, qw_pred), rospy.Time.now(),
                             "estimation", "world")

            [px_gt, py_gt, pz_gt] = [trans_gt[0][0], trans_gt[0][1], trans_gt[0][2]]
            [qx_gt, qy_gt, qz_gt, qw_gt] = [rot_gt[0][0], rot_gt[0][1], rot_gt[0][2], rot_gt[0][3]]

            br.sendTransform((px_gt, py_gt, pz_gt),
                             (qx_gt, qy_gt, qz_gt, qw_gt),
                             rospy.Time.now(), "gt", "world")

            timestamp = rospy.Time.now()

            nn_pose_msg = Odometry()
            nn_pose_msg.header.frame_id = 'world'
            nn_pose_msg.header.stamp = timestamp
            nn_pose_msg.child_frame_id = 'base_link'
            nn_pose_msg.pose.pose.position.x = px_pred
            nn_pose_msg.pose.pose.position.y = py_pred
            nn_pose_msg.pose.pose.position.z = pz_pred
            [nn_pose_msg.pose.pose.orientation.x, nn_pose_msg.pose.pose.orientation.y, nn_pose_msg.pose.pose.orientation.z, nn_pose_msg.pose.pose.orientation.w] = [qx_pred, qy_pred, qz_pred, qw_pred]

            conv = np.zeros((6,6), dtype=np.float32)
            [conv[0][0], conv[1][1], conv[2][2]] = trans_cov[0]
            nn_pose_msg.pose.covariance = conv.flatten().tolist()
            nn_pose_pub.publish(nn_pose_msg)

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

            rospy.sleep(.25)

            cv2.waitKey(0)

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
    rospy.init_node('global_localization_tf_broadcaster')
    main()
    rospy.spin()
