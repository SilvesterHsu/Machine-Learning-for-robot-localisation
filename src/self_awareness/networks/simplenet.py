from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import conv2d, avg_pool2d, max_pool2d, fully_connected
from tensorflow.contrib.layers import batch_norm, l2_regularizer, flatten
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.framework import arg_scope



class Simplenet_Localization(object):
    """Modified version of squeezenet for global localization"""
    name = 'simplenet_localization'

    def __init__(self, args):
        self._weight_decay = args.weight_decay
        self._batch_norm_decay = args.batch_norm_decay
        self._is_built = False

    def build(self, x):
        self._is_built = True
        with tf.variable_scope(self.name, values=[x]):
            with arg_scope(_arg_scope(self._weight_decay)):
                return self._simplenet(x)

    @staticmethod
    def _simplenet(images, num_outputs=6):
        with tf.variable_scope("squeeze_base"):
            net = conv2d(images, 96, [7, 7], stride=2, activation_fn=tf.nn.relu, scope='conv1')
            net = max_pool2d(net, [3, 3], stride=2, scope='maxpool1')
            net = conv2d(net, 128, [5, 5], stride=1, activation_fn=tf.nn.relu, scope='conv2')
            net = max_pool2d(net, [3, 3], stride=2, scope='maxpool2')
            net = conv2d(net, 256, [5, 5], stride=1, activation_fn=tf.nn.relu, scope='conv3')
            net = max_pool2d(net, [3, 3], stride=2, scope='maxpool3')
            net = conv2d(net, 256, [5, 5], stride=1, activation_fn=tf.nn.relu, scope='conv4')
            net = conv2d(net, 64, [1, 1], stride=1, activation_fn=tf.nn.relu, scope='squeeze')
            feat = flatten(net)

        with tf.variable_scope("regressor"):
            net_t = fully_connected(feat, 1024, activation_fn=tf.nn.relu, scope='fc11_trans')
            net_t = fully_connected(net_t, 1024, activation_fn=tf.nn.relu, scope='fc12_trans')
            net_t = fully_connected(net_t, 128, activation_fn=tf.nn.relu, scope='fc13_trans')
            net_r = fully_connected(feat, 1024, activation_fn=tf.nn.relu, scope='fc11_rot')
            net_r = fully_connected(net_r, 1024, activation_fn=tf.nn.relu, scope='fc12_rot')
            net_r = fully_connected(net_r, 128, activation_fn=tf.nn.relu, scope='fc13_rot')
            logits_t = fully_connected(net_t, int(num_outputs/2), activation_fn=None, scope='logits_t')
            logits_r = fully_connected(net_r, int(num_outputs/2), activation_fn=None, scope='logits_r')
            logits = tf.concat([logits_t, logits_r], axis=1, name='logits')
        return logits


def _arg_scope(weight_decay):
    with arg_scope([conv2d],
                   weights_regularizer=l2_regularizer(weight_decay)):
                   #normalizer_fn=batch_norm,
                   #normalizer_params={'is_training': is_training,
                   #                   'fused': True,
                   #                   'decay': bn_decay}):
        with arg_scope([conv2d, avg_pool2d, max_pool2d, batch_norm],
                       data_format='NHWC') as sc:
                return sc


'''
Network in Network: https://arxiv.org/abs/1312.4400
See Section 3.2 for global average pooling
'''
