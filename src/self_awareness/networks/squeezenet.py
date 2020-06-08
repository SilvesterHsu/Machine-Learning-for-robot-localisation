from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import conv2d, avg_pool2d, max_pool2d, fully_connected
from tensorflow.contrib.layers import batch_norm, l2_regularizer, flatten
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.framework import arg_scope


@add_arg_scope
def fire_module(inputs,
                squeeze_depth,
                expand_depth,
                stride=1,
                reuse=None,
                scope=None):
    with tf.variable_scope(scope, 'fire', [inputs], reuse=reuse):
        with arg_scope([conv2d, max_pool2d]):
            net = _squeeze(inputs, squeeze_depth, stride)
            net = _expand(net, expand_depth)
        return net


def _squeeze(inputs, num_outputs, stride):
    return conv2d(inputs, num_outputs, [1, 1], stride=stride, scope='squeeze')


def _expand(inputs, num_outputs):
    with tf.variable_scope('expand'):
        e1x1 = conv2d(inputs, num_outputs, [1, 1], stride=1, scope='1x1')
        e3x3 = conv2d(inputs, num_outputs, [3, 3], scope='3x3')
    return tf.concat([e1x1, e3x3], 3)


class Squeezenet(object):
    """Original squeezenet architecture for 224x224 images."""
    name = 'squeezenet'

    def __init__(self, args):
        self._num_outputs = args.num_classes
        self._weight_decay = args.weight_decay
        self._batch_norm_decay = args.batch_norm_decay
        self._is_built = False

    def build(self, x):
        self._is_built = True
        with tf.variable_scope(self.name, values=[x]):
            with arg_scope(_arg_scope(self._weight_decay)):
                return self._squeezenet(x, self._num_outputs)

    @staticmethod
    def _squeezenet(images, num_outputs=1000):
        net = conv2d(images, 96, [7, 7], stride=2, scope='conv1')
        net = max_pool2d(net, [3, 3], stride=2, scope='maxpool1')
        net = fire_module(net, 16, 64, scope='fire2')
        net = fire_module(net, 16, 64, scope='fire3')
        net = fire_module(net, 32, 128, scope='fire4')
        net = max_pool2d(net, [3, 3], stride=2, scope='maxpool4')
        net = fire_module(net, 32, 128, scope='fire5')
        net = fire_module(net, 48, 192, scope='fire6')
        net = fire_module(net, 48, 192, scope='fire7')
        net = fire_module(net, 64, 256, scope='fire8')
        net = max_pool2d(net, [3, 3], stride=2, scope='maxpool8')
        net = fire_module(net, 64, 256, scope='fire9')
        net = conv2d(net, num_outputs, [1, 1], stride=1, scope='conv10')
        net = avg_pool2d(net, [13, 13], stride=1, scope='avgpool10')
        logits = tf.squeeze(net, [2], name='logits')
        return logits


class Squeezenet_Localization(object):
    """Modified version of squeezenet for global localization"""
    name = 'squeezenet_localization'

    def __init__(self, args):
        self._weight_decay = args.weight_decay
        self._batch_norm_decay = args.batch_norm_decay
        self._is_built = False

    def build(self, x):
        self._is_built = True
        with tf.variable_scope(self.name, values=[x]):
            with arg_scope(_arg_scope(self._weight_decay)):
                return self._squeezenet(x)

    @staticmethod
    def _squeezenet(images, num_outputs=6):
        with tf.variable_scope("squeeze_base"):
            net = conv2d(images, 96, [7, 7], stride=2, scope='conv1')
            net = max_pool2d(net, [3, 3], stride=2, scope='maxpool1')
            net = fire_module(net, 16, 64, scope='fire2')
            net = fire_module(net, 16, 64, scope='fire3')
            net = fire_module(net, 32, 128, scope='fire4')
            net = max_pool2d(net, [3, 3], stride=2, scope='maxpool4')
            net = fire_module(net, 32, 128, scope='fire5')
            net = fire_module(net, 48, 192, scope='fire6')
            net = fire_module(net, 48, 192, scope='fire7')
            net = fire_module(net, 64, 256, scope='fire8')
            net = max_pool2d(net, [3, 3], stride=2, scope='maxpool8')
            net = fire_module(net, 64, 256, scope='fire9')
            net = conv2d(net, 64, [1, 1], stride=1, scope='conv10')
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
        with arg_scope([conv2d, avg_pool2d, max_pool2d],
                       data_format='NHWC') as sc:
                return sc


'''
Network in Network: https://arxiv.org/abs/1312.4400
See Section 3.2 for global average pooling
'''
