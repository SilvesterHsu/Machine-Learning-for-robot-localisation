from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import conv2d, avg_pool2d, max_pool2d, fully_connected
from tensorflow.contrib.layers import batch_norm, l2_regularizer, flatten
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.framework import arg_scope




class Vggnet_Localization(object):
    """Modified version of squeezenet for global localization"""
    name = 'vggnet_localization'

    def __init__(self, args, name='vggnet_localization'):
        self.name = name
        self._weight_decay = args.weight_decay
        self._batch_norm_decay = args.batch_norm_decay
        self._is_built = False

    def build(self, x, is_training, opt="full"):
        self._is_built = True
        with tf.variable_scope(self.name, values=[x]):
            with arg_scope(_arg_scope(self._weight_decay, is_training, self._batch_norm_decay)):
                if opt == "base":
                    return self.build_vggnet_base(x)
                elif opt == "regressor":
                    return self.build_regressor(x)
                elif opt == "context":
                    return self.build_context_stack(x)
                else:
                    return self.build_regressor(self.build_context_stack(self.build_vggnet_base(x)))

    @staticmethod
    def build_vggnet_base(images):
        with tf.variable_scope("vgg_base"):
            net = conv2d(images, 64, [3, 3], stride=2, activation_fn=tf.nn.relu, scope='conv1/conv1_1')
            net = conv2d(net, 64, [3, 3], stride=1, activation_fn=tf.nn.relu, scope='conv1/conv1_2')
            #net = max_pool2d(net, [2, 2], stride=2, scope='maxpool1')

            net = conv2d(net, 128, [3, 3], stride=2, activation_fn=tf.nn.relu, scope='conv2/conv2_1')
            net = conv2d(net, 128, [3, 3], stride=1, activation_fn=tf.nn.relu, scope='conv2/conv2_2')
            #net = max_pool2d(net, [2, 2], stride=2, scope='maxpool2')

            net = conv2d(net, 256, [3, 3], stride=2, activation_fn=tf.nn.relu, scope='conv3/conv3_1')
            net = conv2d(net, 256, [3, 3], stride=1, activation_fn=tf.nn.relu, scope='conv3/conv3_2')
            net = conv2d(net, 256, [3, 3], stride=1, activation_fn=tf.nn.relu, scope='conv3/conv3_3')
            # net = max_pool2d(net, [2, 2], stride=2, scope='maxpool3')

            net = conv2d(net, 512, [3, 3], stride=2, activation_fn=tf.nn.relu, scope='conv4/conv4_1')
            net = conv2d(net, 512, [3, 3], stride=1, activation_fn=tf.nn.relu, scope='conv4/conv4_2')
            net = conv2d(net, 512, [3, 3], stride=1, activation_fn=tf.nn.relu, scope='conv4/conv4_3')
            ###net = max_pool2d(net, [2, 2], stride=2, scope='maxpool4')

        return net

    @staticmethod
    def build_context_stack(net):
        with tf.variable_scope("context_stack"):
            net = conv2d(net, 128, [1, 1], stride=1, activation_fn=tf.nn.relu, scope='squeeze')
            net = conv2d(net, 128, [3, 3], stride=1, padding="SAME", activation_fn=tf.nn.relu, scope='context5_1')
            net = conv2d(net, 128, [3, 3], stride=1, padding="SAME", activation_fn=tf.nn.relu, scope='context5_2')
            net = conv2d(net, 128, [3, 3], stride=1, padding="SAME", activation_fn=tf.nn.relu, scope='context5_3')
            net = conv2d(net, 128, [3, 3], stride=1, padding="SAME", activation_fn=tf.nn.relu, scope='context5_4')
            net = conv2d(net, 64, [1, 1], stride=1, activation_fn=tf.nn.relu, scope='squeeze2')

        return net

    @staticmethod
    def build_regressor(feat, num_outputs=7):

        with tf.variable_scope("regressor"):
            feat = flatten(feat)
            net_t = fully_connected(feat, 4096, activation_fn=tf.nn.relu, scope='fc11_trans')
            net_t = fully_connected(net_t, 4096, activation_fn=tf.nn.relu, scope='fc12_trans')
            feature_t = fully_connected(net_t, 128, activation_fn=tf.nn.relu, scope='fc13_trans')
            net_r = fully_connected(feat, 4096, activation_fn=tf.nn.relu, scope='fc11_rot')
            net_r = fully_connected(net_r, 4096, activation_fn=tf.nn.relu, scope='fc12_rot')
            feature_r = fully_connected(net_r, 128, activation_fn=tf.nn.relu, scope='fc13_rot')

            logits_t = fully_connected(feature_t, 3, activation_fn=None, scope='logits_t')
            logits_r = fully_connected(feature_r, 4, activation_fn=None, scope='logits_r')

            logits_r = tf.nn.l2_normalize(logits_r, axis=1)
            #logits_r = tf.div(logits_r, tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(logits_r), axis=1), 10e-12)))
            logits = tf.concat([logits_t, logits_r], axis=1, name='logits')
        return logits, feature_t, feature_r


def _arg_scope(weight_decay, is_training, bn_decay):
    with arg_scope([conv2d, fully_connected],
                   weights_regularizer=l2_regularizer(weight_decay),
                   reuse=tf.AUTO_REUSE):
                   #normalizer_fn=batch_norm,
                   #normalizer_params={'is_training': is_training,
                   #                   'fused': None,
                   #                   'decay': bn_decay}):
        with arg_scope([conv2d, avg_pool2d, max_pool2d],
                    data_format='NHWC') as sc:
                return sc


'''
Network in Network: https://arxiv.org/abs/1312.4400
See Section 3.2 for global average pooling
'''
