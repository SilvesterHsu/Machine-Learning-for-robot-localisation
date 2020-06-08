import tensorflow as tf


class Metrics(object):
    def __init__(self, labels, clone_predictions, device, name, padded_data=False):
        self.labels = labels
        self.clone_predictions = clone_predictions
        self.device = device
        self.name = name
        self.padded_data = padded_data
        self.accuracy = None
        self.trans_update_op = None
        self.rot_update_op = None
        self.reset_op = None
        self._generate_metrics()

    def _generate_metrics(self):
        with tf.variable_scope('metrics/'), tf.device(self.device):
            with tf.variable_scope(self.name, reuse=True):
                predictions = tf.concat(
                    values=self.clone_predictions,
                    axis=0
                )
                #if self.padded_data:
                    #not_padded = tf.not_equal(self.labels, -1)
                    #self.labels = tf.boolean_mask(self.labels, not_padded)
                    #predictions = tf.boolean_mask(predictions, not_padded)

                trans_pred, rot_pred = tf.split(predictions, [3, 3], axis=1)
                trans_target, rot_target = tf.split(self.labels, [3, 3], axis=1)

                self.trans_error, self.trans_update_op = tf.metrics.root_mean_squared_error(labels=trans_target, predictions=trans_pred, name='trans_error')
                self.rot_error, self.rot_update_op = tf.metrics.root_mean_squared_error(labels=rot_target, predictions=rot_pred, name='rot_error')

                error_vars = tf.contrib.framework.get_local_variables(scope='metrics/{}'.format(self.name))

                self.reset_op = tf.variables_initializer(var_list=error_vars)
