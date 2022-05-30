import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
import tensorflow.contrib.losses as tf_losses

def huberLoss(label, pred, weight=1.0, k=1.0, scope=None):
    # ref http://concise-bio.readthedocs.io/en/latest/_modules/concise/tf_helper.html

    with tf.name_scope(scope, "absolute_difference",[pred, label]) as scope:
        pred.get_shape().assert_is_compatible_with(label.get_shape())
        if weight is None:
            raise ValueError("`weight` value is null please fix the weight data!")
        pred = math_ops.to_float(pred)
        label = math_ops.to_float(label)
        diff = math_ops.subtract(pred, label)
        abs_diff = tf.abs(diff)
        losses = tf.where(abs_diff < k, 0.5 * tf.square(diff), k * abs_diff - 0.5 * k ** 2) #huberLoss implementationfrom the reference
        return tf.losses.compute_weighted_loss(losses, weight)