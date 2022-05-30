import tensorflow as tf
import re
import tensorflow.contrib.slim as tfslim
from tensorflow.contrib.slim.nets import resnet_v1

from toolsetLib.poseDatasetTool import Batch
from toolsetLib import lossesMLToolkit

net_types = {'resnet_50': resnet_v1.resnet_v1_50,
            'resnet_101': resnet_v1.resnet_v1_101}

def predLayer(config, input, name, num_outputs):
    with tfslim.arg_scope([tfslim.conv2d, tfslim.conv2d_transpose], padding='SAME', activation_fn=None, normalizer_fn=None, weights_regularizer=tfslim.l2_regularizer(config.weight_decay)):
        with tf.variable_scope(name):
            pred = tfslim.conv2d_transpose(input, num_outputs, kernel_size=[3, 3], stride=2, scope='block4')
            return pred

def getBatchSpecs(config):
    num_joints = config.num_joints
    batch_size = config.batch_size
    batch_spec = {
        Batch.inputs: [batch_size, None, None, 3],
        Batch.part_score_targets: [batch_size, None, None, num_joints],
        Batch.part_score_weights: [batch_size, None, None, num_joints]
    }
    if config.location_refinement:
        batch_spec[Batch.locref_targets] = [batch_size, None, None, num_joints * 2]
        batch_spec[Batch.locref_mask] = [batch_size, None, None, num_joints * 2]
    if config.pairwise_predict:
        batch_spec[Batch.pairwise_targets] = [batch_size, None, None, num_joints * (num_joints - 1) * 2]
        batch_spec[Batch.pairwise_mask] = [batch_size, None, None, num_joints * (num_joints - 1) * 2]
    return batch_spec

class PoseNet:
    def __init__(self, config):
        self.config = config

    def extractFeatures(self, inputs):
        netFunction = net_types[self.config.net_type]

        mean = tf.constant(self.config.mean_pixel,dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        im_centered = inputs - mean

        with tfslim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, end_points = netFunction(im_centered, is_training=False, global_pool=False)

        return net, end_points   # net: [batch_size, height, width, num_features]

    def predLayer(own, features, endPoints, reuse=None, no_interm=False, scope='pose'):
        config = own.config
        numberLayers = re.findall("resnet_([0-9]+)", config.net_type)[0]
        layerName = 'resnet_v1_{}'.format(numberLayers) + '/block{}/unit_{}/bottleneck_v1'
        out = {}
        with tf.variable_scope(scope, reuse=reuse):
            out['part_pred'] = predLayer(config, features, 'part_pred', config.num_joints)
            if config.location_refinement:
                out['locref_pred'] = predLayer(config, features, 'locref_pred', config.num_joints * 2)