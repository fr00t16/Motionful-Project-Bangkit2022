import tensorflow as tf
import re
#import tensorflow.contrib.slim as tfslim
import tf_slim as tfslim
#from tensorflow.contrib.slim.nets import resnet_v1
from tf_slim.nets import resnet_v1

from toolsetLib.poseDatasetTool import Batch
from toolsetLib import lossesMLToolkit

net_types = {'resnet_50': resnet_v1.resnet_v1_50,
            'resnet_101': resnet_v1.resnet_v1_101}

def predLayer(config, input, name, num_outputs):
    with tfslim.arg_scope([tfslim.conv2d, tfslim.conv2d_transpose], padding='SAME', activation_fn=None, normalizer_fn=None, weights_regularizer=tfslim.l2_regularizer(config.weight_decay)):
        with tf.compat.v1.variable_scope(name):
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

class PoseNet: # PoseNet
    def __init__(self, config):
        self.config = config

    def extractFeatures(self, inputs):
        netFunction = net_types[self.config.net_type]

        mean = tf.constant(self.config.mean_pixel,dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        im_centered = inputs - mean

        with tfslim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, end_points = netFunction(im_centered, is_training=False, global_pool=False)

        return net, end_points   # net: [batch_size, height, width, num_features]

    def predLayer(own, features, endPoints, reuse=None, no_interm=False, scope='pose'): # features: [batch_size, height, width, num_features]
        config = own.config
        numberLayers = re.findall("resnet_([0-9]+)", config.net_type)[0]
        layerName = 'resnet_v1_{}'.format(numberLayers) + '/block{}/unit_{}/bottleneck_v1'
        out = {}
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            out['part_pred'] = predLayer(config, features, 'part_pred', config.num_joints)
            if config.location_refinement:
                out['locref'] = predLayer(config, features, 'locref_pred', config.num_joints * 2)
            if config.pairwise_predict:
                out['pairwise_pred'] = predLayer(config, features, 'pairwise_pred', config.num_joints * (config.num_joints - 1) * 2)
            if config.intermediate_supervision and not no_interm:
                interm_name = layerName.format(3, config.intermediate_supervision_layer)
                block_interm_out = endPoints[interm_name]
                out['part_pred_interm'] = predLayer(config, block_interm_out, 'intermediate_supervision', config.num_joints)
            
        return out

    def get_network(own, input): # input: [batch_size, height, width, 3]
        net, end_points = own.extractFeatures(input)
        return own.predLayer(own, net, end_points, no_interm=True)

    def test(own, input):
        # test network or get_network method
        head = own.get_network(input)
        return own.add_test_losses(head)

    def addTestLayers(own, head): # head: {'part_pred': ..., 'locref': ..., 'pairwise_pred': ...}
        probab = tf.sigmoid(head['part_pred'])
        output = {'part_prob': probab}
        if own.config.location_refinement:
            output['locref'] = head['locref']
        if own.config.pairwise_predict:
            output['pairwise_pred'] = head['pairwise_pred']
        return output

    def partDetectionLoss(own, head, batch, locref, pairwise, intermediate):
        config = own.config
        weightPartPred = config.weight_part_pred
        partScoreWeights = batch[Batch.part_score_weights] if weightPartPred else 1.0

        def addPartLoss(predLayer):
            return tf.compat.v1.losses.sigmoid_cross_entropy(batch[Batch.part_score_targets], head[predLayer], weights=partScoreWeights)

        loss = {}
        loss['part_loss'] = addPartLoss('part_pred_interm')
        totalLoss = totalLoss + loss['part_loss_interm']

        if locref: # location refinement
            locref_pred = head['locref']
            locref_targets = batch[Batch.locref_targets]
            locref_weights = batch[Batch.locref_mask]

            loss_function = lossesMLToolkit.huberLoss if config.locref_huber_loss else tf.compat.v1.losses.mean_squared_error
            loss['locref_loss'] = config.locref_loss_weight * loss_function(locref_targets, locref_pred, locref_weights)
            totalLoss = totalLoss + loss['locref_loss']

        if pairwise: # pairwise loss
            pairwise_pred = head['pairwise_pred']
            pairwise_targets = batch[Batch.pairwise_targets]
            pairwise_weights = batch[Batch.pairwise_mask]

            loss_function = lossesMLToolkit.huberLoss if config.pairwise_huber_loss else tf.compat.v1.losses.mean_squared_error
            loss['pairwise_loss'] = config.pairwise_loss_weight * loss_function(pairwise_targets, pairwise_pred, pairwise_weights)
            totalLoss = totalLoss + loss['pairwise_loss']

        loss['total_loss'] = totalLoss
        return loss

    def train(own, batches):
        config = own.config
        intermediate = config.intermediate_supervision
        locref = config.location_refinement
        pairwise = config.pairwise_predict
        """
train()
  File "/home/albertstarfield/Documents/FileSekolah13(TE)/bangkit_error/runtime/Motionful-Project-Bangkit2022/src/MachineLearning/model/training_mpiimodel.py", line 129, in train
    losses = pose_net(config).train(batches)
  File "/home/albertstarfield/Documents/FileSekolah13(TE)/bangkit_error/runtime/Motionful-Project-Bangkit2022/src/MachineLearning/model/toolsetLib/nnet_toolset_posenet.py", line 125, in train
    head = own.get_network(batches[Batch.image])
  File "/usr/lib/python3.10/enum.py", line 437, in __getattr__
    raise AttributeError(name) from None
AttributeError: image
        """
        head = own.get_network(batches[Batch.inputs])
        return own.partDetectionLoss(head, batches, locref, pairwise, intermediate)