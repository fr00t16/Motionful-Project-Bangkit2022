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
    def __init__(own, config):
        own.config = config

    def extractFeatures(self, inputs):
        netFunction = net_types[self.config.net_type]

        mean = tf.constant(self.config.mean_pixel,dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        centeredImage = inputs - mean

        print("executing tfslim Netfunc extractFeatures!")
        with tfslim.arg_scope(resnet_v1.resnet_arg_scope()):
            net, end_points = netFunction(centeredImage, global_pool=False, output_stride=16, is_training=False)

        return net, end_points   # net: [batch_size, height, width, num_features]

    def predLayer(own, features, endPoints, reuse=None, no_interm=False, scope='pose'): # features: [batch_size, height, width, num_features]
        config = own.config
        #issues
        #positional arguments:
        """
        (positional arg: 1 2 3 4 5 6)
        predLayer(own, features, endPoints, reuse=None, no_interm=False, scope='pose')
        1: Object <toolsetLib.nnet_toolset_posenet.PoseNet object at 0x7f5393ed25f0> (ignore this, it isnt an issue)
        2: Object <toolsetLib.nnet_toolset_posenet.PoseNet object at 0x7f5393ed25f0> (ignore this, it isnt an issue)
        3: Tensor("resnet_v1_101/block4/unit_3/bottleneck_v1/Relu:0", shape=(1, None, None, 2048), dtype=float32)
        4: some jumbled mess
        5: True
        6: pose (ignore this, it isnt an issue)
        """

        #reuse=None
        #no_interm=False
        numberLayers = re.findall("resnet_([0-9]+)", config.net_type)[0]
        layerName = 'resnet_v1_{}'.format(numberLayers) + '/block{}/unit_{}/bottleneck_v1'
        out = {}
        """
        train()
  File "/home/albertstarfield/Documents/FileSekolah13(TE)/bangkit_error/runtime/Motionful-Project-Bangkit2022/src/MachineLearning/model/training_mpiimodel.py", line 129, in train
    losses = pose_net(config).train(batches)
  File "/home/albertstarfield/Documents/FileSekolah13(TE)/bangkit_error/runtime/Motionful-Project-Bangkit2022/src/MachineLearning/model/toolsetLib/nnet_toolset_posenet.py", line 135, in train
    head = own.get_network(batches[Batch.inputs])
  File "/home/albertstarfield/Documents/FileSekolah13(TE)/bangkit_error/runtime/Motionful-Project-Bangkit2022/src/MachineLearning/model/toolsetLib/nnet_toolset_posenet.py", line 71, in get_network
    return own.predLayer(own, net, end_points, no_interm=True)
  File "/home/albertstarfield/Documents/FileSekolah13(TE)/bangkit_error/runtime/Motionful-Project-Bangkit2022/src/MachineLearning/model/toolsetLib/nnet_toolset_posenet.py", line 56, in predLayer
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
  File "/usr/local/lib/python3.10/dist-packages/tensorflow/python/ops/variable_scope.py", line 2407, in __init__
    raise ValueError("The reuse parameter must be True or False or None.")
ValueError: The reuse parameter must be True or False or None.
        """
        print('reuse var')
        print(reuse)
        print('no_interm')
        print(no_interm)
        print('endPoints')
        print(endPoints)
        print('features')
        print(features)
        print('own')
        print(own)
        print('scope')
        print(scope)
        print('-----------')
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
        #return own.predLayer(own, net, end_points, no_interm=True) #what the hell am i doing? Inputting some weird data into a function that is supposed to return an object
        return own.predLayer(net, end_points)

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
        """
        train()
  File "/home/albertstarfield/Documents/FileSekolah13(TE)/bangkit_error/runtime/Motionful-Project-Bangkit2022/src/MachineLearning/model/training_mpiimodel.py", line 129, in train
    losses = pose_net(config).train(batches)
  File "/home/albertstarfield/Documents/FileSekolah13(TE)/bangkit_error/runtime/Motionful-Project-Bangkit2022/src/MachineLearning/model/toolsetLib/nnet_toolset_posenet.py", line 180, in train
    return own.partDetectionLoss(head, batches, locref, pairwise, intermediate)
  File "/home/albertstarfield/Documents/FileSekolah13(TE)/bangkit_error/runtime/Motionful-Project-Bangkit2022/src/MachineLearning/model/toolsetLib/nnet_toolset_posenet.py", line 133, in partDetectionLoss
    weightPartPred = config.weight_part_pred
AttributeError: 'EasyDict' object has no attribute 'weight_part_pred'. Did you mean: 'weight_part_predictions'?"""
        weightPartPred = config.weight_part_predictions
        partScoreWeights = batch[Batch.part_score_weights] if weightPartPred else 1.0

        def addPartLoss(predLayer):
            return tf.compat.v1.losses.sigmoid_cross_entropy(batch[Batch.part_score_targets], head[predLayer], weights=partScoreWeights)

        loss = {}
        loss['part_loss'] = addPartLoss('part_pred_interm')

        """
        train()
  File "/home/albertstarfield/Documents/FileSekolah13(TE)/bangkit_error/runtime/Motionful-Project-Bangkit2022/src/MachineLearning/model/training_mpiimodel.py", line 129, in train
    losses = pose_net(config).train(batches)
  File "/home/albertstarfield/Documents/FileSekolah13(TE)/bangkit_error/runtime/Motionful-Project-Bangkit2022/src/MachineLearning/model/toolsetLib/nnet_toolset_posenet.py", line 189, in train
    return own.partDetectionLoss(head, batches, locref, pairwise, intermediate)
  File "/home/albertstarfield/Documents/FileSekolah13(TE)/bangkit_error/runtime/Motionful-Project-Bangkit2022/src/MachineLearning/model/toolsetLib/nnet_toolset_posenet.py", line 150, in partDetectionLoss
    totalLoss = totalLoss + loss['part_loss_interm']
        """
        totalLoss = loss['part_loss']
        if intermediate:
            loss['part_loss_interm'] = addPartLoss('part_pred')
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