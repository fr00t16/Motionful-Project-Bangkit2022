#!/bin/python3
#imported from the source file about how to train the model (default configiguration)
from easydict import EasyDict as ED
config = ED()

config.stride = 8.0
config.weight_part_predictions = False
config.weight_negatives = False
config.fg_fraction = 0.25
config.weight_only_present_joints = False
config.mean_pixel = [123.68, 116.779, 103.939]
config.shuffle = True
config.snapshot_prefix = "snapshot"
config.log_dir = "log"
config.global_scale = 1.0
config.location_refinement = False
config.locref_stdev = 7.2801
config.locref_loss_weight = 1.0
config.locref_huber_loss = True
config.optimizer = "sgd" #momentumOptimizer
config.intermediate_supervision = False
config.intermediate_supervision_layer = 12
config.regularize = False
config.weight_decay = 0.0001
config.mirror = False
config.crop = False
config.crop_pad = 0
config.scoremap_dir = "test"
config.dataset = ""
config.dataset_type = "default"  # options: "default", "coco" default means mpii
config.use_gt_segm = False
config.batch_size = 1
config.video = False
config.video_batch = False
config.sparse_graph = []
config.pairwise_stats_collect = False
config.pairwise_stats_fn = "pairwise_stats.mat"
config.pairwise_predict = False
config.pairwise_huber_loss = True
config.pairwise_loss_weight = 1.0
config.tensorflow_pairwise_order = True