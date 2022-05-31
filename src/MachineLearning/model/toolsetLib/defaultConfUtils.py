#!/bin/python3
#imported from the source file about how to train the model (default configuration)
from easydict import EasyDict as ED
conf = ED()

conf.stride = 8.0
conf.weigh_part_predictions = False
conf.weigh_negatives = False
conf.fg_fraction = 0.25
conf.weight_only_present_joints = False
conf.mean_pixel = [123.68, 116.779, 103.939]
conf.shuffle = True
conf.snapshot_prefix = "snapshot"
conf.log_dir = "log"
conf.global_scale = 1.0
conf.location_refinement = False
conf.locref_stdev = 7.2801
conf.locref_loss_weight = 1.0
conf.locref_huber_loss = True
conf.optimizer = "sgd"
conf.intermediate_supervision = False
conf.intermediate_supervision_layer = 12
conf.regularize = False
conf.weight_decay = 0.0001
conf.mirror = False
conf.crop = False
conf.crop_pad = 0
conf.scoremap_dir = "test"
conf.dataset = ""
conf.dataset_type = "default"  # options: "default", "coco"
conf.use_gt_segm = False
conf.batch_size = 1
conf.video = False
conf.video_batch = False
conf.sparse_graph = []
conf.pairwise_stats_collect = False
conf.pairwise_stats_fn = "pairwise_stats.mat"
conf.pairwise_predict = False
conf.pairwise_huber_loss = True
conf.pairwise_loss_weight = 1.0
conf.tensorflow_pairwise_order = True