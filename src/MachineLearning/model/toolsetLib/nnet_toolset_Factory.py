#net factory 

from toolsetLib.nnet_toolset_posenet import PoseNet

def pose_net(config):
    if config.video:
        print('This mode does not work!')
        from toolsetLib.nnet_toolset_posenet_seq import PoseSeqNet
        cls = PoseSeqNet
    else:
        cls = PoseNet
    return cls(config)