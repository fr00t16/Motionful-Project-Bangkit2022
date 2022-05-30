from toolsetLib.poseDatasetTool import PoseDataset

def createDataset(conf):
    dataset_type = conf.dataset_type
    if dataset_type == "mpii":
        from toolsetLib.mpiiDatasetTool import MPII
        data = MPII(conf)
    else:
        raise Exception("Unknown Dataset sorry about that!")
    return data