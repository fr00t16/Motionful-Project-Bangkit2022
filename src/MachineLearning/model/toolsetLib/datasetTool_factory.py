from toolsetLib.poseDatasetTool import PoseDataset

# explanation of this function is about selecting dataset and determining whether it is supported or not
def createDataset(conf):
    dataset_type = conf.dataset_type
    if dataset_type == "mpii":
        from toolsetLib.mpiiDatasetTool import MPII #this might be the root why Load attribute not found
        data = MPII(conf) # this is where attributes defined in conf are used
    else:
        raise Exception("Unknown Dataset sorry about that!")
    return data