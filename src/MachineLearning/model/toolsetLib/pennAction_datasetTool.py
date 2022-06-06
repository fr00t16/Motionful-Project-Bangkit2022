from functool import reduce
import numpy as np
from toolsetLib.poseDatasetTool import PoseDataset, Batch

def mergeBatch(batches):
    """
    Merges n=len(batches) batches of size 1 into
    one batch of size n
    """
    res = {}
    for key, tensor in batches[0].items():
        elements = [batch[key] for batch in batches]
        if type(tensor) is np.ndarray:
            elements = reduce(lambda x, y: np.concatenate((x, y), axis=0), elements)
        res[key] = elements
    return res

class PenAction(PoseDataset):
    def __init__(own, cfg):
        cfg.all_joints = [[0], [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
        cfg.all_joints_names = ["head", "shoulder", "elbow", "wrist", "hip", "knee", "ankle"]
        cfg.num_joints = 13
        super().__init__(cfg)
        own.add_extra_fields()

    def add_extra_fields(own):
        dataset = own.raw_data['dataset']
        for i in range(own.num_images):
            raw_item = dataset[0, i]
            item = own.data[i]
            item.seq_id = raw_item[4][0][0]
            item.frame_id = raw_item[5][0][0]

    def mirror_joint_coords(own, joints, image_width):
        joints[:, 1] = image_width - joints[:, 1] + 1  # 1-indexed
        return joints

    def next_batch(own):
        while True:
            imidx, mirror = own.next_training_sample()
            data_item = own.get_training_sample(imidx)

            scale = own.get_scale()
            if not own.validateSize(data_item.im_size, scale):
                continue

            if own.cfg.video_batch:
                sequences = own.raw_data['sequences']
                seq_ids = sequences[0, data_item.seq_id][0]
                num_frames = len(seq_ids)
                start_frame = data_item.frame_id
                num_frames_model = own.cfg.batch_size

                if start_frame + num_frames_model - 1 >= num_frames:
                    start_frame = num_frames - num_frames_model

                seq_subset = seq_ids[start_frame:start_frame+num_frames_model]
                data_items = [own.get_training_sample(imidx) for imidx in seq_subset]
                batches = [own.createBatch(item, scale, mirror) for item in data_items]

                batch = mergeBatch(batches)
            else:
                batch = own.createBatch(data_item, scale, mirror)

            return batch
