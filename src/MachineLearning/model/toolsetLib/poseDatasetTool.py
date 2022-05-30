import logging as log
import random as rnd
from enum import Enum
import numpy as np
from numpy import array as arr
from numpy import concatenate as cat
import scipy.io as sio
#from scipy.misc import imread, imresize
# This doesn't work due to https://stackoverflow.com/questions/9298665/cannot-import-scipy-misc-imread
from skimage.transform import resize as imresize
from matplotlib.pyplot import imread

class Batch(Enum):
    inputs = 0
    part_score_targets = 1
    part_score_weights = 2
    locref_targets = 3
    locref_mask = 4
    pairwise_targets = 5
    pairwise_mask = 6
    data_item = 7

def mirror_joints_map(all_joints, num_joints):
    res = np.arange(num_joints)
    symmetric_joints = [p for p in all_joints if len(p) == 2]
    for pair in symmetric_joints:
        res[pair[0]] = pair[1]
        res[pair[1]] = pair[0]
    return res


def extend_crop(crop, crop_pad, image_size):
    crop[0] = max(crop[0] - crop_pad, 0)
    crop[1] = max(crop[1] - crop_pad, 0)
    crop[2] = min(crop[2] + crop_pad, image_size[2] - 1)
    crop[3] = min(crop[3] + crop_pad, image_size[1] - 1)
    return crop


def data_to_input(data):
    return np.expand_dims(data, axis=0).astype(float)


def collect_pairwise_stats(joint_id, coords):
    pairwise_stats = {}
    for person_id in range(len(coords)):
        num_joints = len(joint_id[person_id])
        for k_start in range(num_joints):
            j_id_start = joint_id[person_id][k_start]
            joint_pt = coords[person_id][k_start, :]
            j_x_start = np.asscalar(joint_pt[0])
            j_y_start = np.asscalar(joint_pt[1])
            for k_end in range(num_joints):
                if k_start != k_end:
                    j_id_end = joint_id[person_id][k_end]
                    joint_pt = coords[person_id][k_end, :]
                    j_x_end = np.asscalar(joint_pt[0])
                    j_y_end = np.asscalar(joint_pt[1])
                    if (j_id_start, j_id_end) not in pairwise_stats:
                        pairwise_stats[(j_id_start, j_id_end)] = []
                    pairwise_stats[(j_id_start, j_id_end)].append([j_x_end - j_x_start, j_y_end - j_y_start])
    return pairwise_stats


def load_pairwise_stats(config):
    mat_stats = sio.loadmat(config.pairwise_stats_fn)
    pairwise_stats = {}
    for id in range(len(mat_stats['graph'])):
        pair = tuple(mat_stats['graph'][id])
        pairwise_stats[pair] = {"mean": mat_stats['means'][id], "std": mat_stats['std_devs'][id]}
    for pair in pairwise_stats:
        pairwise_stats[pair]["mean"] *= config.global_scale
        pairwise_stats[pair]["std"] *= config.global_scale
    return pairwise_stats


def get_pairwise_index(j_id, j_id_end, num_joints):
    return (num_joints - 1) * j_id + j_id_end - int(j_id < j_id_end)

class DataItem:
    print('stub function DataItem()')
    pass

class PoseDataset:
    def __init__(own, config):
        own.config = config
        own.data = own.load.dataset() if config.dataset else []
        own.num_images = len(own.data)
        if own.config.mirror:
            own.symmetric_joints = mirror_joints_map(config.all_joints, config.num_joints)
        own.curr_img = 0
        own.set_shuffle(config.shuffle)
        own.set_pairwise_stats_collect(config.pairwise_stats_collect)
        if own.config.pairwise_predict:
            own.pairwise_stats = load_pairwise_stats(own.config)

    def load_dataset(own):
        cfg = own.cfg
        file_name = cfg.dataset
        # Load Matlab file dataset annotation
        mlab = sio.loadmat(file_name)
        own.raw_data = mlab
        mlab = mlab['dataset']

        num_images = mlab.shape[1]
        data = []
        has_gt = True

        for i in range(num_images):
            sample = mlab[0, i]

            item = DataItem()
            item.image_id = i
            item.im_path = sample[0][0]
            item.im_size = sample[1][0]
            if len(sample) >= 3:
                joints = sample[2][0][0]
                joint_id = joints[:, 0]
                # make sure joint ids are 0-indexed
                if joint_id.size != 0:
                    assert((joint_id < cfg.num_joints).any())
                joints[:, 0] = joint_id
                item.joints = [joints]
            else:
                has_gt = False
            if cfg.crop:
                crop = sample[3][0] - 1
                item.crop = extend_crop(crop, cfg.crop_pad, item.im_size)
            data.append(item)

        own.has_gt = has_gt
        return data

    def num_keypoints(own):
        return own.cfg.num_joints

    def set_test_mode(own, test_mode):
        own.has_gt = not test_mode


    def set_shuffle(own, shuffle):
        own.shuffle = shuffle
        if not shuffle:
            assert not own.cfg.mirror
            own.image_indices = np.arange(own.num_images)


    def set_pairwise_stats_collect(own, pairwise_stats_collect):
        own.pairwise_stats_collect = pairwise_stats_collect
        if own.pairwise_stats_collect:
            assert own.get_scale() == 1.0


    def mirror_joint_coords(own, joints, image_width):
        # horizontally flip the x-coordinate, keep y unchanged
        joints[:, 1] = image_width - joints[:, 1] - 1
        return joints


    def mirror_joints(own, joints, symmetric_joints, image_width):
        # joint ids are 0 indexed
        res = np.copy(joints)
        res = own.mirror_joint_coords(res, image_width)
        # swap the joint_id for a symmetric one
        joint_id = joints[:, 0].astype(int)
        res[:, 0] = symmetric_joints[joint_id]
        return res


    def shuffle_images(own):
        num_images = own.num_images
        if own.cfg.mirror:
            image_indices = np.random.permutation(num_images * 2)
            own.mirrored = image_indices >= num_images
            image_indices[own.mirrored] = image_indices[own.mirrored] - num_images
            own.image_indices = image_indices
        else:
            own.image_indices = np.random.permutation(num_images)


    def num_training_samples(own):
        num = own.num_images
        if own.cfg.mirror:
            num *= 2
        return num


    def next_training_sample(own):
        if own.curr_img == 0 and own.shuffle:
            own.shuffle_images()

        curr_img = own.curr_img
        own.curr_img = (own.curr_img + 1) % own.num_training_samples()

        imidx = own.image_indices[curr_img]
        mirror = own.cfg.mirror and own.mirrored[curr_img]

        return imidx, mirror


    def get_training_sample(own, imidx):
        return own.data[imidx]


    def get_scale(own):
        cfg = own.cfg
        scale = cfg.global_scale
        if hasattr(cfg, 'scale_jitter_lo') and hasattr(cfg, 'scale_jitter_up'):
            scale_jitter = rand.uniform(cfg.scale_jitter_lo, cfg.scale_jitter_up)
            scale *= scale_jitter
        return scale


    def next_batch(own):
        while True:
            imidx, mirror = own.next_training_sample()
            data_item = own.get_training_sample(imidx)
            scale = own.get_scale()

            if not own.is_valid_size(data_item.im_size, scale):
                continue

            return own.make_batch(data_item, scale, mirror)


    def is_valid_size(own, image_size, scale):
        im_width = image_size[2]
        im_height = image_size[1]

        max_input_size = 100
        if im_height < max_input_size or im_width < max_input_size:
            return False

        if hasattr(own.cfg, 'max_input_size'):
            max_input_size = own.cfg.max_input_size
            input_width = im_width * scale
            input_height = im_height * scale
            if input_height * input_width > max_input_size * max_input_size:
                return False

        return True


    def make_batch(own, data_item, scale, mirror):
        im_file = data_item.im_path
        logging.debug('image %s', im_file)
        logging.debug('mirror %r', mirror)
        image = imread(im_file, mode='RGB')

        if own.has_gt:
            joints = np.copy(data_item.joints)

        if own.cfg.crop:
            crop = data_item.crop
            image = image[crop[1]:crop[3] + 1, crop[0]:crop[2] + 1, :]
            if own.has_gt:
                joints[:, 1:3] -= crop[0:2].astype(joints.dtype)

        img = imresize(image, scale) if scale != 1 else image
        scaled_img_size = arr(img.shape[0:2])

        if mirror:
            img = np.fliplr(img)

        batch = {Batch.inputs: img}

        if own.has_gt:
            stride = own.cfg.stride

            if mirror:
                joints = [own.mirror_joints(person_joints, own.symmetric_joints, image.shape[1]) for person_joints in
                          joints]

            sm_size = np.ceil(scaled_img_size / (stride * 2)).astype(int) * 2

            scaled_joints = [person_joints[:, 1:3] * scale for person_joints in joints]

            joint_id = [person_joints[:, 0].astype(int) for person_joints in joints]
            batch = own.compute_targets_and_weights(joint_id, scaled_joints, data_item, sm_size, scale, batch)

            if own.pairwise_stats_collect:
                data_item.pairwise_stats = collect_pairwise_stats(joint_id, scaled_joints)

        batch = {key: data_to_input(data) for (key, data) in batch.items()}

        batch[Batch.data_item] = data_item

        return batch


    def set_locref(own, locref_map, locref_mask, locref_scale, i, j, j_id, dx, dy):
        locref_mask[j, i, j_id * 2 + 0] = 1
        locref_mask[j, i, j_id * 2 + 1] = 1
        locref_map[j, i, j_id * 2 + 0] = dx * locref_scale
        locref_map[j, i, j_id * 2 + 1] = dy * locref_scale


    def set_pairwise_map(own, pairwise_map, pairwise_mask, i, j, j_id, j_id_end, coords, pt_x, pt_y, person_id, k_end):
        num_joints = own.cfg.num_joints
        joint_pt = coords[person_id][k_end, :]
        j_x_end = np.asscalar(joint_pt[0])
        j_y_end = np.asscalar(joint_pt[1])
        pair_id = get_pairwise_index(j_id, j_id_end, num_joints)
        stats = own.pairwise_stats[(j_id, j_id_end)]
        dx = j_x_end - pt_x
        dy = j_y_end - pt_y
        pairwise_mask[j, i, pair_id * 2 + 0] = 1
        pairwise_mask[j, i, pair_id * 2 + 1] = 1
        pairwise_map[j, i, pair_id * 2 + 0] = (dx - stats["mean"][0]) / stats["std"][0]
        pairwise_map[j, i, pair_id * 2 + 1] = (dy - stats["mean"][1]) / stats["std"][1]


    def compute_targets_and_weights(own, joint_id, coords, data_item, size, scale, batch):
        stride = own.cfg.stride
        dist_thresh = own.cfg.pos_dist_thresh * scale
        num_joints = own.cfg.num_joints
        half_stride = stride / 2
        scmap = np.zeros(cat([size, arr([num_joints])]))

        locref_shape = cat([size, arr([num_joints * 2])])
        locref_mask = np.zeros(locref_shape)
        locref_map = np.zeros(locref_shape)

        pairwise_shape = cat([size, arr([num_joints * (num_joints - 1) * 2])])
        pairwise_mask = np.zeros(pairwise_shape)
        pairwise_map = np.zeros(pairwise_shape)

        dist_thresh_sq = dist_thresh ** 2

        width = size[1]
        height = size[0]

        for person_id in range(len(coords)):
            for k, j_id in enumerate(joint_id[person_id]):
                joint_pt = coords[person_id][k, :]
                j_x = np.asscalar(joint_pt[0])
                j_y = np.asscalar(joint_pt[1])

                # don't loop over entire heatmap, but just relevant locations
                j_x_sm = round((j_x - half_stride) / stride)
                j_y_sm = round((j_y - half_stride) / stride)
                min_x = round(max(j_x_sm - dist_thresh - 1, 0))
                max_x = round(min(j_x_sm + dist_thresh + 1, width - 1))
                min_y = round(max(j_y_sm - dist_thresh - 1, 0))
                max_y = round(min(j_y_sm + dist_thresh + 1, height - 1))

                for j in range(min_y, max_y + 1):  # range(height):
                    pt_y = j * stride + half_stride
                    for i in range(min_x, max_x + 1):  # range(width):
                        # pt = arr([i*stride+half_stride, j*stride+half_stride])
                        # diff = joint_pt - pt
                        # The code above is too slow in python
                        pt_x = i * stride + half_stride
                        dx = j_x - pt_x
                        dy = j_y - pt_y
                        dist = dx ** 2 + dy ** 2
                        # print(la.norm(diff))

                        if dist <= dist_thresh_sq:
                            dist = dx ** 2 + dy ** 2
                            locref_scale = 1.0 / own.cfg.locref_stdev
                            current_normalized_dist = dist * locref_scale ** 2
                            prev_normalized_dist = locref_map[j, i, j_id * 2 + 0] ** 2 + \
                                                   locref_map[j, i, j_id * 2 + 1] ** 2
                            update_scores = (scmap[j, i, j_id] == 0) or prev_normalized_dist > current_normalized_dist
                            if own.cfg.location_refinement and update_scores:
                                own.set_locref(locref_map, locref_mask, locref_scale, i, j, j_id, dx, dy)
                            if own.cfg.pairwise_predict and update_scores:
                                for k_end, j_id_end in enumerate(joint_id[person_id]):
                                    if k != k_end:
                                        own.set_pairwise_map(pairwise_map, pairwise_mask, i, j, j_id, j_id_end,
                                                              coords, pt_x, pt_y, person_id, k_end)
                            scmap[j, i, j_id] = 1

        scmap_weights = own.compute_scmap_weights(scmap.shape, joint_id, data_item)

        # Update batch
        batch.update({
            Batch.part_score_targets: scmap,
            Batch.part_score_weights: scmap_weights
        })
        if own.cfg.location_refinement:
            batch.update({
                Batch.locref_targets: locref_map,
                Batch.locref_mask: locref_mask
            })
        if own.cfg.pairwise_predict:
            batch.update({
                Batch.pairwise_targets: pairwise_map,
                Batch.pairwise_mask: pairwise_mask
            })

        return batch


    def compute_scmap_weights(own, scmap_shape, joint_id, data_item):
        cfg = own.cfg
        if cfg.weigh_only_present_joints:
            weights = np.zeros(scmap_shape)
            for person_joint_id in joint_id:
                for j_id in person_joint_id:
                    weights[:, :, j_id] = 1.0
        else:
            weights = np.ones(scmap_shape)
        return weights