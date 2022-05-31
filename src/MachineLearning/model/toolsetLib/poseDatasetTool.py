from ast import Num
import logging as log
import random as rnd
from enum import Enum
from re import I
import numpy as np
from numpy import array as arr
from numpy import concatenate as cat
import scipy.io as sciIO
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


def extendCrop(crop, crop_pad, image_size):
    crop[0] = max(crop[0] - crop_pad, 0)
    crop[1] = max(crop[1] - crop_pad, 0)
    crop[2] = min(crop[2] + crop_pad, image_size[2] - 1)
    crop[3] = min(crop[3] + crop_pad, image_size[1] - 1)
    return crop

def mirrorJointMap(all_joints, num_joints):
    res = np.arange(num_joints)
    symmetric_joints = [p for p in all_joints if len(p) == 2]
    for pair in symmetric_joints:
        res[pair[0]] = pair[1]
        res[pair[1]] = pair[0]
    return res



def getPairwiseStats(joint_id, coords):
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

def data2input(data):
    return np.expand_dims(data, axis=0).astype(float)

def getLoadPairwiseStat(config):
    mat_stats = sciIO.loadmat(config.pairwise_stats_fn)
    pairwise_stats = {}
    for id in range(len(mat_stats['graph'])):
        pair = tuple(mat_stats['graph'][id])
        pairwise_stats[pair] = {"mean": mat_stats['means'][id], "std": mat_stats['std_devs'][id]}
    for pair in pairwise_stats:
        pairwise_stats[pair]["mean"] *= config.global_scale
        pairwise_stats[pair]["std"] *= config.global_scale
    return pairwise_stats


def getIndexPairwise(j_id, j_id_end, num_joints):
    return (num_joints - 1) * j_id + j_id_end - int(j_id < j_id_end)

class DataItem:
    print('stub function DataItem()')
    pass

class PoseDataset:
    def numKeypoints(own):
        return own.config.num_joints

    def setTestMode(own, mode):
        own.has_gt = not mode

    def shuffleSet(own, shuffleInput):
        own.shuffle = shuffleInput

    def getPairwiseStats(own, pairwiseStats):
        own.pairwiseStats = pairwiseStats

    def mirrorCoords(own, joint, imageWidth):
        joint[:, 1] = imageWidth - joint[:, 1] - 1
        return joint

    def jointMirror(own, joint, symmetJoint, imageWidth):
        mir = np.copy(joint)
        mir = own.mirrorCoords(mir, imageWidth)

    def loadDataset(own):
        config = own.config
        fileName = config.dataset
        matLabData = sciIO.loadmat(fileName) #load Matlab *.mat
        own.rawData = matLabData
        matLabData = matLabData['dataset']
        imageCount = matLabData['dataset']
        dataPool = []
        has_gt = True
        for a in range(imageCount):
            sampl = matLabData[0, a]
            items = DataItem() 
            items.image_id = i
            items.image_path = sampl[0][0]
            items.image_size = sampl[1][0]
            if len(sampl) > 2:
                joint = sampl[2][0][0]
                jointIDs = joint[:, 0]
                #if joint ID isn't sized 0
                if jointIDs.size != 0:
                    assert((jointIDs < config.num_joints).any())
                joint[:, 0] = jointIDs
                items.joint = [joint]
            else:
                has_get = False
            if config.crop:
                crop = sampl[3][0] - 1
                items.crop = extendCrop(crop, config.crop_pad, items.im_size)
            dataPool.append(items)
        own.has_gt = has_gt
        return dataPool

    def shuffleImages(own):
        countImages = own.num_images
        if own.config.mirror:
            indicesImage = np.random.permutation(countImages)
            own.mirrored = indicesImage >= countImages
            indicesImage[own.mirrored] = indicesImage[own.mirrored] - countImages
            own.indicesImage = indicesImage
        else:
            own.indicesImage = np.random.permutation(countImages)

    def trainingCountSamples(own):
        count = own.num_images
        if own.config.mirror:
            count *= 2
        return count

    def forwardTrainingSampl(own):
        if own.curr_img == 0 and own.shuffle:
            own.shuffleImages()
        currentImage = own.curr_img
        own.currentImage = (own.currentImage + 1) % own.trainingCountSamples()
        imindex = own.indicesImage[currentImage]
        mirror = own.config.mirror and own.mirrored[currentImage]
        return imindex, mirror

    def trainingGetSample(own, imIndex):
        return own.data[imIndex]

    def getScale(own):
        config = own.config
        scaling = config.global_scale
        if hasattr(config, 'scale_jitter_lo') and hasattr(config, 'scale_jitter_up'):
            scaling *= np.random.uniform(config.scale_jitter_lo, config.scale_jitter_up)
        return scaling
    
    def createBatch(own, dataPoolItems, scaling, mirrors):
        imageFile = dataPoolItems.image_path
        log.debug('Loading image: %s', imageFile)
        log.debug('Scaling: %f', scaling)
        log.debug('Mirror: %s', mirrors)
        log.debug('Image size: %s', dataPoolItems.image_size)
        log.debug('Crop: %s', dataPoolItems.crop)
        imageInput = imread(imageFile, mode='RGB') #RGB usually jpg not png with RGBA
        if own.has_gt:
            joint = np.copy(dataPoolItems.joint)
        if own.config.crop:
            crop = dataPoolItems.crop
            imageInput = imageInput[crop[1]:crop[3], crop[0]:crop[2] + 1, :]
            if own.has_gt:
                joint[:, 1:3] -= crop[0:2].astype(joint.dtype)
        imageResize = imresize(imageInput, scaling) if scaling != 1 else imageInput
        resizedImageSize = arr(imageResize.shape[0:2])
        if mirrors:
            imageResize = np.fliplr(imageResize)
        batches = {Batch.inputs: imageResize}
        if own.has_gt:
            stride = own.config.stride
            if mirrors: #if mirror true
                joint = [own.jointMirror(person_joint, own.symmetric_joints, imageInput.shape[1]) for person_joint in joint]

            sm_size = np.ceil(resizedImageSize / (stride * 2)).astype(np.int) * 2
            jointsScaled = [person_]
                


    def batchNext(own):
        while True: # this is create batch
            imindex, mirrored = own.forwardTrainingSampl()
            itemsDatapool = own.trainingGetSample(imindex)
            scaling = own.getScale()
            if not own.is_valid_size(itemsDatapool.image_size, scaling):
                continue

            return own.createBatch
        

# main class function
    def __init__(own, config):
        print("own attributes debug")
        print(dir(own))
        print("------------------------------")
        own.config = config
        own.data = own.loadDataset() if config.dataset else []
        own.num_images = len(own.data)
        if own.config.mirror:
            own.symmetric_joints = mirrorJointMap(config.all_joints, config.num_joints)
        own.curr_img = 0
        own.set_shuffle(config.shuffle)
        own.set_pairwise_stats_collect(config.pairwise_stats_collect)
        if own.config.pairwise_predict:
            own.pairwise_stats = getLoadPairwiseStat(own.config)

        