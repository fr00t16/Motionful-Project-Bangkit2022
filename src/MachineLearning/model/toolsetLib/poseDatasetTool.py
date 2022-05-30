import logging as log
import random as rnd
from enum import Enum
import numpy as np
from numpy import array as arr
from numpy import concatenate as cat
import scipy.io as sio
from scipy.misc import imread, imresize

class Batch(Enum)
    inputs = 0
    part_score_targets = 1
    part_score_weights = 2
    locref_targets = 3
    locref_mask = 4
    pairwise_targets = 5
    pairwise_mask = 6
    data_item = 7