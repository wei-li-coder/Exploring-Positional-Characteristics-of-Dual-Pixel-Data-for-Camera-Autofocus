import numpy as np
import cv2
from patchify import patchify
import bisect

def get_idx_from_depth(focal_distance, pdepth):
    pdiopters = 1.0 / pdepth
    focal_diopters = np.array([1.0 / d for d in focal_distance])
    return int(round(np.interp(pdiopters, focal_diopters, np.arange(focal_diopters.shape[0]))))

def cal_groundtruth_index(patch, focal_distance):
    patch = patch/255
    # distance in meter
    max = 100
    min = 0.1

    depth = (max * min) / (max - (max - min) * patch)
    index = get_idx_from_depth(focal_distance, np.median(depth))
    return index