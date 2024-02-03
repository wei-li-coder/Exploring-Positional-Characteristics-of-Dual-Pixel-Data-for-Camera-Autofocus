import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
from patchify import patchify
from generate_patch import cal_groundtruth_index
import numpy as np
from random import randint

# focal distance in mm
focal_distance = [102.01, 104.23, 106.54, 108.47, 110.99, 113.63, 116.40, 118.73, 121.77,\
                 124.99, 127.69, 131.23, 134.99, 138.98, 142.35, 146.81, 150.59, 155.61,\
                 160.99, 165.57, 171.69, 178.29, 183.96, 191.60, 198.18, 207.10, 216.88,\
                 225.41, 237.08, 247.35, 261.53, 274.13, 291.72, 307.54, 329.95, 350.41,\
                 379.91, 407.40, 447.99, 486.87, 546.23, 605.39, 700.37, 801.09, 935.91,\
                 1185.83, 1508.71, 2289.27, 3910.92]
# to fit the order in supplementary material
focal_distance.reverse()
# to convert in meters
focal_distance = np.array([d / 1000.0 for d in focal_distance])

root_path = "/data/wl/autofocus/learn2focus/dataset/train_set/" 

def read_directory(directory_name):
    cnt = 0 
    right_dir = ['result_up_pd_right_center.png']
    left_dir = ['result_up_pd_left_center.png']
    depth_dir = ['result_merged_depth_center.png']
    conf_dir = ['result_merged_conf_center.exr']
    dirs_1 = os.listdir(directory_name)
    # print(dirs_1)
    dirs_1.sort()
    # dirs_1 = ['train1','train2',...,'train7']
    for dir_1 in dirs_1:
        dir_right_1 = 'raw_up_right_pd/'
        dir_left_1 = 'raw_up_left_pd/'
        dir_depth_1 = 'merged_depth/'
        dir_conf_1 = 'merged_conf/'
        dirs_2 = os.listdir(directory_name + dir_1 + '/' + dir_right_1)
        dirs_2.sort()
        # dirs_2 = ['apt1_0',...] scene name
        for dir_2 in dirs_2:
            mkdir(path = root_path + dir_2)
            mkdir(path = root_path + dir_2 + '/' + 'single')
            mkdir(path = root_path + dir_2 + '/' + 'stack')
            dirs_3 = os.listdir(directory_name + dir_1 + '/' + dir_right_1 + dir_2)
            dirs_3.sort(key=int)
            # dirs_3 = ['0','1',...,'48']
            i = []
            j = [] # random sample index
            # one image sample _ times patches
            for _ in range(400):
                i.append(randint(0, 47))
                j.append(randint(0, 34))
            # for every patch, open pic once
            for cnt_idx in range(100):
                rawimg_dep = cv2.imread(directory_name + dir_1 + '/' + dir_depth_1 + dir_2 + '/' + depth_dir[0], cv2.IMREAD_ANYDEPTH)
                rawimg_dep = cv2.resize(rawimg_dep,dsize=None,fx=4,fy=4,interpolation=cv2.INTER_LINEAR)
                patches_dep = patchify(rawimg_dep, (128,128), step=40)
                rawimg_conf = cv2.imread(directory_name + dir_1 + '/' + dir_conf_1 + dir_2 + '/' + conf_dir[0], cv2.IMREAD_UNCHANGED)
                rawimg_conf = cv2.resize(rawimg_conf[:,:,2],dsize=None,fx=4,fy=4,interpolation=cv2.INTER_LINEAR)
                patches_conf = patchify(rawimg_conf, (128,128), step=40)
                # filter with median confidence for each patch, to remove patches
                if np.median(patches_conf[i[cnt_idx],j[cnt_idx],:,:]) >= 0.99:
                    idx = cal_groundtruth_index(patches_dep[i[cnt_idx],j[cnt_idx],:,:], focal_distance)
                    for dir_3 in dirs_3:
                        rawimg_right = cv2.imread(directory_name +  dir_1 + '/' + dir_right_1 + dir_2 + '/' + dir_3 + '/' + right_dir[0], cv2.IMREAD_UNCHANGED) # read as uint 16
                        # resize to get the same size with depth map  
                        patches_right = patchify(rawimg_right, (128,128), step=40) # (48,35,128,128)
                        rawimg_left = cv2.imread(directory_name +  dir_1 + '/' + dir_left_1 + dir_2 + '/' + dir_3 + '/' + left_dir[0], cv2.IMREAD_UNCHANGED)
                        patches_left = patchify(rawimg_left, (128,128), step=40)
                        # save both dp into one image
                        newimg = np.concatenate((patches_left[i[cnt_idx],j[cnt_idx],:,:], patches_right[i[cnt_idx],j[cnt_idx],:,:]), axis = 1)
                        if dir_3 == '0':
                            imgstack = newimg
                        else:
                            imgstack = np.concatenate((imgstack, newimg), axis = 1)
                        mkdir(path = root_path + dir_2 + '/' + 'single/' + str(idx).rjust(2,'0'))
                        mkdir(path = root_path + dir_2 + '/' + 'stack/' + str(idx).rjust(2,'0'))
                        cv2.imwrite(root_path + dir_2 + '/' + 'single/' + str(idx).rjust(2,'0') + '/' + str(i[cnt_idx]).rjust(2, '0') + '_' + str(j[cnt_idx]).rjust(2, '0') + '_id' + dir_3.rjust(2, '0') + '.png', newimg)
                        cnt += 1
                        print(cnt)
                    cv2.imwrite(root_path + dir_2 + '/' + 'stack/' + str(idx).rjust(2,'0') + '/' + str(i[cnt_idx]).rjust(2, '0') + '_' + str(j[cnt_idx]).rjust(2, '0') + '.png', imgstack)

########## make directory ###############
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

read_directory("/data/wl/autofocus/learn2focus/dataset/train/")