''' Helper class and functions for loading KITTI objects

Author: Charles R. Qi
Date: September 2017
'''
from __future__ import print_function

import os
import numpy as np
from PIL import Image
import kitti_util as utils

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3


class kitti_object(object):
    '''Load and parse object data into a usable format.'''
    
    def __init__(self, root_dir, split='training'):
        '''root_dir contains training and testing folders'''
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        if split == 'training':
            self.num_samples = 7481
        elif split == 'testing':
            self.num_samples = 7518
        else:
            print('Unknown split: %s' % (split))
            exit(-1)

        self.image_dir = os.path.join(self.split_dir, 'image_2')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.lidar_dir = os.path.join(self.split_dir, 'velodyne')
        self.label_dir = os.path.join(self.split_dir, 'label_2')

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert(idx<self.num_samples) 
        img_filename = os.path.join(self.image_dir, '%06d.png'%(idx))
        return utils.load_image(img_filename)

    def get_lidar(self, idx): 
        assert(idx<self.num_samples) 
        lidar_filename = os.path.join(self.lidar_dir, '%06d.bin'%(idx))
        return utils.load_velo_scan(lidar_filename)

    def get_calibration(self, idx):
        assert(idx<self.num_samples) 
        calib_filename = os.path.join(self.calib_dir, '%06d.txt'%(idx))
        return utils.Calibration(calib_filename)

    def get_label_objects(self, idx):
        assert(idx<self.num_samples and self.split=='training')
        label_filename = os.path.join(self.label_dir, '%06d.txt'%(idx))
        return utils.read_label(label_filename)

    def get_depth_map(self, idx):
        pass

    def get_top_down(self, idx):
        pass

def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:,0]<xmax) & (pts_2d[:,0]>=xmin) & \
        (pts_2d[:,1]<ymax) & (pts_2d[:,1]>=ymin)
    fov_inds = fov_inds & (pc_velo[:,0]>clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds,:]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo