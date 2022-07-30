from __future__ import print_function

import os
import sys
import numpy as np
import cv2
from PIL import Image
import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from kitti_object import *


def save_depth_as_uint16png_upload(img, filename):
    img = np.squeeze(img)
    img = (img * 256.0).astype('uint16')
    img_buffer = img.tobytes()
    imgsave = Image.new("I", img.T.shape)
    imgsave.frombytes(img_buffer, 'raw', "I;16")
    imgsave.save(filename)

def lidar_to_depth(idx_filename, split, sparse_depth_dir):
    if not os.path.exists(sparse_depth_dir):
        os.makedirs(sparse_depth_dir)
    dataset = kitti_object('data/kitti_sfd_seguv_twise', split)

    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    for data_idx in tqdm.tqdm(data_idx_list):
        calib = dataset.get_calibration(data_idx)  # 3 by 4 matrix
        pc_velo = dataset.get_lidar(data_idx)
        pc_rect = np.zeros_like(pc_velo)
        pc_rect[:, 0:3] = calib.project_velo_to_rect(pc_velo[:, 0:3])
        pc_rect[:, 3] = pc_velo[:, 3]
        img = dataset.get_image(data_idx)
        img_height, img_width, img_channel = img.shape
        _, pc_image_coord, img_fov_inds = get_lidar_in_image_fov(pc_velo[:, 0:3],
                                                                 calib, 0, 0, img_width, img_height, True)
        pc_image_coord = pc_image_coord[img_fov_inds].astype(np.int32)[:,::-1]
        pc_rect_xyz = pc_rect[img_fov_inds][:, 0:3]

        sparse_depth_map = np.zeros(img.shape[:-1],np.float32)

        for i in  range(len(pc_image_coord)):
            sparse_depth_map[tuple(pc_image_coord[i])] = pc_rect_xyz[i,2]
            # img[tuple(pc_image_coord[i])] = (0,255,0) if pc_rect_xyz[i,2] > 1 else img[tuple(pc_image_coord[i])]
            
        print(sparse_depth_map.max())
        # cv2.imshow('img',img)
        # cv2.waitKey(0)

        file_name = os.path.join(sparse_depth_dir, '%06d.png'%(data_idx))
        save_depth_as_uint16png_upload(sparse_depth_map, file_name)

print('---------------Start to generate training sparse depth maps---------------')
lidar_to_depth( \
    'data/kitti_sfd_seguv_twise/ImageSets/trainval.txt',
    'training',
    'data/kitti_sfd_seguv_twise/training/depth_sparse',
    )

print('---------------Start to generate testing sparse depth maps---------------')
lidar_to_depth( \
    'data/kitti_sfd_seguv_twise/ImageSets/test.txt',
    'testing',
    'data/kitti_sfd_seguv_twise/testing/depth_sparse',
    )
