import os
import numpy as np
from PIL import Image
import tqdm
from kitti_object import *

def depth_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png), filename)
    depth = depth_png.astype(np.float) / 256.
    return depth

def save_depth_as_bin(pc_velo, filename):
    pc_velo.tofile(filename)
    return

def depth_to_lidar(idx_filename, split, dense_depth_dir, pseudo_depth_dir):
    dataset = kitti_object('data/kitti_sfd_seguv_twise', split)

    data_idx_list = [int(line.rstrip()) for line in open(idx_filename)]

    if not os.path.exists(pseudo_depth_dir):
        os.makedirs(pseudo_depth_dir)

    for data_idx in tqdm.tqdm(data_idx_list):
        calib = dataset.get_calibration(data_idx)
        img_rgb = dataset.get_image(data_idx)
        dense_img_filename = os.path.join(dense_depth_dir, '%06d.png'%(data_idx))
        img_depth_small = depth_read(dense_img_filename)
        h, w, _ = img_rgb.shape
        th,tw = img_depth_small.shape
        i = h - th
        j = int(round((w - tw) / 2.))
        img_depth = np.zeros((h, w),dtype=np.float)
        img_depth[i:i + th, j:j + tw] = img_depth_small

        pc_uv = np.stack((img_depth > 0).nonzero(), axis=0).transpose()[:,::-1].astype(np.float32)
        pc_z = img_depth[img_depth > 0].reshape(-1,1).astype(np.float32)
        pc_uvz = np.concatenate([pc_uv,pc_z],axis=-1)
        pc_velo = calib.project_image_to_velo(pc_uvz)

        pc_rgb = img_rgb[img_depth > 0].reshape(-1,3).astype(np.float32)[:,::-1]
        pc_seg = np.ones((pc_rgb.shape[0],1),dtype=pc_velo.dtype)
        pc_velo_rgb_seg_uv = np.concatenate([pc_velo, pc_rgb, pc_seg, pc_uv],axis=-1)
        dense_velo_filename = os.path.join(pseudo_depth_dir, '%06d.bin'%(data_idx))
        save_depth_as_bin(pc_velo_rgb_seg_uv.astype(np.float32), dense_velo_filename)

print('---------------Start to generate training pseudo point clouds---------------')
depth_to_lidar( \
    'data/kitti_sfd_seguv_twise/ImageSets/trainval.txt',
    'training',
    'data/kitti_sfd_seguv_twise/training/depth_dense_twise',
    'data/kitti_sfd_seguv_twise/training/depth_pseudo_rgbseguv_twise'
)

print('---------------Start to generate testing pseudo point clouds---------------')
depth_to_lidar( \
    'data/kitti_sfd_seguv_twise/ImageSets/test.txt',
    'testing',
    'data/kitti_sfd_seguv_twise/testing/depth_dense_twise',
    'data/kitti_sfd_seguv_twise/testing/depth_pseudo_rgbseguv_twise'
)