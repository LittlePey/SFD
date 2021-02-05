#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# set -x
NGPUS=8
CFG_DIR=cfgs/voxel_rcnn

CFG_NAME=voxel_rcnn_car

python -m torch.distributed.launch --nproc_per_node=${NGPUS} train.py --launcher pytorch --cfg_file $CFG_DIR/$CFG_NAME.yaml --epochs 80 --workers 8


# CFG_NAME=voxel_rcnn_cube_34_test_server_epoch70_run2

# python -m torch.distributed.launch --nproc_per_node=${NGPUS} train.py --launcher pytorch --cfg_file $CFG_DIR/$CFG_NAME.yaml --epochs 70 --workers 8

# python -m torch.distributed.launch --nproc_per_node=${NGPUS} test.py --launcher pytorch --cfg_file $CFG_DIR/$CFG_NAME.yaml --batch_size 8 --ckpt ../output/20200901/$CFG_NAME/default/ckpt/checkpoint_epoch_59.pth --save_to_file

