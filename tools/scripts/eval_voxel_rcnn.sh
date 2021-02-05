#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# set -x
NGPUS=8
CFG_DIR=cfgs

CFG_NAME=voxel_rcnn/voxel_rcnn_car

python -m torch.distributed.launch --nproc_per_node=${NGPUS} test.py --launcher pytorch --cfg_file cfgs/$CFG_NAME.yaml --batch_size 8 --ckpt ../output/$CFG_NAME/default/ckpt/checkpoint_epoch_1.pth

