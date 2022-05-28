import torch
import torch.nn as nn
from .roi_head_template import RoIHeadTemplate
from ...utils import common_utils, spconv_utils
from ...ops.pointnet2.pointnet2_stack import voxel_pool_modules as voxelpool_stack_modules
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
import spconv
import numpy as np
import torch.nn.functional as F
from ...utils import box_coder_utils, common_utils, loss_utils, box_utils
import time
import cv2
from ...ops.iou3d import oriented_iou_loss
from ...ops.iou3d_nms import iou3d_nms_utils
from .sfd_head_utils import GAF, CPConvs

class SFDHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, point_cloud_range, voxel_size, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.pool_cfg = model_cfg.ROI_GRID_POOL
        LAYER_cfg = self.pool_cfg.POOL_LAYERS
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        
        # RoI Grid Pool
        c_out = 0
        self.roi_grid_pool_layers = nn.ModuleList()
        for src_name in self.pool_cfg.FEATURES_SOURCE:
            mlps = LAYER_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [input_channels[src_name]] + mlps[k]
            pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                query_ranges=LAYER_cfg[src_name].QUERY_RANGES,
                nsamples=LAYER_cfg[src_name].NSAMPLE,
                radii=LAYER_cfg[src_name].POOL_RADIUS,
                mlps=mlps, 
                pool_method=LAYER_cfg[src_name].POOL_METHOD,
            )
            self.roi_grid_pool_layers.append(pool_layer)
            c_out += sum([x[-1] for x in mlps])
        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
    
        # RoI Point Pool
        self.cpconvs_layer = CPConvs()
        self.coords_3x3        = torch.tensor([ [0, 0], 
        [-1,  0], [-1,  1], 
        [ 0,  1], [ 1,  1], 
        [ 1,  0], [ 1, -1], 
        [ 0, -1], [-1, -1]]).long()
        self.coords_5x5_dilate = torch.tensor([ [0, 0], 
        [-2,  0], [-2,  2], 
        [ 0,  2], [ 2,  2], 
        [ 2,  0], [ 2, -2], 
        [ 0, -2], [-2, -2]]).long()
        self.coords_9x9_dilate = torch.tensor([ [0, 0], 
        [-2,  0], [-2,  2], 
        [ 0,  2], [ 2,  2], 
        [ 2,  0], [ 2, -2], 
        [ 0, -2], [-2, -2],
        [-4, -2], [-4,  0], [-4,  2], [-4,  4],
        [-2,  4], [ 0,  4], [ 2,  4], [ 4,  4],
        [ 4,  2], [ 4,  0], [ 4, -2], [ 4, -4],
        [ 2, -4], [ 0, -4], [-2, -4], [-4, -4]]).long()
        self.pointnet_kernel_size = {'coords_3x3':3, 'coords_5x5_dilate':5, 'coords_9x9_dilate':9}

        # RoI Voxel Pool
        self.roiaware_pool3d_layer = roiaware_pool3d_utils.RoIAwarePool3d(
            out_size=self.model_cfg.ROI_AWARE_POOL.POOL_SIZE,
            max_pts_each_voxel=self.model_cfg.ROI_AWARE_POOL.MAX_POINTS_PER_VOXEL
        )
        block = self.post_act_block
        c0 = self.model_cfg.ROI_AWARE_POOL.NUM_FEATURES_RAW
        c1 = self.model_cfg.ROI_AWARE_POOL.NUM_FEATURES
        self.conv_pseudo = spconv.SparseSequential(
            block(c0, c0, 3, padding=1, indice_key='rcnn_subm1'),
            block(c0, c1, 3, stride=2, padding=1, indice_key='rcnn_spconv1', conv_type='spconv'),
            block(c1, c1, 3, padding=1, indice_key='rcnn_subm2'),
        )

        self.fusion_layer = GAF(c_out, c1, self.model_cfg.ATTENTION_FC[0])
        shared_pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out * 2
        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Linear(shared_pre_channel, self.model_cfg.SHARED_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU(inplace=True)
            ])
            shared_pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        cls_fc_list = []
        cls_pre_channel = shared_pre_channel
        for k in range(0, self.model_cfg.CLS_FC.__len__()):
            cls_fc_list.extend([
                nn.Linear(cls_pre_channel, self.model_cfg.CLS_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.CLS_FC[k]),
                nn.ReLU()
            ])
            cls_pre_channel = self.model_cfg.CLS_FC[k]

            if k != self.model_cfg.CLS_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                cls_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.cls_fc_layers = nn.Sequential(*cls_fc_list)
        self.cls_pred_layer = nn.Linear(cls_pre_channel, self.num_class, bias=True)

        reg_fc_list = []
        reg_pre_channel = shared_pre_channel
        for k in range(0, self.model_cfg.REG_FC.__len__()):
            reg_fc_list.extend([
                nn.Linear(reg_pre_channel, self.model_cfg.REG_FC[k], bias=False),
                nn.BatchNorm1d(self.model_cfg.REG_FC[k]),
                nn.ReLU()
            ])
            reg_pre_channel = self.model_cfg.REG_FC[k]

            if k != self.model_cfg.REG_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                reg_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))
        self.reg_fc_layers = nn.Sequential(*reg_fc_list)
        self.reg_pred_layer = nn.Linear(reg_pre_channel, self.box_coder.code_size * self.num_class, bias=True)

        if self.training:
            shared_fc_list_pseudo = []
            pre_channel_pseudo = GRID_SIZE * GRID_SIZE * GRID_SIZE * c1
            for k in range(0, self.model_cfg.SHARED_FC_PSEUDO.__len__()):
                shared_fc_list_pseudo.extend([
                    nn.Linear(pre_channel_pseudo, self.model_cfg.SHARED_FC_PSEUDO[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.SHARED_FC_PSEUDO[k]),
                    nn.ReLU(inplace=True)
                ])
                pre_channel_pseudo = self.model_cfg.SHARED_FC_PSEUDO[k]

                if k != self.model_cfg.SHARED_FC_PSEUDO.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    shared_fc_list_pseudo.append(nn.Dropout(self.model_cfg.DP_RATIO))
            self.shared_fc_layer_pseudo = nn.Sequential(*shared_fc_list_pseudo)   

            cls_fc_list_pseudo = []
            cls_pre_channel_pseudo = pre_channel_pseudo
            for k in range(0, self.model_cfg.CLS_FC_PSEUDO.__len__()):
                cls_fc_list_pseudo.extend([
                    nn.Linear(cls_pre_channel_pseudo, self.model_cfg.CLS_FC_PSEUDO[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.CLS_FC_PSEUDO[k]),
                    nn.ReLU()
                ])
                cls_pre_channel_pseudo = self.model_cfg.CLS_FC_PSEUDO[k]

                if k != self.model_cfg.CLS_FC_PSEUDO.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    cls_fc_list_pseudo.append(nn.Dropout(self.model_cfg.DP_RATIO))
            self.cls_fc_layers_pseudo = nn.Sequential(*cls_fc_list_pseudo)
            self.cls_pred_layer_pseudo = nn.Linear(cls_pre_channel_pseudo, self.num_class, bias=True)

            reg_fc_list_pseudo = []
            reg_pre_channel_pseudo = pre_channel_pseudo
            for k in range(0, self.model_cfg.REG_FC_PSEUDO.__len__()):
                reg_fc_list_pseudo.extend([
                    nn.Linear(reg_pre_channel_pseudo, self.model_cfg.REG_FC_PSEUDO[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.REG_FC_PSEUDO[k]),
                    nn.ReLU()
                ])
                reg_pre_channel_pseudo = self.model_cfg.REG_FC_PSEUDO[k]

                if k != self.model_cfg.REG_FC_PSEUDO.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    reg_fc_list_pseudo.append(nn.Dropout(self.model_cfg.DP_RATIO))
            self.reg_fc_layers_pseudo = nn.Sequential(*reg_fc_list_pseudo)
            self.reg_pred_layer_pseudo = nn.Linear(reg_pre_channel_pseudo, self.model_cfg.AUXILIARY_CODE_SIZE * self.num_class, bias=True)


            shared_fc_list_valid = []
            pre_channel_valid = GRID_SIZE * GRID_SIZE * GRID_SIZE * c1
            for k in range(0, self.model_cfg.SHARED_FC_VALID.__len__()):
                shared_fc_list_valid.extend([
                    nn.Linear(pre_channel_valid, self.model_cfg.SHARED_FC_VALID[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.SHARED_FC_VALID[k]),
                    nn.ReLU(inplace=True)
                ])
                pre_channel_valid = self.model_cfg.SHARED_FC_VALID[k]

                if k != self.model_cfg.SHARED_FC_VALID.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    shared_fc_list_valid.append(nn.Dropout(self.model_cfg.DP_RATIO))
            self.shared_fc_layer_valid = nn.Sequential(*shared_fc_list_valid)   

            cls_fc_list_valid = []
            cls_pre_channel_valid = pre_channel_valid
            for k in range(0, self.model_cfg.CLS_FC_VALID.__len__()):
                cls_fc_list_valid.extend([
                    nn.Linear(cls_pre_channel_valid, self.model_cfg.CLS_FC_VALID[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.CLS_FC_VALID[k]),
                    nn.ReLU()
                ])
                cls_pre_channel_valid = self.model_cfg.CLS_FC_VALID[k]

                if k != self.model_cfg.CLS_FC_VALID.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    cls_fc_list_valid.append(nn.Dropout(self.model_cfg.DP_RATIO))
            self.cls_fc_layers_valid = nn.Sequential(*cls_fc_list_valid)
            self.cls_pred_layer_valid = nn.Linear(cls_pre_channel_valid, self.num_class, bias=True)

            reg_fc_list_valid = []
            reg_pre_channel_valid = pre_channel_valid
            for k in range(0, self.model_cfg.REG_FC_VALID.__len__()):
                reg_fc_list_valid.extend([
                    nn.Linear(reg_pre_channel_valid, self.model_cfg.REG_FC_VALID[k], bias=False),
                    nn.BatchNorm1d(self.model_cfg.REG_FC_VALID[k]),
                    nn.ReLU()
                ])
                reg_pre_channel_valid = self.model_cfg.REG_FC_VALID[k]

                if k != self.model_cfg.REG_FC_VALID.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                    reg_fc_list_valid.append(nn.Dropout(self.model_cfg.DP_RATIO))
            self.reg_fc_layers_valid = nn.Sequential(*reg_fc_list_valid)
            self.reg_pred_layer_valid = nn.Linear(reg_pre_channel_valid, self.model_cfg.AUXILIARY_CODE_SIZE * self.num_class, bias=True)

        self.init_weights()
        self.build_losses_pseudo(self.model_cfg.LOSS_CONFIG)
        self.build_losses_valid(self.model_cfg.LOSS_CONFIG)

    def build_losses_pseudo(self, losses_cfg):
        self.add_module(
            'reg_loss_func_pseudo',
            loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg.LOSS_WEIGHTS_PSEUDO['code_weights'])
        )

    def build_losses_valid(self, losses_cfg):
        self.add_module(
            'reg_loss_func_valid',
            loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg.LOSS_WEIGHTS_VALID['code_weights'])
        )

    def init_weights(self):
        init_func = nn.init.xavier_normal_

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # main head
        for module_list in [self.shared_fc_layer, self.cls_fc_layers, self.reg_fc_layers]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.cls_pred_layer.weight, 0, 0.01)
        nn.init.constant_(self.cls_pred_layer.bias, 0)
        nn.init.normal_(self.reg_pred_layer.weight, mean=0, std=0.001)
        nn.init.constant_(self.reg_pred_layer.bias, 0)

        # pseudo head
        for module_list in [self.shared_fc_layer_pseudo, self.cls_fc_layers_pseudo, self.reg_fc_layers_pseudo]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.cls_pred_layer_pseudo.weight, 0, 0.01)
        nn.init.constant_(self.cls_pred_layer_pseudo.bias, 0)
        nn.init.normal_(self.reg_pred_layer_pseudo.weight, mean=0, std=0.001)
        nn.init.constant_(self.reg_pred_layer_pseudo.bias, 0)

        # valid head
        for module_list in [self.shared_fc_layer_valid, self.cls_fc_layers_valid, self.reg_fc_layers_valid]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    init_func(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.cls_pred_layer_valid.weight, 0, 0.01)
        nn.init.constant_(self.cls_pred_layer_valid.bias, 0)
        nn.init.normal_(self.reg_pred_layer_valid.weight, mean=0, std=0.001)
        nn.init.constant_(self.reg_pred_layer_valid.bias, 0)

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        rois = batch_dict['rois']
        batch_size = batch_dict['batch_size']
        with_vf_transform = batch_dict.get('with_voxel_feature_transform', False)

        # (BxN, 6x6x6, 3)
        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            rois, grid_size=self.pool_cfg.GRID_SIZE
        )  
        # (B, Nx6x6x6, 3)
        roi_grid_xyz = roi_grid_xyz.view(batch_size, -1, 3)  

        # (B, Nx6x6x6, 3) compute the voxel coordinates of grid points
        roi_grid_coords_x = (roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]) // self.voxel_size[0]
        roi_grid_coords_y = (roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]) // self.voxel_size[1]
        roi_grid_coords_z = (roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]) // self.voxel_size[2]
        roi_grid_coords = torch.cat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], dim=-1)

        # (B, Nx6x6x6, 1) compute the batch index of grid points
        batch_idx = rois.new_zeros(batch_size, roi_grid_coords.shape[1], 1)
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
            
        # (B) [Nx6x6x6, Nx6x6x6, ..., Nx6x6x6]
        roi_grid_batch_cnt = rois.new_zeros(batch_size).int().fill_(roi_grid_coords.shape[1])

        #  grouper --> query range --> 
        pooled_features_list = []
        for k, src_name in enumerate(self.pool_cfg.FEATURES_SOURCE):
            pool_layer = self.roi_grid_pool_layers[k]
            cur_stride = batch_dict['multi_scale_3d_strides'][src_name]
            cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]
            if with_vf_transform:
                cur_sp_tensors = batch_dict['multi_scale_3d_features_post'][src_name]
            else:
                cur_sp_tensors = batch_dict['multi_scale_3d_features'][src_name]

            # [18196, 4] / [8372, 4]
            cur_coords = cur_sp_tensors.indices 

            # [18196, 3] / [8372, 3]
            cur_voxel_xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_times=cur_stride,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            ) 

            # 18196 / 8372
            cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
            for bs_idx in range(batch_size):
                cur_voxel_xyz_batch_cnt[bs_idx] = (cur_coords[:, 0] == bs_idx).sum()

            # [1, 11, 400, 352] / [1, 5, 200, 176]
            v2p_ind_tensor = spconv_utils.generate_voxel2pinds(cur_sp_tensors)

            # [1, 27648, 4]
            cur_roi_grid_coords = roi_grid_coords // cur_stride
            cur_roi_grid_coords = torch.cat([batch_idx, cur_roi_grid_coords], dim=-1)
            cur_roi_grid_coords = cur_roi_grid_coords.int()

            # [27648, 64]
            pooled_features = pool_layer(
                xyz=cur_voxel_xyz.contiguous(),
                xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                new_xyz=roi_grid_xyz.contiguous().view(-1, 3),
                new_xyz_batch_cnt=roi_grid_batch_cnt,
                new_coords=cur_roi_grid_coords.contiguous().view(-1, 4),
                features=cur_sp_tensors.features.contiguous(),
                voxel2point_indices=v2p_ind_tensor
            )

            # [BxN, 6x6x6, 64]
            pooled_features = pooled_features.view(
                -1, self.pool_cfg.GRID_SIZE ** 3,
                pooled_features.shape[-1]
            )  
            pooled_features_list.append(pooled_features)
        
        # [BxN, 6x6x6, 128]
        ms_pooled_features = torch.cat(pooled_features_list, dim=-1)
        
        return ms_pooled_features

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def post_act_block(self, in_channels, out_channels, kernel_size, indice_key, stride=1, padding=0, conv_type='subm'):
        if conv_type == 'subm':
            m = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )
        elif conv_type == 'spconv':
            m = spconv.SparseSequential(
                spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                    bias=False, indice_key=indice_key),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )
        elif conv_type == 'inverseconv':
            m = spconv.SparseSequential(
                spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size,
                                           indice_key=indice_key, bias=False),
                nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            )
        else:
            raise NotImplementedError
        return m

    def roiaware_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size     = batch_dict['batch_size']
        rois           = batch_dict['rois']
        batch_idx        = batch_dict['points_pseudo_features'][:, 0]
        point_coords   = batch_dict['points_pseudo_features'][:, 1:4]
        point_features = batch_dict['points_pseudo_features'][:, 4:]          

        pooled_pseudo_features_list = []
        for bs_idx in range(batch_size):
            bs_mask = (batch_idx == bs_idx)
            if bs_mask.sum() == 0:
                cur_point_coords = point_coords.new_zeros((1,point_coords.shape[-1]))
                cur_pseudo_features = point_features.new_zeros((1,point_features.shape[-1]))
            else:
                cur_point_coords = point_coords[bs_mask]
                cur_pseudo_features = point_features[bs_mask]

            cur_roi = rois[bs_idx][:, 0:7].contiguous()  # (N, 7)

            pooled_pseudo_features = self.roiaware_pool3d_layer.forward(
                cur_roi, cur_point_coords, cur_pseudo_features, pool_method=self.model_cfg.ROI_AWARE_POOL.POOL_METHOD
            )  # (N, out_x, out_y, out_z, C)

            pooled_pseudo_features_list.append(pooled_pseudo_features)
        pooled_pseudo_features = torch.cat(pooled_pseudo_features_list, dim=0)  # (B * N, out_x, out_y, out_z, C)

        return  pooled_pseudo_features

    @staticmethod
    def fake_sparse_idx(sparse_idx, batch_size_rcnn):
        print('Warning: Sparse_Idx_Shape(%s) \r' % (str(sparse_idx.shape)), end='', flush=True)
        # at most one sample is non-empty, then fake the first voxels of each sample(BN needs at least
        # two values each channel) as non-empty for the below calculation
        sparse_idx = sparse_idx.new_zeros((batch_size_rcnn, 3))
        bs_idxs = torch.arange(batch_size_rcnn).type_as(sparse_idx).view(-1, 1)
        sparse_idx = torch.cat((bs_idxs, sparse_idx), dim=1)
        return sparse_idx

    def roicrop3d_gpu(self, batch_dict, pool_extra_width):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size     = batch_dict['batch_size']
        rois           = batch_dict['rois']
        num_rois       = rois.shape[1]

        enlarged_rois = box_utils.enlarge_box3d(rois.view(-1, 7).clone().detach(), pool_extra_width).view(batch_size, -1, 7) 
        batch_idx      = batch_dict['points_pseudo'][:, 0]
        point_coords   = batch_dict['points_pseudo'][:, 1:4]
        point_features = batch_dict['points_pseudo'][:,1:]    # N, 8{x,y,z,r,g,b,u,v}                          

        point_depths = point_coords.norm(dim=1) / self.model_cfg.ROI_POINT_CROP.DEPTH_NORMALIZER - 0.5
        point_features_list = [point_features, point_depths[:, None]]
        point_features = torch.cat(point_features_list, dim=1)   
        w, h = 1400, 400

        with torch.no_grad():
            total_pts_roi_index = []
            total_pts_batch_index = []
            total_pts_features = []
            for bs_idx in range(batch_size):
                bs_mask          = (batch_idx == bs_idx)
                cur_point_coords = point_coords[bs_mask]
                cur_features     = point_features[bs_mask]
                cur_roi          = enlarged_rois[bs_idx][:, 0:7].contiguous()

                box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(       
                    cur_point_coords.unsqueeze(0), cur_roi.unsqueeze(0)
                )      
                cur_box_idxs_of_pts = box_idxs_of_pts[0]

                points_in_rois = cur_box_idxs_of_pts != -1
                cur_box_idxs_of_pts = cur_box_idxs_of_pts[points_in_rois] + num_rois * bs_idx

                cur_pts_batch_index = cur_box_idxs_of_pts.new_zeros((cur_box_idxs_of_pts.shape[0]))
                cur_pts_batch_index[:] = bs_idx

                cur_features        = cur_features[points_in_rois]

                total_pts_roi_index.append(cur_box_idxs_of_pts)
                total_pts_batch_index.append(cur_pts_batch_index)
                total_pts_features.append(cur_features)

            total_pts_roi_index     =  torch.cat(total_pts_roi_index, dim=0)
            total_pts_batch_index =  torch.cat(total_pts_batch_index, dim=0)
            total_pts_features      =  torch.cat(total_pts_features, dim=0)
            total_pts_features_xyz_src = total_pts_features.clone()[...,:3]
            total_pts_rois = torch.index_select(rois.view(-1,7), 0, total_pts_roi_index.long())

            total_pts_features[:, 0:3] -= total_pts_rois[:, 0:3]
            total_pts_features[:, 0:3] = common_utils.rotate_points_along_z(
                total_pts_features[:, 0:3].unsqueeze(dim=1), -total_pts_rois[:, 6]
            ).squeeze(dim=1)          
            total_pts_features_raw = total_pts_features.clone()
            global_dv = total_pts_roi_index * h 
            total_pts_features[:, 7] += global_dv

        image = total_pts_features.new_zeros((batch_size*num_rois*h, w)).long()  
        global_index = torch.arange(1, total_pts_features.shape[0]+1)
        image[total_pts_features[:,7].long(), total_pts_features[:,6].long()] = global_index.to(device=total_pts_features.device)

        coords = getattr(self, self.model_cfg.ROI_AWARE_POOL.KERNEL_TYPE)
        points_list = []
        for circle_i in range(len(coords)):
            dx, dy = coords[circle_i]
            points_cur = image[total_pts_features[:, 7].long() + dx, total_pts_features[:, 6].long() + dy]
            points_list.append(points_cur)
        total_pts_neighbor = torch.stack(points_list,dim=0).transpose(0,1).contiguous()

        zero_features = total_pts_features.new_zeros((1,total_pts_features.shape[-1]))
        total_pts_features = torch.cat([zero_features,total_pts_features],dim=0)
        zero_neighbor = total_pts_neighbor.new_zeros((1,total_pts_neighbor.shape[-1]))
        total_pts_neighbor = torch.cat([zero_neighbor,total_pts_neighbor],dim=0)
        total_pts_batch_index = total_pts_batch_index.float().unsqueeze(dim=-1)

        return total_pts_features, total_pts_neighbor, total_pts_batch_index, total_pts_roi_index, total_pts_features_xyz_src


    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        ) 
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']
            batch_dict['roi_scores'] = targets_dict['roi_scores']

        # RoI Grid Pool
        x_valid = self.roi_grid_pool(batch_dict)
        x_valid  = x_valid.transpose(1,2)

        # RoI Point Pool
        B, N, _ = batch_dict['rois'].shape
        points_features, points_neighbor, points_batch, points_roi, points_coords_src =\
              self.roicrop3d_gpu(batch_dict, self.model_cfg.ROI_POINT_CROP.POOL_EXTRA_WIDTH)
        points_features_expand = self.cpconvs_layer(points_features, points_neighbor)[1:]
        batch_dict['points_pseudo_features'] = torch.cat([points_batch,points_coords_src,points_features_expand],dim=-1)

        # RoI Voxel Pool
        pooled_pseudo_features = self.roiaware_pool(batch_dict)
        batch_size_rcnn = pooled_pseudo_features.shape[0]
        sparse_shape = np.array(pooled_pseudo_features.shape[1:4], dtype=np.int32)
        sparse_idx = pooled_pseudo_features.sum(dim=-1).nonzero()
        if sparse_idx.shape[0] < 3:
            sparse_idx = self.fake_sparse_idx(sparse_idx, batch_size_rcnn)
            if self.training:
                targets_dict['rcnn_cls_labels'].fill_(-1)
                targets_dict['reg_valid_mask'].fill_(-1)
        pseudo_features = pooled_pseudo_features[sparse_idx[:, 0], sparse_idx[:, 1], sparse_idx[:, 2], sparse_idx[:, 3]]
        coords = sparse_idx.int()

        pseudo_features = spconv.SparseConvTensor(pseudo_features, coords, sparse_shape, batch_size_rcnn)
        x_pseudo = self.conv_pseudo(pseudo_features)
        x_pseudo = x_pseudo.dense()
        N, C, D, H, W = x_pseudo.shape
        x_pseudo = x_pseudo.reshape(N, C , D*H*W)

        fusion_features = self.fusion_layer(x_valid, x_pseudo)
        fusion_features = fusion_features.reshape(fusion_features.size(0), -1)

        shared_features = self.shared_fc_layer(fusion_features)
        rcnn_cls = self.cls_pred_layer(self.cls_fc_layers(shared_features))
        rcnn_reg = self.reg_pred_layer(self.reg_fc_layers(shared_features))

        if self.training:
            x_pseudo = x_pseudo.reshape(x_pseudo.size(0), -1)
            shared_features_pseudo = self.shared_fc_layer_pseudo(x_pseudo)
            rcnn_cls_pseudo = self.cls_pred_layer_pseudo(self.cls_fc_layers_pseudo(shared_features_pseudo))
            rcnn_reg_pseudo = self.reg_pred_layer_pseudo(self.reg_fc_layers_pseudo(shared_features_pseudo))

            x_valid = x_valid.reshape(x_valid.size(0), -1)
            shared_features_valid = self.shared_fc_layer_valid(x_valid)
            rcnn_cls_valid = self.cls_pred_layer_valid(self.cls_fc_layers_valid(shared_features_valid))
            rcnn_reg_valid = self.reg_pred_layer_valid(self.reg_fc_layers_valid(shared_features_valid))

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg
            targets_dict['rcnn_cls_pseudo'] = rcnn_cls_pseudo
            targets_dict['rcnn_reg_pseudo'] = rcnn_reg_pseudo
            targets_dict['rcnn_cls_valid'] = rcnn_cls_valid
            targets_dict['rcnn_reg_valid'] = rcnn_reg_valid
            self.forward_ret_dict = targets_dict

        return batch_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        
        # main
        rcnn_loss = 0
        rcnn_loss_reg, reg_tb_dict, iou_target = self.get_box_reg_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_reg
        tb_dict.update(reg_tb_dict)

        rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict, iou_target)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)

        tb_dict['rcnn_loss'] = rcnn_loss.item()

        # pseudo
        rcnn_loss_pseudo = 0
        rcnn_loss_cls_pseudo, cls_tb_dict_pseudo = self.get_box_cls_layer_loss_pseudo(self.forward_ret_dict)
        rcnn_loss_pseudo += rcnn_loss_cls_pseudo
        tb_dict.update(cls_tb_dict_pseudo)

        rcnn_loss_reg_pseudo, reg_tb_dict_pseudo = self.get_box_reg_layer_loss_pseudo(self.forward_ret_dict)
        rcnn_loss_pseudo += rcnn_loss_reg_pseudo
        tb_dict.update(reg_tb_dict_pseudo)

        tb_dict['rcnn_loss_pseudo'] = rcnn_loss_pseudo.item()
        rcnn_loss += rcnn_loss_pseudo

        # valid
        rcnn_loss_valid = 0
        rcnn_loss_cls_valid, cls_tb_dict_valid = self.get_box_cls_layer_loss_valid(self.forward_ret_dict)
        rcnn_loss_valid += rcnn_loss_cls_valid
        tb_dict.update(cls_tb_dict_valid)

        rcnn_loss_reg_valid, reg_tb_dict_valid = self.get_box_reg_layer_loss_valid(self.forward_ret_dict)
        rcnn_loss_valid += rcnn_loss_reg_valid
        tb_dict.update(reg_tb_dict_valid)

        tb_dict['rcnn_loss_valid'] = rcnn_loss_valid.item()
        rcnn_loss += rcnn_loss_valid

        return rcnn_loss, tb_dict

    def get_box_cls_layer_loss(self, forward_ret_dict, iou_target):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls']
        rcnn_cls_valid  = forward_ret_dict['rcnn_cls_labels'].view(-1)
        iou_target = iou_target.view(-1)

        rcnn_cls_labels = iou_target

        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
            cls_valid_mask = (rcnn_cls_valid >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_valid >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict

    def get_box_reg_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
        rcnn_reg = forward_ret_dict['rcnn_reg']
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()

        tb_dict = {}

        if True:
            rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0

            dt_boxes = self.box_coder.decode_torch(
                rcnn_reg, rois_anchor
            )
            gt_boxes = gt_boxes3d_ct.view(rcnn_batch_size, code_size)

            if loss_cfgs.REG_LOSS == 'iou':
                iou3d = oriented_iou_loss.cal_iou_3d(dt_boxes.unsqueeze(0),  gt_boxes.unsqueeze(0))
                iou_loss = 1. - iou3d
            elif loss_cfgs.REG_LOSS == 'giou':
                iou_loss, iou3d = oriented_iou_loss.cal_giou_3d(dt_boxes.unsqueeze(0), gt_boxes.unsqueeze(0))
            elif loss_cfgs.REG_LOSS == 'diou':
                iou_loss, iou3d = oriented_iou_loss.cal_diou_3d(dt_boxes.unsqueeze(0), gt_boxes.unsqueeze(0))
            iou_loss = iou_loss.squeeze(0)
            rcnn_loss_reg = iou_loss

            iou3d = iou3d_nms_utils.boxes_iou3d_gpu(dt_boxes, gt_boxes)
            iou3d_index = range(0,len(iou3d))
            iou_target = iou3d[iou3d_index, iou3d_index].clone().detach()

            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()

            if loss_cfgs.CORNER_LOSS_REGULARIZATION and fg_sum > 0:
                # TODO: NEED to BE CHECK
                fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
                fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]

                fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
                batch_anchors = fg_roi_boxes3d.clone().detach()
                roi_ry = fg_roi_boxes3d[:, :, 6].view(-1)
                roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
                batch_anchors[:, :, 0:3] = 0
                rcnn_boxes3d = self.box_coder.decode_torch(
                    fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors
                ).view(-1, code_size)

                rcnn_boxes3d = common_utils.rotate_points_along_z(
                    rcnn_boxes3d.unsqueeze(dim=1), roi_ry
                ).squeeze(dim=1)
                rcnn_boxes3d[:, 0:3] += roi_xyz

                loss_corner = loss_utils.get_corner_loss_lidar(
                    rcnn_boxes3d[:, 0:7],
                    gt_of_rois_src[fg_mask][:, 0:7]
                )
                loss_corner = loss_corner.mean()
                loss_corner = loss_corner * loss_cfgs.LOSS_WEIGHTS['rcnn_corner_weight']

                rcnn_loss_reg += loss_corner
                tb_dict['rcnn_loss_corner'] = loss_corner.item()
        else:
            raise NotImplementedError

        return rcnn_loss_reg, tb_dict, iou_target

    def get_box_cls_layer_loss_pseudo(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls_pseudo']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS_PSEUDO['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls_pseudo': rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict

    def get_box_reg_layer_loss_pseudo(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
        rcnn_reg = forward_ret_dict['rcnn_reg_pseudo']
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()

        tb_dict = {}

        if loss_cfgs.REG_LOSS_PSEUDO == 'smooth-l1':
            rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            reg_targets = self.box_coder.encode_torch(
                gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
            )

            reg_targets_pseudo = reg_targets
            rcnn_loss_reg = self.reg_loss_func_pseudo(
                rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
                reg_targets_pseudo.unsqueeze(dim=0),
            )
            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS_PSEUDO['rcnn_reg_weight']
            tb_dict['rcnn_loss_reg_pseudo'] = rcnn_loss_reg.item()

        else:
            raise NotImplementedError

        return rcnn_loss_reg, tb_dict

    def get_box_cls_layer_loss_valid(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls_valid']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS_VALID['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls_valid': rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict

    def get_box_reg_layer_loss_valid(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
        rcnn_reg = forward_ret_dict['rcnn_reg_valid']
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()

        tb_dict = {}

        if loss_cfgs.REG_LOSS_VALID == 'smooth-l1':
            rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            reg_targets = self.box_coder.encode_torch(
                gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
            )

            reg_targets_valid = reg_targets
            rcnn_loss_reg = self.reg_loss_func_valid(
                rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
                reg_targets_valid.unsqueeze(dim=0),
            )
            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS_VALID['rcnn_reg_weight']
            tb_dict['rcnn_loss_reg_valid'] = rcnn_loss_reg.item()

        else:
            raise NotImplementedError

        return rcnn_loss_reg, tb_dict