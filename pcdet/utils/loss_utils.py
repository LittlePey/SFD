import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import box_utils, common_utils


class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedL1Loss(nn.Module):
    def __init__(self, code_weights: list = None):
        """
        Args:
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedL1Loss, self).__init__()
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = torch.abs(diff)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        input = input.permute(0, 2, 1)
        target = target.argmax(dim=-1)
        loss = F.cross_entropy(input, target, reduction='none') * weights
        return loss


def get_corner_loss_lidar(pred_bbox3d: torch.Tensor, gt_bbox3d: torch.Tensor):
    """
    Args:
        pred_bbox3d: (N, 7) float Tensor.
        gt_bbox3d: (N, 7) float Tensor.

    Returns:
        corner_loss: (N) float Tensor.
    """
    assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

    pred_box_corners = box_utils.boxes_to_corners_3d(pred_bbox3d)
    gt_box_corners = box_utils.boxes_to_corners_3d(gt_bbox3d)

    gt_bbox3d_flip = gt_bbox3d.clone()
    gt_bbox3d_flip[:, 6] += np.pi
    gt_box_corners_flip = box_utils.boxes_to_corners_3d(gt_bbox3d_flip)
    # (N, 8)
    corner_dist = torch.min(torch.norm(pred_box_corners - gt_box_corners, dim=2),
                            torch.norm(pred_box_corners - gt_box_corners_flip, dim=2))
    # (N, 8)
    corner_loss = WeightedSmoothL1Loss.smooth_l1_loss(corner_dist, beta=1.0)

    return corner_loss.mean(dim=1)


def get_gridify_iou3d_loss(gt_bbox3d: torch.Tensor, pred_bbox3d: torch.Tensor, grid_size=10):
    """
    Args:
        gt_bbox3d: (N, 7) float Tensor.
        pred_bbox3d: (N, 7) float Tensor.

    Returns:
        iou3d_loss: (N) float Tensor.
    """
    assert gt_bbox3d.shape[0] == pred_bbox3d.shape[0]
    gt_bbox3d_area = torch.prod(gt_bbox3d[:, 3:6], dim=1)
    pred_bbox3d_area = torch.prod(pred_bbox3d[:, 3:6], dim=1)
    area_per_grid = gt_bbox3d_area / grid_size**3
    
    # angle alpha
    angle_cos = torch.cos(-1 * pred_bbox3d[:, 6]).unsqueeze(-1)
    angle_sin = torch.sin(-1 * pred_bbox3d[:, 6]).unsqueeze(-1)

    # (N, 7x7x7, 3)
    grid_xyz, _ = get_global_grid_points_of_roi(gt_bbox3d, grid_size)
    dist_grid_to_center = grid_xyz - pred_bbox3d[:, :3].unsqueeze(1)

    dist_x = dist_grid_to_center[:, :, 0]
    dist_y = dist_grid_to_center[:, :, 1]
    dist_z = dist_grid_to_center[:, :, 2]

    rot_dist_x = dist_x * angle_cos + dist_y * (-1 * angle_sin)
    rot_dist_y = dist_y * angle_sin + dist_y * angle_cos
    rot_dist_z = dist_z
    
    kernel_x = inside_kernel(rot_dist_x, pred_bbox3d[:, 3].unsqueeze(1))
    kernel_y = inside_kernel(rot_dist_y, pred_bbox3d[:, 4].unsqueeze(1))
    kernel_z = inside_kernel(rot_dist_z, pred_bbox3d[:, 5].unsqueeze(1))

    inside_indicator = kernel_x * kernel_y * kernel_z

    inside_num = torch.sum(inside_indicator, dim=1)

    intersect_area = inside_num * area_per_grid

    gridify_iou3d = intersect_area / (gt_bbox3d_area + pred_bbox3d_area - intersect_area)

    loss_iou3d = 1 - gridify_iou3d
    
    return loss_iou3d


def get_global_grid_points_of_roi(rois, grid_size):
    rois = rois.view(-1, rois.shape[-1])
    batch_size_rcnn = rois.shape[0]

    local_roi_grid_points = get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
    global_roi_grid_points = common_utils.rotate_points_along_z(
        local_roi_grid_points.clone(), rois[:, 6]
    ).squeeze(dim=1)
    global_center = rois[:, 0:3].clone()
    global_roi_grid_points += global_center.unsqueeze(dim=1)
    return global_roi_grid_points, local_roi_grid_points


def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
    faked_features = rois.new_ones((grid_size, grid_size, grid_size))
    dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
    dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

    local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
    roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                        - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
    return roi_grid_points
    

def inside_kernel(dist, gt_size, scalar=10, eps=1e-8):
    dist_abs = torch.abs(dist + eps)
    half_size = gt_size.float() / 2
    inside_weight = 1 - 1./ (1 + torch.exp(-1 * scalar*(dist_abs - half_size))) 
    return inside_weight