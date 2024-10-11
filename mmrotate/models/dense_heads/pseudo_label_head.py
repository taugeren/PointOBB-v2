# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Scale
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean

from mmrotate.core import build_bbox_coder, multiclass_nms_rotated
from ..builder import ROTATED_HEADS, build_loss
from .rotated_anchor_free_head import RotatedAnchorFreeHead
from .rotated_fcos_head import RotatedFCOSHead
import numpy as np
import os
from PIL import Image

INF = 1e8


@ROTATED_HEADS.register_module()
class PseudoLabelHead(RotatedFCOSHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.
    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to suppress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: False.
        separate_angle (bool): If true, angle prediction is separated from
            bbox regression loss. Default: False.
        scale_angle (bool): If true, add scale to angle pred branch. Default: True.
        h_bbox_coder (dict): Config of horzional bbox coder, only used when separate_angle is True.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_angle (dict): Config of angle loss, only used when separate_angle is True.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
    Example:
        >>> self = RotatedFCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, angle_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iter = 0
        self.train_duration = 50
        self.test_duration = 32
        self.store_dir = None
        self.cls_weight = 20
        self.thresh3 = 0.1
        self.multiple_factor = 1/16
        
        train_cfg = kwargs.get('train_cfg', {})
        test_cfg = kwargs.get('test_cfg', {})
        
        if 'store_dir' in train_cfg:
            self.store_dir = train_cfg['store_dir']
        elif 'store_dir' in test_cfg:
            self.store_dir = test_cfg['store_dir']
        if 'thresh3' in train_cfg:
            self.thresh3 = train_cfg['thresh3']
        assert self.store_dir is not None
        os.makedirs(self.store_dir + "/visualize/", exist_ok=True)
        if 'cls_weight' in train_cfg:
            self.cls_weight = train_cfg['cls_weight']
        if 'pca_length' in train_cfg:
            self.pca_length = train_cfg['pca_length']
        if 'multiple_factor' in train_cfg:
            self.multiple_factor = train_cfg['multiple_factor']
        assert len(self.thresh3) == self.num_classes
        self.store_ann_dir = kwargs.get('train_cfg')['store_ann_dir']
        os.makedirs(self.store_ann_dir, exist_ok=True)
        
    def get_mask_image(self, max_probs, max_indices, thr, num_width):
        PALETTE = [
            (165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
            (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
            (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
            (255, 255, 0), (147, 116, 116), (0, 0, 255)
        ]
        mask_image = np.ones((num_width, num_width, 3), dtype=np.uint8) * 255
        for i in range(num_width):
            for j in range(num_width):
                if max_probs[i, j] > thr:
                    mask_image[i, j] = PALETTE[max_indices[i, j]]
        return mask_image
    
    def _draw_image(self, max_probs, max_indices, thr, img_flip_direction, img_A, num_width):
        mask_image = self.get_mask_image(max_probs, max_indices, thr, num_width)
        img_B = Image.fromarray(mask_image)
        if img_flip_direction is None:
            pass
        elif img_flip_direction == 'horizontal':
            img_B = img_B.transpose(Image.FLIP_LEFT_RIGHT)
        elif img_flip_direction == 'vertical':
            img_B = img_B.transpose(Image.FLIP_TOP_BOTTOM)
        elif img_flip_direction == 'diagonal':
            img_B = img_B.transpose(Image.FLIP_TOP_BOTTOM).transpose(Image.FLIP_LEFT_RIGHT)
        combined_image = Image.new('RGB', (img_A.width + img_B.width, img_A.height))
        combined_image.paste(img_A, (0, 0))
        combined_image.paste(img_B, (img_A.width, 0))
        return combined_image
        
    def draw_image(self, img_path, flip, score_probs):
        
        num_width = score_probs.shape[2]
        img_A = Image.open(img_path).convert("RGB")
        img_A = img_A.resize((num_width, num_width))
        
        max_probs, max_indices = torch.max(score_probs, dim=0)
        os.makedirs(self.store_dir + "/visualize/" + str(self.iter), exist_ok=True)
        
        for i in range(19):
            thr = round(0.05 * (i + 1), 2)
            conbine_image = self._draw_image(max_probs, max_indices, thr, flip, img_A, num_width)
            output_path = self.store_dir + "/visualize/" + str(self.iter) + "/" + str(thr) + ".jpg"
            conbine_image.save(output_path)
        
    def generate_labels(self, labels, bbox_targets, angle_targets, img_metas):
        classes = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
           'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
           'basketball-court', 'storage-tank', 'soccer-ball-field',
           'roundabout', 'harbor', 'swimming-pool', 'helicopter',
           'container-crane', 'airport', 'helipad')
        # angle_targets = -angle_targets
        direction = img_metas['flip_direction']
        img_name = img_metas['filename']
        # angle_targets contains the angle of the gt boxes in le90 format
        # bbox_targets contains the center, width, height of the gt boxes in le90 format
        # labels contains the class of the gt boxes
        if direction == 'horizontal':
            angle_targets = -angle_targets
            bbox_targets[:, 0] = 1024-bbox_targets[:, 0]
        elif direction == 'vertical':
            angle_targets = -angle_targets
            bbox_targets[:, 1] = 1024-bbox_targets[:, 1]
        elif direction == 'diagonal':
            bbox_targets[:, 0], bbox_targets[:, 1] = 1024-bbox_targets[:, 0], 1024-bbox_targets[:, 1]
        
        cos_vals = torch.cos(angle_targets).view(-1, 1, 1)
        sin_vals = torch.sin(angle_targets).view(-1, 1, 1)

        rotation_matrix = torch.cat([torch.cat([cos_vals, -sin_vals], dim=2), torch.cat([sin_vals, cos_vals], dim=2)], dim=1)

        x1, y1 = - bbox_targets[:, 2] / 2, - bbox_targets[:, 3] / 2
        x2, y2 = bbox_targets[:, 2] / 2, - bbox_targets[:, 3] / 2
        x3, y3 = bbox_targets[:, 2] / 2, bbox_targets[:, 3] / 2
        x4, y4 = - bbox_targets[:, 2] / 2, bbox_targets[:, 3] / 2

        vertices = torch.stack([torch.stack([x1, x2, x3, x4], dim=1), torch.stack([y1, y2, y3, y4], dim=1)], dim=1)  # [n, 2, 4]

        rotated_vertices = torch.einsum('bij,bjk->bik', rotation_matrix, vertices)  # [n, 2, 4]

        # seperate the vertices after rotation
        x1_rot, y1_rot = bbox_targets[:, 0] + rotated_vertices[:, 0, 0], bbox_targets[:, 1] + rotated_vertices[:, 1, 0]
        x2_rot, y2_rot = bbox_targets[:, 0] + rotated_vertices[:, 0, 1], bbox_targets[:, 1] + rotated_vertices[:, 1, 1]
        x3_rot, y3_rot = bbox_targets[:, 0] + rotated_vertices[:, 0, 2], bbox_targets[:, 1] + rotated_vertices[:, 1, 2]
        x4_rot, y4_rot = bbox_targets[:, 0] + rotated_vertices[:, 0, 3], bbox_targets[:, 1] + rotated_vertices[:, 1, 3]
        
        filename_raw = img_name.split('/')[-1].split('.')[0]
        
        with open(self.store_ann_dir + filename_raw + '.txt', 'w') as w:
            for i in range(len(labels)):
                w.write(f"{x1_rot[i].item():.1f} {y1_rot[i].item():.1f} {x2_rot[i].item():.1f} {y2_rot[i].item():.1f} "
                    f"{x3_rot[i].item():.1f} {y3_rot[i].item():.1f} {x4_rot[i].item():.1f} {y4_rot[i].item():.1f} "
                    f"{classes[labels[i]]} 0\n")
        # print('invoke generate_labels')
                
    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             angle_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            angle_preds (list[Tensor]): Box angle for each scale level, \
                each is a 4D-tensor, the channel number is num_points * 1.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) \
               == len(angle_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        labels = self.get_targets(
            all_level_points, gt_bboxes, gt_labels, cls_scores, img_metas)
        
        # if self.iter % self.train_duration == 0:
        #     self.draw_image(img_metas[0]['filename'], img_metas[0]['flip_direction'], cls_scores[0][0].sigmoid())
            
        self.iter += 1   
         
        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(-1, 1)
            for angle_pred in angle_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_angle_preds = torch.cat(flatten_angle_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)

        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        
        neg_inds = (flatten_labels == bg_class_ind).nonzero().reshape(-1)
        num_neg = torch.tensor(
            len(neg_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_neg = max(reduce_mean(num_neg), 1.0)
        
        avail_inds = (flatten_labels >= 0).nonzero().reshape(-1)
        num_avail = torch.tensor(
            len(avail_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_avail = max(reduce_mean(num_avail), 1.0)
        
        # print("num_pos, num_avail: ", num_pos, num_avail)
        loss_cls = self.loss_cls(
            flatten_cls_scores[avail_inds], flatten_labels[avail_inds], avg_factor=num_avail)

        if self.separate_angle:
            return dict(
                loss_cls=loss_cls,
                loss_bbox=0. * loss_cls,
                loss_angle=0. * loss_cls,
                loss_centerness=0. * loss_cls)
        else:
            return dict(
                loss_cls=self.cls_weight * loss_cls,
                loss_bbox=0. * loss_cls,
                loss_centerness=0. * loss_cls)

    def get_targets(self, points, gt_bboxes_list, gt_labels_list, cls_scores, img_metas):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).
        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
                concat_lvl_angle_targets (list[Tensor]): Angle targets of \
                    each level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        cls_scores_all = []
        batch_size = cls_scores[0].size(0)
        for i in range(batch_size):
            cls_scores_all.append([])
            for j in range(len(cls_scores)):
                cls_scores_all[i].append(cls_scores[j][i])
        
        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, angle_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            cls_scores_all,
            img_metas,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]

        # concat per level image
        concat_lvl_labels = []
        
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))

        return concat_lvl_labels
        
    def _get_target_single(self, gt_bboxes, gt_labels, cls_scores_all, img_metas, points, regress_ranges,
                           num_points_per_lvl, alpha=1, thresh1=8, thresh2_bg=4, thresh3=0.1, pca_length=28, default_max_length=128, mode='near'):
        """Compute regression, classification and angle targets for a single
        image.
        
        Args:
            cls_scores_all (list[Tensor]): feature map scores on each point for each scale level, 
                len(cls_scores_all) = num_levels, cls_scores_all[i].size() = (num_classes, w, h)
        """
        alpha = alpha
        thresh3 = self.thresh3
        pca_length = self.pca_length
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        center_point_gt = gt_bboxes[:, :2]
        
        # The following code only use the center point of the gt_bboxes, the code include "gt_bboxes" is for the vector shape
        if num_gts == 0:
            filename_raw = img_metas['filename'].split('/')[-1].split('.')[0]
            with open(self.store_ann_dir + filename_raw + '.txt', 'w') as w:
                w.write('\n')
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_bboxes.new_zeros((num_points, 1))

        dist_sample_and_gt = torch.cdist(points, center_point_gt) # [num_sample, num_gt]
        dist_gt_and_gt = torch.cdist(center_point_gt, center_point_gt) + torch.eye(num_gts).to(dist_sample_and_gt.device) * INF # [num_gt, num_gt]
        dist_min_gt_and_gt, dist_min_gt_and_gt_index = dist_gt_and_gt.min(dim=1)
        dist_min_sample_and_gt = dist_sample_and_gt.min(dim=1)[0]
        labels = -1 * torch.ones(num_points, dtype=gt_labels.dtype, device=gt_labels.device)
        
        index_neg = ((alpha * dist_sample_and_gt) > dist_min_gt_and_gt).all(dim=1).nonzero().squeeze(-1)
        if len(index_neg) > 0:
            # gt_labels = gt_labels.clone()
            labels[index_neg] = self.num_classes
            
        # positive labels
        thresh1_tensor = thresh1 * torch.ones_like(dist_min_gt_and_gt)
        dist_min_thresh1_gt = torch.min(dist_min_gt_and_gt/2, thresh1_tensor)
        index_pos = (dist_sample_and_gt < dist_min_thresh1_gt).nonzero()
        if len(index_pos) > 0:
            # gt_labels = gt_labels.clone()
            labels[index_pos[:, 0]] = gt_labels[index_pos[:, 1]]
            
        # additional background labels
        # gt_labels[dist_min_gt_and_gt_index] is the nearest gt label
        # center_point_gt[dist_min_gt_and_gt_index] is the nearest gt center point
        is_nearest_same_class = (gt_labels[dist_min_gt_and_gt_index] == gt_labels) # [num_gt]
        valid_middle_point = (center_point_gt[is_nearest_same_class] + center_point_gt[dist_min_gt_and_gt_index][is_nearest_same_class]) / 2
        dist_sample_and_gt = torch.cdist(points, valid_middle_point)
        index_neg_additional = (dist_sample_and_gt < thresh2_bg).any(dim=1).nonzero().squeeze(-1)
        if len(index_neg_additional) > 0:
            labels[index_neg_additional] = self.num_classes
        
        # extract the rectangle of each gt, and gt_ctr is the center of the gt box
        center_factor = self.get_center_factor(center_point_gt, gt_labels, cls_scores_all[0])
        gt_ctr_rect = self.get_rectangle_cls_prob(cls_scores_all[0].sigmoid(), self.strides[0], center_factor, center_point_gt, pca_length, mode='near')
        # gt_ctr_rect = self.process_rectangle_decay(gt_ctr_rect.detach().clone())
        
        # get the PCA of each gt rectangle
        gt_ctr_rect_label = gt_ctr_rect[torch.arange(num_gts), gt_labels, :, :] # [num_gts, pca_length, pca_length]
        gt_rect_ctr2edge = gt_ctr_rect_label.shape[-1]//2
        points_rect_x = torch.arange(-gt_rect_ctr2edge, gt_rect_ctr2edge + 1, 1).to(gt_ctr_rect.device)
        points_rect_y = torch.arange(-gt_rect_ctr2edge, gt_rect_ctr2edge + 1, 1).to(gt_ctr_rect.device)
        points_rect_xy = torch.stack(torch.meshgrid(points_rect_x, points_rect_y), -1).reshape(-1, 2) # [pca_length^2, 2]
        gt_ctr_rect_label = gt_ctr_rect_label.transpose(1, 2).contiguous().view(num_gts, -1) # [num_gts, pca_length^2]
        points_rect_xy_adapt = points_rect_xy.unsqueeze(0).repeat(num_gts, 1, 1) * torch.sqrt(gt_ctr_rect_label).unsqueeze(-1) # [num_gts, pca_length^2, 2] 
        points_cov_matrix = torch.matmul(points_rect_xy_adapt.transpose(1, 2), points_rect_xy_adapt) / (gt_ctr_rect_label.shape[-1] ** 2 - 1) # [num_gts, 2, 2]
        eigvals, eigvecs = torch.symeig(points_cov_matrix, eigenvectors=True)
        
        # get the largest eigval and the corresponding eigvec
        larger_eigvals_index = (eigvals[:, 1] > eigvals[:, 0]).int()
        eigval_first = eigvals[:, 0] * (1 - larger_eigvals_index) + eigvals[:, 1] * larger_eigvals_index
        eigvec_first = eigvecs[:, 0, :] * (1 - larger_eigvals_index).unsqueeze(1).repeat(1, 2) + eigvecs[:, 1, :] * larger_eigvals_index.unsqueeze(1).repeat(1, 2)
        # eigvec_first = eigvec_first + 1e-6 # [num_gts, 2]
        mask_eigvec = (eigvec_first[:, 1] > 0).int()
        epsilon = mask_eigvec * 1e-6 + (1 - mask_eigvec) * -1e-6
        
        # get the angle target (only for le90)
        angle_targets = -torch.atan(eigvec_first[:, 0] / (eigvec_first[:, 1] + epsilon)).unsqueeze(-1)
        
        # get the second axis direction
        eigvec_second = torch.stack([-eigvec_first[:, 1], eigvec_first[:, 0]], -1)
        
        first_axis_range = self.get_closest_gt_first_axis(gt_labels, eigvec_first, center_point_gt, angle_threshold=0.866)
        
        top_simple, bottom_simple = self.get_edge_boundary_simple(gt_labels, eigvec_first, center_point_gt, cls_scores_all, thresh3, default_max_length, mode='simple',
                                                                  is_secondary=False, is_nearest_same_class=is_nearest_same_class, nearest_gt_point=center_point_gt[dist_min_gt_and_gt_index], first_axis_range=first_axis_range) 
        left_simple, right_simple = self.get_edge_boundary_simple(gt_labels, eigvec_second, center_point_gt, cls_scores_all, thresh3, default_max_length, mode='simple', 
                                                                  is_secondary=True, is_nearest_same_class=is_nearest_same_class, nearest_gt_point=center_point_gt[dist_min_gt_and_gt_index]) 
        top_simple, bottom_simple = top_simple * self.strides[0] + 1, bottom_simple * self.strides[0] + 1
        left_simple, right_simple = left_simple * self.strides[0] + 1, right_simple * self.strides[0] + 1
        pseudo_gt_bboxes = torch.cat([center_point_gt, (left_simple+right_simple).unsqueeze(-1), (top_simple+bottom_simple).unsqueeze(-1), angle_targets], -1)
        pseudo_gt_bboxes = pseudo_gt_bboxes.detach()
        
        self.generate_labels(gt_labels, pseudo_gt_bboxes, angle_targets, img_metas)
        
        if num_gts == 1:
            index_pos = (dist_sample_and_gt < 8).nonzero().reshape(-1)
            index_neg = (dist_sample_and_gt > 128).nonzero().reshape(-1)
            labels[index_pos] = gt_labels[0]
            labels[index_neg] = self.num_classes
            return labels, gt_bboxes.new_zeros((num_points, 4)), gt_bboxes.new_zeros((num_points, 1))

        return labels, None, None
    
    def get_rectangle_cls_prob(self, cls_score, stride, center_factor, gt_ctr, pca_length, mode='near'):
        """ Get the classification probability of a rectangle of each gt_ctr.

            Args:
                cls_score (Tensor): classification score of each point, shape (num_classes, H, W)
                stride (int): stride of the feature map
                center_factor (Tensor): center factor of each point, shape (num_gts, H, W)
                gt_ctr (Tensor): center of the gt box, shape (num_gts, 2)
                pca_length (int): length of the PCA rectangle
        """
        
        assert mode in ['near', 'bilinear'], "mode must be either 'near' or 'bilinear'"
        
        H, W = cls_score.shape[1], cls_score.shape[2]
        gt_ctr_lvl = gt_ctr / stride
        length_lvl = pca_length / stride
        rect_length = 2*int((length_lvl-1)/2) + 1
        padding = (int((length_lvl-1)/2), int((length_lvl-1)/2), int((length_lvl-1)/2), int((length_lvl-1)/2))
        padding = (10, 10, 10, 10)
        padded_cls_score = F.pad(cls_score, padding, mode='constant', value=0)
        padded_center_factor = F.pad(center_factor, padding, mode='constant', value=0)
        gt_ctr_based_rect = torch.zeros(gt_ctr_lvl.shape[0], cls_score.shape[0], rect_length, rect_length).to(cls_score.device)
        
        gt_ctr_lvl = gt_ctr_lvl + 10
        
        if mode == 'near':
            gt_ctr_lvl = gt_ctr_lvl.round().long()
            x_max = gt_ctr_lvl[:, 0] + int((length_lvl-1)/2)
            x_min = gt_ctr_lvl[:, 0] - int((length_lvl-1)/2)
            y_max = gt_ctr_lvl[:, 1] + int((length_lvl-1)/2)
            y_min = gt_ctr_lvl[:, 1] - int((length_lvl-1)/2)
            for i in range(gt_ctr_lvl.shape[0]):
                try:
                    center_factor_i = padded_center_factor[i, int(y_min[i]):int(y_min[i] + length_lvl), int(x_min[i]):int(x_min[i] + length_lvl)]
                    # print(center_factor_i)
                    # import time
                    # time.sleep(5)
                    gt_ctr_based_rect[i] = padded_cls_score[:, int(y_min[i]):int(y_min[i] + length_lvl), int(x_min[i]):int(x_min[i] + length_lvl)] * center_factor_i.unsqueeze(0)
                except:
                    print("gt_ctr_lvl: ", gt_ctr_lvl)
                    print("gt_ctr_lvl_max: ", gt_ctr_lvl.max())
                    print("gt_ctr_lvl_min: ", gt_ctr_lvl.min())
                    print("x_min, x_max, y_min, y_max: ", x_min, x_max, y_min, y_max)
                    print("x_min_min, x_min_max, x_max_min, x_max_max: ", x_min.min(), x_min.max(), x_max.min(), x_max.max())
                    print("y_min_min, y_min_max, y_max_min, y_max_max: ", y_min.min(), y_min.max(), y_max.min(), y_max.max())
                    print("padded_cls_score: ", padded_cls_score.shape)
                    print("cls_score: ", cls_score.shape)
                    print("i: ", i)
                    
    
        elif mode == 'bilinear':
            x_max = gt_ctr_lvl[:, 0] + (length_lvl-1)/2
            x_min = gt_ctr_lvl[:, 0] - (length_lvl-1)/2
            y_max = gt_ctr_lvl[:, 1] + (length_lvl-1)/2
            y_min = gt_ctr_lvl[:, 1] - (length_lvl-1)/2
            for i in range(gt_ctr_lvl.shape[0]):
                x_max_i = int(x_max[i])
                x_min_i = int(x_min[i])
                y_max_i = int(y_max[i])
                y_min_i = int(y_min[i])
                x_max_weight = x_max[i] - x_max_i
                x_min_weight = 1 - x_max_weight
                y_max_weight = y_max[i] - y_max_i
                y_min_weight = 1 - y_max_weight
                gt_ctr_based_rect[i] = padded_cls_score[:, y_min_i, x_min_i] * x_min_weight * y_min_weight + \
                                      padded_cls_score[:, y_min_i, x_max_i] * x_max_weight * y_min_weight + \
                                      padded_cls_score[:, y_max_i, x_min_i] * x_min_weight * y_max_weight + \
                                      padded_cls_score[:, y_max_i, x_max_i] * x_max_weight * y_max_weight
        return gt_ctr_based_rect
    
    def process_rectangle_decay(self, gt_ctr_based_rect):
        """ Process the rectangle probability of each gt_ctr. The point closer to center is always larger than the point further from the center.
        
        Args:
            gt_ctr_based_rect (Tensor): rectangle probability of each gt_ctr, shape (num_gts, num_classes, pca_length, pca_length)
        """
        length = gt_ctr_based_rect.shape[-1]
        center = int((length-1)/2)
        for i in reversed(range(center)):
            gt_ctr_based_rect[:, :, i] = torch.min(gt_ctr_based_rect[:, :, i], gt_ctr_based_rect[:, :, i+1])
            gt_ctr_based_rect[:, i, :] = torch.min(gt_ctr_based_rect[:, i, :], gt_ctr_based_rect[:, i+1, :])
            gt_ctr_based_rect[:, :, length-1-i] = torch.min(gt_ctr_based_rect[:, :, length-1-i], gt_ctr_based_rect[:, :, length-2-i])
            gt_ctr_based_rect[:, length-1-i, :] = torch.min(gt_ctr_based_rect[:, length-1-i, :], gt_ctr_based_rect[:, length-2-i, :])
        
        for i in reversed(range(center)):
            for j in reversed(range(center)):
                gt_ctr_based_rect[:, i, j] = torch.min(gt_ctr_based_rect[:, i, j], torch.max(gt_ctr_based_rect[:, i+1, j], gt_ctr_based_rect[:, i, j+1]))
                gt_ctr_based_rect[:, length-1-i, j] = torch.min(gt_ctr_based_rect[:, length-1-i, j], torch.max(gt_ctr_based_rect[:, length-2-i, j], gt_ctr_based_rect[:, length-1-i, j+1]))
                gt_ctr_based_rect[:, i, length-1-j] = torch.min(gt_ctr_based_rect[:, i, length-1-j], torch.max(gt_ctr_based_rect[:, i+1, length-1-j], gt_ctr_based_rect[:, i, length-2-j]))
                gt_ctr_based_rect[:, length-1-i, length-1-j] = torch.min(gt_ctr_based_rect[:, length-1-i, length-1-j], torch.max(gt_ctr_based_rect[:, length-2-i, length-1-j], gt_ctr_based_rect[:, length-1-i, length-2-j]))
                
        return gt_ctr_based_rect
    
    def process_rectangle_decay_1(self, gt_ctr_based_rect):
        """ Process the rectangle probability of each gt_ctr. The point closer to center is always larger than the point further from the center.
        
        Args:
            gt_ctr_based_rect (Tensor): rectangle probability of each gt_ctr, shape (num_gts, num_classes, pca_length, pca_length)
        """
        length = gt_ctr_based_rect.shape[-1]
        center = int((length-1)/2)
        for i in reversed(range(center)):
            gt_ctr_based_rect[:, :, i] = torch.min(gt_ctr_based_rect[:, :, i], gt_ctr_based_rect[:, :, i+1])
            gt_ctr_based_rect[:, i, :] = torch.min(gt_ctr_based_rect[:, i, :], gt_ctr_based_rect[:, i+1, :])
            gt_ctr_based_rect[:, :, length-1-i] = torch.min(gt_ctr_based_rect[:, :, length-1-i], gt_ctr_based_rect[:, :, length-2-i])
            gt_ctr_based_rect[:, length-1-i, :] = torch.min(gt_ctr_based_rect[:, length-1-i, :], gt_ctr_based_rect[:, length-2-i, :])
        
        for i in reversed(range(center)):
            for j in reversed(range(center)):
                if i == j:
                    gt_ctr_based_rect[:, i, j] = torch.min(gt_ctr_based_rect[:, i, j], gt_ctr_based_rect[:, i+1, j+1])
                    gt_ctr_based_rect[:, length-1-i, j] = torch.min(gt_ctr_based_rect[:, length-1-i, j], gt_ctr_based_rect[:, length-2-i, j+1])
                    gt_ctr_based_rect[:, i, length-1-j] = torch.min(gt_ctr_based_rect[:, i, length-1-j], gt_ctr_based_rect[:, i+1, length-2-j])
                    gt_ctr_based_rect[:, length-1-i, length-1-j] = torch.min(gt_ctr_based_rect[:, length-1-i, length-1-j], gt_ctr_based_rect[:, length-2-i, length-2-j])
                elif i < j:
                    ratio = (center - j) / (center - i)
                    gt_ctr_based_rect[:, i, j] = torch.min(gt_ctr_based_rect[:, i, j], (1 - ratio) * gt_ctr_based_rect[:, i+1, j] + ratio * gt_ctr_based_rect[:, i+1, j+1])
                    gt_ctr_based_rect[:, length-1-i, j] = torch.min(gt_ctr_based_rect[:, length-1-i, j], (1 - ratio) * gt_ctr_based_rect[:, length-2-i, j] + ratio * gt_ctr_based_rect[:, length-2-i, j+1])
                    gt_ctr_based_rect[:, i, length-1-j] = torch.min(gt_ctr_based_rect[:, i, length-1-j], (1 - ratio) * gt_ctr_based_rect[:, i+1, length-1-j] + ratio * gt_ctr_based_rect[:, i+1, length-2-j])
                    gt_ctr_based_rect[:, length-1-i, length-1-j] = torch.min(gt_ctr_based_rect[:, length-1-i, length-1-j], (1 - ratio) * gt_ctr_based_rect[:, length-2-i, length-1-j] + ratio * gt_ctr_based_rect[:, length-2-i, length-2-j])
                elif i > j:
                    ratio = (center - i) / (center - j)
                    gt_ctr_based_rect[:, i, j] = torch.min(gt_ctr_based_rect[:, i, j], (1 - ratio) * gt_ctr_based_rect[:, i, j+1] + ratio * gt_ctr_based_rect[:, i+1, j+1])
                    gt_ctr_based_rect[:, length-1-i, j] = torch.min(gt_ctr_based_rect[:, length-1-i, j], (1 - ratio) * gt_ctr_based_rect[:, length-1-i, j+1] + ratio * gt_ctr_based_rect[:, length-2-i, j+1])
                    gt_ctr_based_rect[:, i, length-1-j] = torch.min(gt_ctr_based_rect[:, i, length-1-j], (1 - ratio) * gt_ctr_based_rect[:, i, length-2-j] + ratio * gt_ctr_based_rect[:, i+1, length-2-j])
                    gt_ctr_based_rect[:, length-1-i, length-1-j] = torch.min(gt_ctr_based_rect[:, length-1-i, length-1-j], (1 - ratio) * gt_ctr_based_rect[:, length-1-i, length-2-j] + ratio * gt_ctr_based_rect[:, length-2-i, length-2-j])
                                
        return gt_ctr_based_rect
    
    def get_center_factor(self, center_point_gt, gt_labels, cls_scores_first_level):
        """ Get the center factor of each gt box.
        
        Args:
            center_point_gt (Tensor): center of each gt box, shape (num_gts, 2)
            gt_labels (Tensor): labels of each gt box, shape (num_gts,)
            cls_scores_first_level (Tensor): classification score of each point, shape (num_classes, H, W)
            
        Returns:
            Tensor: center factor of each gt box, shape (num_gts, H, W)
        """
        num_gts = center_point_gt.shape[0]
        _, H, W = cls_scores_first_level.shape
        
        unique_labels = gt_labels.unique()

        center_factors = []

        for label in unique_labels:
            # get the gt index
            mask = (gt_labels == label)
            gt_ctrs = center_point_gt[mask]

            m_i = gt_ctrs.shape[0]  
            center_factor_i = self.get_center_factor_cls(gt_ctrs, H, W)

            center_factors.append(center_factor_i)

        # concat all center factors according to the gt_labels
        final_center_factors = torch.zeros((num_gts, H, W), dtype=center_factors[0].dtype, device=center_factors[0].device)
        for label, center_factor_i in zip(unique_labels, center_factors):
            if center_factor_i is None:
                continue
            mask = (gt_labels == label)
            final_center_factors[mask] = center_factor_i

        return final_center_factors
    
    def get_center_factor_cls(self, gt_ctrs, H, W):
        """ Get the center factor of each gt box for a specific class.
        
        Args:
            gt_ctrs (Tensor): center of each gt box of a specific class, shape (num_gts_cls, 2)
            H (int): height of the feature map
            W (int): width of the feature map
            
        Returns:
            Tensor: center factor of each gt box of a specific class, shape (num_gts_cls, H, W)
        """
        num_gts_cls = gt_ctrs.shape[0]
        if num_gts_cls == 0:
            return None
        elif num_gts_cls == 1:
            center_factor_cls = torch.ones((1, H, W), dtype=torch.float32, device=gt_ctrs.device)
            return center_factor_cls
        else:
            # center_factor_cls = torch.zeros((num_gts_cls, H, W), dtype=torch.float32, device=gt_ctrs.device)
            points_rect_x = torch.arange(0, W * self.strides[0], self.strides[0]).to(gt_ctrs.device).float()
            points_rect_y = torch.arange(0, H * self.strides[0], self.strides[0]).to(gt_ctrs.device).float()
            points_rect_xy = torch.stack(torch.meshgrid(points_rect_x, points_rect_y), -1).reshape(-1, 2) # [H*W, 2]
            each_gt_factor = torch.cdist(points_rect_xy, gt_ctrs) # [H*W, num_gts_cls]
            # print("each_gt_factor_max: ", each_gt_factor.max())
            # print("each_gt_factor_min: ", each_gt_factor.min())
            # import time
            # time.sleep(20)
            each_gt_factor = each_gt_factor.transpose(0, 1).reshape(num_gts_cls, H, W)
            each_gt_factor = each_gt_factor.transpose(1, 2)
            each_gt_factor_exp = torch.exp(-each_gt_factor * self.multiple_factor) + 1e-6
            sum = each_gt_factor_exp.sum(dim=0).unsqueeze(0)
            center_factor_cls = each_gt_factor_exp / sum
            return center_factor_cls
        
    def get_closest_gt_first_axis(self, gt_labels, eigvec_first, center_point_gt, angle_threshold):
        """ Get the closest gt box to the first eigvec of each gt box.
        
        Args:
            gt_labels (Tensor): labels of each gt box, shape (num_gts,)
            eigvec_first (Tensor): the first eigenvector of the PCA of each gt box, shape (num_gts, 2)
            center_point_gt (Tensor): center of each gt box, shape (num_gts, 2)
            angle_threshold (float): the threshold of the angle between the first eigvec and the vector from the center to the nearest point
        
        Returns:
            Tensor: the range of the closest gt box to the first eigvec of each gt box, shape (num_gts, )
        """
        num_gts = center_point_gt.shape[0]
        
        unique_labels = gt_labels.unique()

        first_axis_range = []

        for label in unique_labels:
            # get the gt index
            mask = (gt_labels == label)
            gt_ctrs = center_point_gt[mask]
            eigvec_first_cls = eigvec_first[mask]

            m_i = gt_ctrs.shape[0]  
            first_axis_range_i = self.get_closest_gt_first_axis_cls(gt_ctrs, eigvec_first_cls, angle_threshold)

            first_axis_range.append(first_axis_range_i)

        # concat all center factors according to the gt_labels
        final_first_axis_range = torch.zeros((num_gts, ), dtype=first_axis_range[0].dtype, device=first_axis_range[0].device)
        for label, first_axis_range_i in zip(unique_labels, first_axis_range):
            if first_axis_range_i is None:
                continue
            mask = (gt_labels == label)
            final_first_axis_range[mask] = first_axis_range_i

        return torch.abs(final_first_axis_range)
    
    def get_closest_gt_first_axis_cls(self, gt_ctrs, eigvec_first, angle_threshold=0.866):
        """ Get the closest gt box to the first eigvec of each gt box for a specific class.
        
        Args:
            gt_ctrs (Tensor): center of each gt box of a specific class, shape (num_gts_cls, 2)
            eigvec_first (Tensor): the first eigenvector of the PCA of each gt box, shape (num_gts_cls, 2)
            angle_threshold (float): the threshold of the angle between the first eigvec and the vector from the center to the nearest point
            
        Returns:
            first_eigvec_range: Tensor: the range of the closest gt box to the first eigvec of each gt box for a specific class, shape (num_gts_cls, )
        """
        num_gts_cls = gt_ctrs.shape[0]
        if num_gts_cls == 0:
            return None
        elif num_gts_cls == 1:
            return 512 * torch.ones((1, ), dtype=torch.float32, device=gt_ctrs.device)
        else:
            # find valid gt boxes according to the angle
            first_eigvec_range = torch.zeros((num_gts_cls, ), dtype=torch.float32, device=gt_ctrs.device)
            eigvec_first_norm = eigvec_first / torch.norm(eigvec_first, dim=1, keepdim=True) #[num_gts_cls, 2]
            gt_and_gt_vector = gt_ctrs - gt_ctrs.unsqueeze(1) # [num_gts_cls, num_gts_cls, 2]
            gt_vec_proj = torch.abs((gt_and_gt_vector * eigvec_first_norm.unsqueeze(1)).sum(dim=-1)) # [num_gts_cls, num_gts_cls]
            gt_and_gt_norm_cos_angle = gt_vec_proj / torch.norm(gt_and_gt_vector, dim=-1) # [num_gts_cls, num_gts_cls]
            mask_valid_angle = gt_and_gt_norm_cos_angle > angle_threshold
            for i in range(num_gts_cls):
                mask_valid_angle_i = mask_valid_angle[i]
                if mask_valid_angle_i.sum() == 0:
                    first_eigvec_range[i] = 512
                    continue
                gt_proj = gt_vec_proj[i, mask_valid_angle_i]
                gt_proj_min = torch.min(gt_proj)
                first_eigvec_range[i] = gt_proj_min
            return first_eigvec_range
    
    def get_edge_boundary(self, gt_labels, eigvec_first, center_point_gt, cls_scores_all, thresh3=0.1, default_max_length=128, mode='near'):
        """ Get the edge boundary of the long edge of each gt box.
        
        Args:
            gt_labels (Tensor): labels of each gt box, shape (num_gts,)
            eigvec_first (Tensor): the first eigenvector of the PCA of each gt box, shape (num_gts, 2)
            center_point_gt (Tensor): center of each gt box, shape (num_gts, 2)
            cls_scores_all (list[Tensor]): feature map scores on each point for each scale level, 
                len(cls_scores_all) = num_levels, cls_scores_all[i].size() = (num_classes, H, W)
        """
        # TODO there is still some bugs in this function, please use get_edge_boundary_simple instead
        
        num_gts = center_point_gt.shape[0]
        center_point_gt = center_point_gt / self.strides[0]
        
        # get the rectangle boundary of long edge
        eigvec_first_norm = eigvec_first / torch.norm(eigvec_first, dim=1, keepdim=True) #[num_gts, 2]
        first_axis_sample = torch.stack([torch.arange(default_max_length) + 1, -1 * torch.arange(default_max_length) - 1]) # [2, default_max_length]
        first_axis_sample = first_axis_sample.view(1, 2, -1, 1).repeat(num_gts, 1, 1, 1).to(eigvec_first.device) # [num_points, 2, default_max_length, 1]
        first_axis_sample = first_axis_sample * eigvec_first_norm.view(num_gts, 1, 1, 2) # [num_gts, 2, default_max_length, 2]
        gt_first_axis_sample = first_axis_sample + center_point_gt.view(num_gts, 1, 1, 2) # [num_gts, 2, default_max_length, 2]
        cls_scores_sigmoid = cls_scores_all[0].sigmoid() # [num_classes, w, h]
        padding = (default_max_length, default_max_length, default_max_length, default_max_length)
        if mode == 'near':
            gt_first_axis_sample = gt_first_axis_sample.round().long()
            x_coords = gt_first_axis_sample[..., 0]
            y_coords = gt_first_axis_sample[..., 1]
            
            # mask the points that are out of the image
            mask = (x_coords >= 0) & (x_coords < cls_scores_sigmoid.shape[2]) & (y_coords >= 0) & (y_coords < cls_scores_sigmoid.shape[1])
            x_coords = torch.clamp(x_coords, 0, cls_scores_sigmoid.shape[2] - 1)
            y_coords = torch.clamp(y_coords, 0, cls_scores_sigmoid.shape[1] - 1)
            cls_scores_first_axis_points = cls_scores_sigmoid[:, y_coords, x_coords]
            cls_scores_first_axis_points = cls_scores_first_axis_points.permute(1, 2, 3, 0)
            cls_scores_first_axis_points[~mask] = 0
            gt_labels_indices = gt_labels.view(num_gts, 1, 1, 1).expand(num_gts, 2, default_max_length, 1) # [num_gts, 2, default_max_length, 1]
            selected_cls_scores_first_axis_points = torch.gather(cls_scores_first_axis_points, 3, gt_labels_indices)
            selected_cls_scores_first_axis_points = selected_cls_scores_first_axis_points.squeeze(-1) # [num_gts, 2, default_max_length]
            
            # find the first place that the score is less than thresh3
            mask_thresh3 = selected_cls_scores_first_axis_points < thresh3
            first_indices = mask_thresh3.int().argmax(dim=-1)
            all_above_thresh3 = mask_thresh3.all(dim=-1)
            first_indices[all_above_thresh3] = default_max_length # [num_gts, 2], long edge boundary
            top = (first_indices[:, 0] + first_indices[:, 1])/2
            bottom = (first_indices[:, 0] + first_indices[:, 1])/2
            return top, bottom
        elif mode == 'bilinear':
            # ToDo
            pass
        
    def get_edge_boundary_simple(self, gt_labels, eigvec, center_point_gt, cls_scores_all, thresh3=0.1, default_max_length=128, mode='near', 
                                 is_secondary=False, is_nearest_same_class=None, nearest_gt_point=None, first_axis_range=None):
        """ Get the edge boundary of the long edge of each gt box.
        
        Args:
            gt_labels (Tensor): labels of each gt box, shape (num_gts,)
            eigvec (Tensor): the first eigenvector of the PCA of each gt box, shape (num_gts, 2)
            center_point_gt (Tensor): center of each gt box, shape (num_gts, 2)
            cls_scores_all (list[Tensor]): feature map scores on each point for each scale level, 
                len(cls_scores_all) = num_levels, cls_scores_all[i].size() = (num_classes, H, W)
            is_secondary (bool): whether the eigvec is the secondary eigvec
            is_nearest_same_class (Tensor): whether the nearest gt box is the same class as the current gt box, shape (num_gts,)
            nearest_gt_point (Tensor): the nearest gt point of the current gt box, shape (num_gts, 2)
        """
        num_gts = center_point_gt.shape[0]
        center_point_gt = center_point_gt / self.strides[0]
        
        # get the rectangle boundary of long edge
        eigvec_norm = eigvec / torch.norm(eigvec, dim=1, keepdim=True)
        top_bottom = torch.zeros(num_gts, 2).to(center_point_gt.device)
        cls_scores_sigmoid = cls_scores_all[0].sigmoid()
        if first_axis_range is not None:
            first_axis_range = first_axis_range / self.strides[0]
        for i in range(num_gts):
            ctr = center_point_gt[i]
            eigvec_i = eigvec_norm[i]
            # if is_secondary:
            is_same_class = is_nearest_same_class[i]
            nearest_gt_point_i = nearest_gt_point[i] / self.strides[0]
            direction = nearest_gt_point_i - ctr
            direction_norm = direction / torch.norm(direction)
            distance = torch.abs((direction * eigvec_i).sum())
            if not is_secondary:
                valid_secondary_duplicate_remove = torch.abs((direction_norm * eigvec_i).sum()) > 0.866
            else:
                valid_secondary_duplicate_remove = torch.abs((direction_norm * eigvec_i).sum()) > 0.5
            
            for j in range(default_max_length):
                point = (ctr + j * eigvec_i).round().long()
                if point[0] < 0 or point[0] >= cls_scores_all[0].shape[2] or point[1] < 0 or point[1] >= cls_scores_all[0].shape[1]:
                    top_bottom[i, 0] = j
                    break
                if cls_scores_sigmoid[gt_labels[i], point[1], point[0]] < self.thresh3[gt_labels[i]]:
                    top_bottom[i, 0] = j
                    break
                if valid_secondary_duplicate_remove:
                    if is_same_class and j > 0.5 * distance:
                        top_bottom[i, 0] = j
                        break
                if first_axis_range is not None:
                    if j > 0.6 * first_axis_range[i]:
                        top_bottom[i, 0] = j
                        break
            for j in range(default_max_length):
                point = (ctr - j * eigvec_i).round().long()
                if point[0] < 0 or point[0] >= cls_scores_all[0].shape[2] or point[1] < 0 or point[1] >= cls_scores_all[0].shape[1]:
                    top_bottom[i, 1] = j
                    break
                if cls_scores_sigmoid[gt_labels[i], point[1], point[0]] < self.thresh3[gt_labels[i]]:
                    top_bottom[i, 1] = j
                    break
                if is_secondary and valid_secondary_duplicate_remove:
                    if is_same_class and j > 0.5 * distance:
                        top_bottom[i, 0] = j - 1
                        break
                if first_axis_range is not None:
                    if j > 0.6 * first_axis_range[i]:
                        top_bottom[i, 0] = j
                        break
        return top_bottom[:, 0], top_bottom[:, 1]
                
        
    @force_fp32(
        apply_to=('cls_scores', 'bbox_preds', 'angle_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   angle_preds,
                   centernesses,
                   img_metas,
                   cfg=None,
                   rescale=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W)
            angle_preds (list[Tensor]): Box angle for each scale level \
                with shape (N, num_points * 1, H, W)
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 6) tensor, where the first 5 columns
                are bounding box positions (x, y, w, h, angle) and the 6-th
                column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        # if self.iter % self.test_duration == 0:
        #     self.draw_image(img_metas[0]['filename'], img_metas[0]['flip_direction'], cls_scores[0][0].sigmoid(), thr1=0.03, thr2=0.05, thr3=0.07, thr4=0.10)
            
        # self.iter += 1   
        
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        mlvl_points = self.prior_generator.grid_priors(featmap_sizes,
                                                       bbox_preds[0].dtype,
                                                       bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            angle_pred_list = [
                angle_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(cls_score_list,
                                                 bbox_pred_list,
                                                 angle_pred_list,
                                                 centerness_pred_list,
                                                 mlvl_points, img_shape,
                                                 scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list
    
    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           angle_preds,
                           centernesses,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * 4, H, W).
            angle_preds (list[Tensor]): Box angle for a single scale level \
                with shape (N, num_points * 1, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points * 1, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 6), where the first 5 columns
                are bounding box positions (x, y, w, h, angle) and the
                6-th column is a score between 0 and 1.
        """
        
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, angle_pred, centerness, points in zip(
                cls_scores, bbox_preds, angle_preds, centernesses,
                mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(-1, 1)
            bbox_pred = torch.cat([bbox_pred, angle_pred], dim=1)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bboxes = self.bbox_coder.decode(
                points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            scale_factor = mlvl_bboxes.new_tensor(scale_factor)
            mlvl_bboxes[..., :4] = mlvl_bboxes[..., :4] / scale_factor
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)
        det_bboxes, det_labels = multiclass_nms_rotated(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness)
        return det_bboxes, det_labels
