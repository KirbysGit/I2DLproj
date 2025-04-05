# model / detection_head.py

# -----

# Generates Anchors for Diff Scales & Aspect Ratios.
# Predicts Class Scores & Bounding Box Offsets.
# Assigns Targets to Anchors Based on IoU Quality.
# Computes Losses for Classification & Bounding Box Regression.

# -----

# Imports.
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from v1.utils.box_ops import box_iou
from v1.model.anchor_generator import AnchorGenerator
import math
import matplotlib.pyplot as plt
import os
# Box Coder Class.
class BoxCoder:
    """Box Coordinate Conversion Utilities."""
    
    # Encode Boxes.
    def encode(self, src_boxes, dst_boxes):
        """Convert Box Coordinates to Deltas."""

        # Get Source Boxes.
        widths = src_boxes[:, 2] - src_boxes[:, 0]
        heights = src_boxes[:, 3] - src_boxes[:, 1]
        ctr_x = src_boxes[:, 0] + 0.5 * widths
        ctr_y = src_boxes[:, 1] + 0.5 * heights

        # Get Destination Boxes.
        dst_widths = dst_boxes[:, 2] - dst_boxes[:, 0]
        dst_heights = dst_boxes[:, 3] - dst_boxes[:, 1]
        dst_ctr_x = dst_boxes[:, 0] + 0.5 * dst_widths
        dst_ctr_y = dst_boxes[:, 1] + 0.5 * dst_heights

        # Compute Deltas.
        dx = (dst_ctr_x - ctr_x) / widths
        dy = (dst_ctr_y - ctr_y) / heights
        dw = torch.log(dst_widths / widths)
        dh = torch.log(dst_heights / heights)

        return torch.stack([dx, dy, dw, dh], dim=1)

# Detection Head Class.
class DetectionHead(nn.Module):
    """
    Detection Head for Object Detection.
    Predicts Object Presence and Bounding Box Refinements for Each Anchor.
    """
    
    # Initialize Detection Head.
    def __init__(self, 
                 in_channels=256,          # FPN Channels.
                 num_anchors=1,            # Single anchor per location to match checkpoint
                 num_classes=3,            # Multi-class Classification (3 classes).
                 num_convs=4,              # Number of Shared Convolutions.
                 debug=False,
                 min_size=0.02,  # Minimum allowed prediction size (2% of image)
                 max_size=0.5,   # Maximum allowed prediction size (50% of image)
                 max_delta=2.0): # Maximum allowed box delta
        super().__init__()
        
        # Debug settings
        self.debug = debug
        self.min_size = min_size
        self.max_size = max_size
        self.max_delta = max_delta
        
        # Create Anchor Generator with improved configuration for dense objects
        self.anchor_generator = AnchorGenerator(
            min_size=min_size,
            max_size=max_size
        )
        
        # Cache for anchor matching
        self._cached_anchors = None
        self._cached_image_size = None
        
        # Update num_anchors based on Generator
        self.num_anchors = len(self.anchor_generator.aspect_ratios) * len(self.anchor_generator.scales)
        self.num_classes = num_classes
        
        # Shared Convolutions with BatchNorm and ReLU
        self.shared_convs = nn.ModuleList()
        curr_channels = in_channels
        
        for _ in range(num_convs):
            conv_block = nn.Sequential(
                nn.Conv2d(curr_channels, curr_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(curr_channels),
                nn.ReLU(inplace=True)
            )
            self.shared_convs.append(conv_block)
        
        # Classification Branch
        self.cls_head = nn.Conv2d(
            curr_channels, 
            self.num_anchors * num_classes,
            3, 
            padding=1
        )
        
        # Box Regression Branch
        self.box_head = nn.Conv2d(
            curr_channels,
            self.num_anchors * 4,  # Remove multiplication by num_classes
            3,
            padding=1
        )
        
        # Loss Functions with improved weighting
        self.alpha = 0.25  # Focal loss alpha
        self.gamma = 2.0   # Focal loss gamma
        self.box_beta = 0.05  # L1-smooth loss beta
        self.cls_weight = 1.0  # Classification loss weight
        self.box_weight = 1.0  # Box regression loss weight
        
        # Initialize Weights
        self._initialize_weights()
        
        # Initialize Box Coder
        self.box_coder = BoxCoder()
        
    # ----------------------------------------------------------------------------

    # Initialize Weights.
    def _initialize_weights(self):
        """Initialize weights with Xavier/Kaiming initialization"""

        # Iterate Over All Modules.
        for m in self.modules():
            # If Module is a Convolution.
            if isinstance(m, nn.Conv2d):
                # Initialize Weights.
                nn.init.kaiming_uniform_(m.weight, mode='fan_out')

                # If Bias is Not None.
                if m.bias is not None:
                    # Initialize Bias to Zero.
                    nn.init.zeros_(m.bias)
    
    # ----------------------------------------------------------------------------

    # Forward Pass for a Single Feature Level.
    def forward_single(self, x):
        """Forward pass for a single feature level with improved feature extraction."""
        # Apply shared convolutions with BatchNorm and ReLU
        feat = x
        for conv_block in self.shared_convs:
            feat = conv_block(feat)  # Already includes BatchNorm and ReLU
            if self.debug:
                print(f"Feature shape after conv block: {feat.shape}")
        
        # Classification head
        cls_scores = self.cls_head(feat)

        if self.debug:
            probs = torch.sigmoid(cls_scores)
            high_conf = (probs > 0.9).sum().item()
            print(f"[🔍] High confidence predictions (>0.9): {high_conf} / {probs.numel()}")

        
        # Box regression head
        bbox_preds = self.box_head(feat)
        
        if self.debug:
            print(f"Classification scores shape: {cls_scores.shape}")
            print(f"Box predictions shape: {bbox_preds.shape}")
            
            # Add feature statistics for debugging
            print(f"Feature statistics:")
            print(f"  Mean: {feat.mean():.4f}")
            print(f"  Std: {feat.std():.4f}")
            print(f"  Max: {feat.max():.4f}")
            print(f"  Min: {feat.min():.4f}")
        
        return cls_scores, bbox_preds
    
    # ----------------------------------------------------------------------------

    # Match Anchors to Targets.
    def match_anchors_to_targets(self, anchors, target_boxes, target_labels, 
                               iou_threshold=0.2,  # Lower threshold for dense objects
                               max_matches_per_target=50):  # More matches for dense scenes
        """Improved anchor matching for dense object detection with dynamic thresholding."""
        device = target_boxes.device
        
        # Convert list of anchors to single tensor if needed
        if isinstance(anchors, list):
            anchors = torch.cat(anchors, dim=0)
        
        # Initialize matched labels and boxes
        num_anchors = len(anchors)
        matched_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        matched_boxes = torch.zeros_like(anchors)
        
        if len(target_boxes) == 0:
            return matched_labels, matched_boxes
        
        # Compute IoU matrix efficiently
        ious = box_iou(anchors, target_boxes)  # [num_anchors, num_targets]
        
        # Dynamic IoU thresholding based on training progress
        if hasattr(self, 'training_progress'):
            # Start with very lenient threshold and gradually increase
            min_iou = 0.1
            max_iou = 0.3
            current_iou = min_iou + (max_iou - min_iou) * self.training_progress
            iou_threshold = min(current_iou, iou_threshold)
        
        # Get matches above threshold
        matches_above_thresh = (ious >= iou_threshold).sum().item()
        
        # If too few matches, gradually lower threshold
        if matches_above_thresh < len(target_boxes) * 2:  # Aim for at least 2 matches per target
            while iou_threshold > 0.1 and matches_above_thresh < len(target_boxes) * 2:
                iou_threshold *= 0.8  # Reduce threshold by 20%
                matches_above_thresh = (ious >= iou_threshold).sum().item()
        
        # For each target, find top-k matching anchors
        max_ious, anchor_indices = ious.topk(k=min(max_matches_per_target, len(anchors)), dim=0)
        
        # Filter matches by IoU threshold
        valid_matches = max_ious >= iou_threshold
        
        # Get corresponding target indices
        target_indices = torch.arange(len(target_boxes), device=device).unsqueeze(0).expand_as(anchor_indices)
        
        # Filter and flatten indices
        valid_anchor_idx = anchor_indices[valid_matches]
        valid_target_idx = target_indices[valid_matches]
        
        # Ensure each target gets at least one anchor
        best_anchor_per_target, _ = ious.max(dim=0)  # [num_targets]
        force_match_mask = best_anchor_per_target < iou_threshold
        if force_match_mask.any():
            force_match_anchors = ious.argmax(dim=0)[force_match_mask]
            force_match_targets = torch.arange(len(target_boxes), device=device)[force_match_mask]
            valid_anchor_idx = torch.cat([valid_anchor_idx, force_match_anchors])
            valid_target_idx = torch.cat([valid_target_idx, force_match_targets])
        
        if len(valid_anchor_idx) > 0:
            # Assign labels and boxes
            matched_labels[valid_anchor_idx] = target_labels[valid_target_idx]
            matched_boxes[valid_anchor_idx] = target_boxes[valid_target_idx]
            
            if self.debug:
                print(f"\nMatching Statistics:")
                print(f"Total matches: {len(valid_anchor_idx)}")
                print(f"Matches per target: {len(valid_anchor_idx)/len(target_boxes):.1f}")
                print(f"Final IoU threshold: {iou_threshold:.3f}")
                print(f"Force-matched targets: {force_match_mask.sum().item()}")
        
        return matched_labels, matched_boxes

    # ----------------------------------------------------------------------------

    # Forward Pass.
    def forward(self, features, image_size):
        """Forward Pass Through Detection Head."""

        # Get Feature List.
        feature_list = [features[f'p{i}'] for i in range(2, 6)]  # P2 to P5
        batch_size = feature_list[0].shape[0]
        device = feature_list[0].device
        
        # Generate Anchors First to Know How Many We Need Per Location.
        anchors = []
        anchor_nums = []  # Store number of anchors per level
        
        for level_id, feature in enumerate(feature_list):
            H, W = feature.shape[2:]
            level_anchors = self.anchor_generator.generate_anchors_for_level(
                feature_map_size=(H, W),
                stride=self.anchor_generator.base_sizes[level_id],
                device=device
            )
            anchors.append(level_anchors)
            # Calculate Anchors Per Grid Cell.
            anchors_per_cell = len(level_anchors) // (H * W)
            anchor_nums.append(anchors_per_cell)
        
        # Store for Loss Computation.
        self.last_anchors = anchors
        self.total_anchors = sum(len(a) for a in anchors)
        
        # Get Predictions Using Correct Number of Anchors Per Level.
        cls_scores = []
        bbox_preds = []
        
        for level_id, feature in enumerate(feature_list):
            # Shared Features.
            feat = feature
            for conv in self.shared_convs:
                feat = F.relu(conv(feat))
            
            # Get Raw Predictions
            cls_score = self.cls_head(feat)  # [B, num_anchors*num_classes, H, W]
            bbox_pred = self.box_head(feat)  # [B, num_anchors*4, H, W]
            
            # Ensure consistent dtype
            cls_score = cls_score.to(dtype=torch.float32)
            bbox_pred = bbox_pred.to(dtype=torch.float32)
            
            # Reshape Predictions
            B, _, H, W = cls_score.shape
            cls_score = cls_score.view(B, self.num_anchors, self.num_classes, H, W)
            bbox_pred = bbox_pred.view(B, self.num_anchors, 4, H, W)
            
            # Apply temperature scaling and bias correction for better calibration
            temperature = 2.0  # Higher temperature = softer probabilities
            cls_score = cls_score / temperature
            
            # Add small bias to prevent extreme probabilities
            eps = 1e-6
            cls_probs = torch.sigmoid(cls_score)
            cls_probs = cls_probs * (1 - 2 * eps) + eps  # Squash to [eps, 1-eps]
            
            # Quality-aware confidence adjustment
            if self.training:
                # During training, use raw probabilities
                final_scores = cls_probs
            else:
                # During inference, adjust confidence based on prediction quality
                box_quality = self._compute_box_quality(bbox_pred)  # [B, num_anchors, H, W]
                box_quality = box_quality.unsqueeze(2)  # Add class dimension
                final_scores = cls_probs * box_quality  # Reduce confidence for low-quality boxes
            
            # Take the box predictions for the highest scoring class
            bbox_pred = bbox_pred.permute(0, 1, 3, 4, 2)  # [B, num_anchors, H, W, 4]
            
            # Get scores and corresponding box predictions
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
        
        # Debug Info.
        if getattr(self, 'verbose', False):
            # Print Anchor Counts.
            print("\nAnchor counts per level:")
            for i, a in enumerate(anchors):
                print(f"P{i+2}: {len(a)} anchors ({anchor_nums[i]} per cell)")
            print(f"Total anchors: {self.total_anchors}")
            
            print("\nPrediction shapes:")
            for i, (cls, box) in enumerate(zip(cls_scores, bbox_preds)):
                print(f"P{i+2}: cls {cls.shape}, box {box.shape}")
                pred_anchors = cls.shape[1] * cls.shape[3] * cls.shape[4]
                print(f"Predicted anchors: {pred_anchors}")
        
        # Return Predictions.
        return {
            'cls_scores': cls_scores,
            'bbox_preds': bbox_preds,
            'anchors': anchors
        }

    def _compute_box_quality(self, bbox_pred):
        """Compute box prediction quality score."""
        # Extract box deltas
        dx = bbox_pred[..., 0, :, :]
        dy = bbox_pred[..., 1, :, :]
        dw = bbox_pred[..., 2, :, :]
        dh = bbox_pred[..., 3, :, :]
        
        # Penalize large shifts and extreme size changes
        center_quality = torch.exp(-(dx.pow(2) + dy.pow(2)) / 0.5)  # Penalize large center shifts
        size_quality = torch.exp(-(dw.pow(2) + dh.pow(2)) / 0.5)   # Penalize extreme size changes
        
        # Combine qualities
        box_quality = center_quality * size_quality
        
        return box_quality

    # ----------------------------------------------------------------------------

    # Classification Loss.
    def cls_loss(self, pred_scores, target_boxes, target_labels):
        """Updated classification loss with proper anchor matching."""
        batch_size = pred_scores[0].shape[0]  # Get batch size from first level
        device = pred_scores[0].device
        
        if self.debug:
            print("\nClassification Loss Computation:")
            for i, level_scores in enumerate(pred_scores):
                print(f"Level {i} scores shape: {level_scores.shape}")
            print(f"Target boxes shape: {target_boxes.shape}")
            print(f"Target labels shape: {target_labels.shape}")
        
        # Process & Verify Predictions.
        all_scores = []
        total_preds = 0
        
        # Iterate Over Each Level.
        for level_scores in pred_scores:
            # level_scores shape: [B, num_anchors, num_classes, H, W]
            B, num_anchors, num_classes, H, W = level_scores.shape
            # Reshape to [B, H*W*num_anchors, num_classes].
            level_scores = level_scores.permute(0, 1, 3, 4, 2).reshape(B, -1, num_classes)
            total_preds += level_scores.shape[1]
            all_scores.append(level_scores)
        
        # Verify Sizes Match.
        assert total_preds == self.total_anchors, \
            f"Mismatch between predictions ({total_preds}) and anchors ({self.total_anchors})"
        
        # Combine Predictions.
        pred_scores = torch.cat(all_scores, dim=1)  # [B, total_anchors, num_classes]
        
        if self.debug:
            print(f"\nCombined predictions shape: {pred_scores.shape}")
            print(f"Total anchors: {self.total_anchors}")
        
        # Initialize Final Labels Tensor.
        final_labels = torch.zeros((batch_size, self.total_anchors), dtype=torch.long, device=device)
        
        # Match Anchors For Each Image.
        for b in range(batch_size):
            matched_labels, _ = self.match_anchors_to_targets(
                self.last_anchors,
                target_boxes[b],
                target_labels[b]
            )
            final_labels[b, :len(matched_labels)] = matched_labels
        
        if self.debug:
            print(f"\nFinal labels shape: {final_labels.shape}")
            print(f"Positive samples: {(final_labels > 0).sum().item()}")
        
        # Ensure no NaN values in predictions
        pred_scores = torch.clamp(pred_scores, min=1e-7, max=1-1e-7)
        
        return self.focal_loss(pred_scores, final_labels)

    def focal_loss(self, pred_scores, target_labels):
        """Compute focal loss for better handling of class imbalance."""
        # Ensure predictions are valid
        pred_scores = torch.clamp(pred_scores, min=1e-7, max=1-1e-7)
        
        # Convert targets to one-hot if needed and ensure it's long type
        if target_labels.dim() == pred_scores.dim() - 1:
            target_labels = target_labels.to(torch.long)  # Ensure long type
            target_labels = F.one_hot(target_labels, num_classes=pred_scores.shape[-1]).float()
        
        # Compute focal loss
        ce_loss = -(target_labels * torch.log(pred_scores) + 
                   (1 - target_labels) * torch.log(1 - pred_scores))
        p_t = target_labels * pred_scores + (1 - target_labels) * (1 - pred_scores)
        alpha_t = target_labels * self.alpha + (1 - target_labels) * (1 - self.alpha)
        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)
        
        return (focal_weight * ce_loss).mean()

    # ----------------------------------------------------------------------------

    # Box Loss.
    def box_loss(self, bbox_preds, gt_boxes, gt_labels):
        """Compute box regression loss with proper shape handling."""
        if self.debug:
            print("\nBox Loss Computation:")
            for i, level_preds in enumerate(bbox_preds):
                print(f"Level {i} predicted boxes shape: {level_preds.shape}")
            print(f"Ground truth boxes shape: {gt_boxes.shape}")
            print(f"Ground truth labels shape: {gt_labels.shape}")
        
        # Combine predictions from all levels
        # Each level has shape [B, num_anchors, 4, H, W]
        all_bbox_preds = []
        for level_preds in bbox_preds:
            B, num_anchors, _, H, W = level_preds.shape
            # Reshape to [B, H*W*num_anchors, 4]
            level_preds = level_preds.permute(0, 1, 3, 4, 2).reshape(B, -1, 4)
            all_bbox_preds.append(level_preds)
        
        # Concatenate predictions from all levels
        bbox_preds = torch.cat(all_bbox_preds, dim=1)  # [B, total_anchors, 4]
        
        if self.debug:
            print(f"\nCombined bbox_preds shape: {bbox_preds.shape}")
        
        # Match anchors to ground truth boxes
        batch_size = bbox_preds.shape[0]
        device = bbox_preds.device
        
        # Initialize tensors to store matched indices and valid masks
        all_matched_preds = []
        all_matched_targets = []
        
        for b in range(batch_size):
            # Get matched labels and boxes for this image
            matched_labels, matched_boxes = self.match_anchors_to_targets(
                self.last_anchors,
                gt_boxes[b],
                gt_labels[b]
            )
            
            # Find positive matches
            pos_mask = matched_labels > 0
            
            if pos_mask.sum() > 0:
                # Get corresponding predictions and targets
                matched_preds = bbox_preds[b][pos_mask]  # These are predicted deltas
                matched_targets = matched_boxes[pos_mask]  # These are target deltas
                
                # Ensure no invalid values
                matched_preds = torch.clamp(matched_preds, min=-4.0, max=4.0)
                
                all_matched_preds.append(matched_preds)
                all_matched_targets.append(matched_targets)
        
        # If no positive matches in batch, return zero loss
        if len(all_matched_preds) == 0:
            if self.debug:
                print("WARNING: No positive matches in batch!")
            return torch.tensor(0.0, device=device)
        
        # Stack all matches
        pos_pred_deltas = torch.cat(all_matched_preds, dim=0)
        raw_target_boxes = torch.cat(all_matched_targets, dim=0)
        raw_anchors = torch.cat([torch.cat(self.last_anchors).to(pos_pred_deltas.device)] * batch_size, dim=0)
        matched_anchors = raw_anchors[pos_pred_deltas.shape[0] * b : pos_pred_deltas.shape[0] * (b + 1)]

        # ✅ Encode GT boxes into deltas
        pos_target_deltas = self.box_coder.encode(matched_anchors, raw_target_boxes)

        
        if self.debug:
            print(f"\nPositive matches:")
            print(f"  Pred deltas shape: {pos_pred_deltas.shape}")
            print(f"  Target deltas shape: {pos_target_deltas.shape}")
            
            # Analyze predictions
            print("\nPrediction statistics:")
            print(f"  Range: [{pos_pred_deltas.min():.3f}, {pos_pred_deltas.max():.3f}]")
            print(f"  Mean: {pos_pred_deltas.mean():.3f}")
            print(f"  Std: {pos_pred_deltas.std():.3f}")
            
            # Analyze targets
            print("\nTarget statistics:")
            print(f"  Range: [{pos_target_deltas.min():.3f}, {pos_target_deltas.max():.3f}]")
            print(f"  Mean: {pos_target_deltas.mean():.3f}")
            print(f"  Std: {pos_target_deltas.std():.3f}")
        
        # Compute loss on deltas directly with gradient clipping
        loss = self.balanced_l1_loss(pos_pred_deltas, pos_target_deltas, self.box_beta)
        
        if self.debug:
            print(f"\nBox loss value: {loss.item():.6f}")
        
        return loss

    def balanced_l1_loss(self, pred, target, beta=0.05):
        """Compute balanced L1 loss for box regression."""
        diff = torch.abs(pred - target)
        b = math.exp(1) - 1
        loss = torch.where(
            diff < beta,
            beta / b * (b * diff + 1) * torch.log(b * diff / beta + 1) - diff,
            diff - beta / 2
        )
        return loss.mean()

    # ----------------------------------------------------------------------------

    # Assign Targets.
    def assign_targets(self, anchors, target_boxes, target_labels):
        """Quality-aware target assignment."""

        # Compute IoU Matrix.
        ious = box_iou(anchors, target_boxes)  # [num_anchors, num_targets]
        
        # Get Highest Quality Match for Each Anchor.
        max_iou_per_anchor, matched_targets = ious.max(dim=1)
        
        # Get Highest Quality Match for Each Target.
        max_iou_per_target, matched_anchors = ious.max(dim=0)
        
        # Ensure Each Target Has at Least One Positive Anchor.
        for target_idx in range(len(target_boxes)):
            best_anchor_idx = matched_anchors[target_idx]
            matched_targets[best_anchor_idx] = target_idx
            max_iou_per_anchor[best_anchor_idx] = max_iou_per_target[target_idx]
        
        # Assign Labels Based on IoU Quality.
        labels = torch.zeros_like(matched_targets)
        labels[max_iou_per_anchor > self.iou_threshold] = 1
        
        # Return Labels and Matched Targets.
        return labels, matched_targets 

    def apply_deltas(self, deltas, anchors):
        """Apply predicted deltas to anchors with size constraints."""
        # Get anchor dimensions
        anchor_widths = anchors[:, 2] - anchors[:, 0]
        anchor_heights = anchors[:, 3] - anchors[:, 1]
        anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_heights
        
        # Clamp deltas to prevent extreme values
        dx = torch.clamp(deltas[:, 0], -self.max_delta, self.max_delta)
        dy = torch.clamp(deltas[:, 1], -self.max_delta, self.max_delta)
        dw = torch.clamp(deltas[:, 2], -self.max_delta, self.max_delta)
        dh = torch.clamp(deltas[:, 3], -self.max_delta, self.max_delta)
        
        # Compute new centers
        ctr_x = dx * anchor_widths + anchor_ctr_x
        ctr_y = dy * anchor_heights + anchor_ctr_y
        
        # Compute new dimensions with exponential to ensure positive values
        w = anchor_widths * torch.exp(dw)
        h = anchor_heights * torch.exp(dh)
        
        # Enforce minimum and maximum sizes
        w = torch.clamp(w, min=self.min_size, max=self.max_size)
        h = torch.clamp(h, min=self.min_size, max=self.max_size)
        
        # Convert back to box coordinates
        x1 = ctr_x - 0.5 * w
        y1 = ctr_y - 0.5 * h
        x2 = ctr_x + 0.5 * w
        y2 = ctr_y + 0.5 * h
        
        # Clamp boxes to image boundaries
        x1 = torch.clamp(x1, 0, 1)
        y1 = torch.clamp(y1, 0, 1)
        x2 = torch.clamp(x2, 0, 1)
        y2 = torch.clamp(y2, 0, 1)
        
        # Stack coordinates
        boxes = torch.stack([x1, y1, x2, y2], dim=1)
        
        # Validate final boxes
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        
        # Create validity mask
        valid_sizes = (widths >= self.min_size) & (heights >= self.min_size) & \
                     (widths <= self.max_size) & (heights <= self.max_size)
        
        if self.debug and not valid_sizes.all():
            invalid_count = (~valid_sizes).sum().item()
            print(f"[WARNING] {invalid_count} predictions had invalid sizes after delta application")
            print(f"Size ranges: width [{widths.min():.3f}, {widths.max():.3f}], "
                  f"height [{heights.min():.3f}, {heights.max():.3f}]")
        
        return boxes, valid_sizes 