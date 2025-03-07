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
from .anchor_generator import AnchorGenerator
from typing import List, Tuple
from src.utils.box_ops import box_iou

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
                 debug=False):             # Debug flag
        super().__init__()
        
        # Debug settings
        self.debug = debug
        
        # Create Anchor Generator with reduced anchors
        self.anchor_generator = AnchorGenerator(
            base_sizes=[32, 64, 128, 256],  # For P2, P3, P4, P5
            aspect_ratios=[1.0],  # Single aspect ratio
            scales=[1.0]  # Single scale
        )
        
        # Cache for anchor matching
        self._cached_anchors = None
        self._cached_image_size = None
        
        # Update num_anchors based on Generator.
        self.num_anchors = len(self.anchor_generator.aspect_ratios) * len(self.anchor_generator.scales)
        self.num_classes = num_classes
        
        # Shared Convolutions with consistent channels
        self.shared_convs = nn.ModuleList()
        curr_channels = in_channels
        
        # First n-1 convolutions maintain input channels
        for _ in range(num_convs - 1):
            self.shared_convs.append(
                nn.Conv2d(curr_channels, curr_channels, 3, padding=1)
            )
        
        # Last convolution can reduce channels if needed
        self.shared_convs.append(
            nn.Conv2d(curr_channels, curr_channels, 3, padding=1)
        )
        
        # Classification Branch (3 anchors × 3 classes = 9 channels)
        self.cls_head = nn.Conv2d(
            curr_channels, 
            self.num_anchors * num_classes,  # Will be 9 channels
            3, 
            padding=1
        )
        
        # Box Regression Branch (3 anchors × 4 coordinates × 3 classes = 36 channels)
        self.box_head = nn.Conv2d(
            curr_channels,
            self.num_anchors * 4 * num_classes,  # Will be 36 channels
            3,
            padding=1
        )
        
        # Loss Functions
        self.cls_criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.box_criterion = nn.SmoothL1Loss(reduction='mean', beta=0.1)
        
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
        """
        Forward pass for a single feature level.
        
        Args:
            x (torch.Tensor): Feature Map from FPN [B, C, H, W].
            
        Returns:
            tuple:
                cls_scores [B, num_anchors * num_classes, H, W]
                bbox_preds [B, num_anchors * 4, H, W]
        """
        # Shared Convolutions with ReLU activation
        feat = x
        for conv in self.shared_convs:
            feat = F.relu(conv(feat))
            if self.debug:
                print(f"Feature shape after conv: {feat.shape}")
        
        # Classification Prediction
        cls_scores = self.cls_head(feat)
        
        # Box Regression Prediction
        bbox_preds = self.box_head(feat)
        
        if self.debug:
            print(f"Classification scores shape: {cls_scores.shape}")
            print(f"Box predictions shape: {bbox_preds.shape}")
        
        return cls_scores, bbox_preds
    
    # ----------------------------------------------------------------------------

    # Match Anchors to Targets.
    def match_anchors_to_targets(self, anchors, target_boxes, target_labels, 
                               iou_threshold=0.35,  # Lowered threshold for more matches
                               max_matches_per_target=25):  # Increased matches
        """Improved anchor matching with lower threshold."""
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
        
        # For each target, find top-k matching anchors
        max_ious, anchor_indices = ious.topk(k=min(max_matches_per_target, len(anchors)), dim=0)
        
        # Filter matches by IoU threshold
        valid_matches = max_ious >= iou_threshold
        
        # Get corresponding target indices
        target_indices = torch.arange(len(target_boxes), device=device).unsqueeze(0).expand_as(anchor_indices)
        
        # Filter and flatten indices
        valid_anchor_idx = anchor_indices[valid_matches]
        valid_target_idx = target_indices[valid_matches]
        
        if len(valid_anchor_idx) > 0:
            # Assign labels and boxes
            matched_labels[valid_anchor_idx] = target_labels[valid_target_idx]
            matched_boxes[valid_anchor_idx] = target_boxes[valid_target_idx]
            
            if self.debug:
                print(f"\nAnchor matching statistics:")
                print(f"  Total anchors: {num_anchors}")
                print(f"  Valid matches: {len(valid_anchor_idx)}")
                print(f"  Average IoU: {max_ious[valid_matches].mean():.4f}")
        
        return matched_labels, matched_boxes

    # ----------------------------------------------------------------------------

    # Forward Pass.
    def forward(self, features):
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
            
            # Get Raw Predictions.
            cls_score = self.cls_head(feat)  # [B, num_anchors*num_classes, H, W]
            bbox_pred = self.box_head(feat)  # [B, num_anchors*4*num_classes, H, W]
            
            # Reshape Predictions.
            B, _, H, W = cls_score.shape
            cls_score = cls_score.view(B, self.num_anchors, self.num_classes, H, W)  # [B, num_anchors, num_classes, H, W]
            bbox_pred = bbox_pred.view(B, self.num_anchors, 4, self.num_classes, H, W)  # [B, num_anchors, 4, num_classes, H, W]
            
            # Take the box predictions for the highest scoring class
            bbox_pred = bbox_pred.permute(0, 1, 3, 2, 4, 5)  # [B, num_anchors, num_classes, 4, H, W]
            
            # Get scores and corresponding box predictions
            cls_probs = torch.sigmoid(cls_score)  # [B, num_anchors, num_classes, H, W]
            best_class = cls_probs.argmax(dim=2, keepdim=True)  # [B, num_anchors, 1, H, W]
            best_class_expanded = best_class.unsqueeze(3).expand(-1, -1, -1, 4, -1, -1)  # [B, num_anchors, 1, 4, H, W]
            bbox_pred = torch.gather(bbox_pred, 2, best_class_expanded).squeeze(2)  # [B, num_anchors, 4, H, W]
            
            # Add box size constraints and normalization
            # Convert deltas to actual coordinates
            bbox_pred = bbox_pred.permute(0, 1, 3, 4, 2).reshape(B, -1, 4)  # [B, H*W*num_anchors, 4]
            level_anchors = anchors[level_id].unsqueeze(0).expand(B, -1, -1)  # [B, H*W*num_anchors, 4]
            
            # Normalize anchor coordinates to [0, 1] range
            image_size = torch.tensor([W, H, W, H], device=device).float()
            level_anchors_norm = level_anchors / image_size
            
            # Convert predicted deltas to actual coordinates (using normalized anchors)
            pred_ctr_x = level_anchors_norm[..., 0] + bbox_pred[..., 0] * level_anchors_norm[..., 2]
            pred_ctr_y = level_anchors_norm[..., 1] + bbox_pred[..., 1] * level_anchors_norm[..., 3]
            pred_w = level_anchors_norm[..., 2] * torch.exp(torch.clamp(bbox_pred[..., 2], min=-3.0, max=3.0))  # Less restrictive
            pred_h = level_anchors_norm[..., 3] * torch.exp(torch.clamp(bbox_pred[..., 3], min=-3.0, max=3.0))
            
            # Add size constraints relative to image size
            min_size = 0.005  # Allow smaller boxes
            max_size = 0.98   # Allow larger boxes
            
            pred_w = torch.clamp(pred_w, min=min_size, max=max_size)
            pred_h = torch.clamp(pred_h, min=min_size, max=max_size)
            
            # Convert back to x1,y1,x2,y2 format (normalized)
            pred_x1 = pred_ctr_x - 0.5 * pred_w
            pred_y1 = pred_ctr_y - 0.5 * pred_h
            pred_x2 = pred_ctr_x + 0.5 * pred_w
            pred_y2 = pred_ctr_y + 0.5 * pred_h
            
            # Ensure boxes are within normalized image bounds [0, 1] with small margin
            pred_x1 = torch.clamp(pred_x1, min=0.0, max=0.99)
            pred_y1 = torch.clamp(pred_y1, min=0.0, max=0.99)
            pred_x2 = torch.clamp(pred_x2, min=0.01, max=1.0)
            pred_y2 = torch.clamp(pred_y2, min=0.01, max=1.0)
            
            # Ensure minimum box size
            box_w = pred_x2 - pred_x1
            box_h = pred_y2 - pred_y1
            invalid_boxes = (box_w < min_size) | (box_h < min_size)
            pred_x2[invalid_boxes] = pred_x1[invalid_boxes] + min_size
            pred_y2[invalid_boxes] = pred_y1[invalid_boxes] + min_size
            
            # Stack and reshape back to original format
            bbox_pred = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)
            bbox_pred = bbox_pred.view(B, self.num_anchors, H, W, 4).permute(0, 1, 4, 2, 3)
            
            # Append Predictions.
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
        final_labels = torch.zeros((batch_size, self.total_anchors), device=device)
        
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
        
        return self.cls_criterion(pred_scores.squeeze(-1), final_labels)

    # ----------------------------------------------------------------------------

    # Box Loss.
    def box_loss(self, bbox_preds, gt_boxes, gt_labels):
        """Compute box regression loss."""
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
                
                all_matched_preds.append(matched_preds)
                all_matched_targets.append(matched_targets)
        
        # If no positive matches in batch, return zero loss
        if len(all_matched_preds) == 0:
            if self.debug:
                print("WARNING: No positive matches in batch!")
            return torch.tensor(0.0, device=device)
        
        # Stack all matches
        pos_pred_deltas = torch.cat(all_matched_preds, dim=0)
        pos_target_deltas = torch.cat(all_matched_targets, dim=0)
        
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
        
        # Compute loss on deltas directly
        loss = F.smooth_l1_loss(pos_pred_deltas, pos_target_deltas, reduction='mean', beta=0.1)
        
        if self.debug:
            print(f"\nBox loss value: {loss.item():.6f}")
        
        return loss

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