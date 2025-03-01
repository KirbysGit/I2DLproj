import torch
import torch.nn as nn
import torch.nn.functional as F
from .anchor_generator import AnchorGenerator
from typing import List, Tuple
from utils.box_ops import box_iou

class BoxCoder:
    """Box coordinate conversion utilities."""
    def encode(self, src_boxes, dst_boxes):
        """Convert box coordinates to deltas."""
        widths = src_boxes[:, 2] - src_boxes[:, 0]
        heights = src_boxes[:, 3] - src_boxes[:, 1]
        ctr_x = src_boxes[:, 0] + 0.5 * widths
        ctr_y = src_boxes[:, 1] + 0.5 * heights

        dst_widths = dst_boxes[:, 2] - dst_boxes[:, 0]
        dst_heights = dst_boxes[:, 3] - dst_boxes[:, 1]
        dst_ctr_x = dst_boxes[:, 0] + 0.5 * dst_widths
        dst_ctr_y = dst_boxes[:, 1] + 0.5 * dst_heights

        # Compute deltas
        dx = (dst_ctr_x - ctr_x) / widths
        dy = (dst_ctr_y - ctr_y) / heights
        dw = torch.log(dst_widths / widths)
        dh = torch.log(dst_heights / heights)

        return torch.stack([dx, dy, dw, dh], dim=1)

class DetectionHead(nn.Module):
    """
    Detection head for object detection.
    Predicts object presence and bounding box refinements for each anchor.
    """
    
    def __init__(self, 
                 in_channels=256,          # FPN channels
                 num_anchors=9,            # Anchors per location (3 ratios * 3 scales)
                 num_classes=1,            # Binary classification (object vs background)
                 num_convs=4):             # Number of shared convolutions
        super().__init__()
        
        # Create anchor generator
        self.anchor_generator = AnchorGenerator(
            base_sizes=[32, 64, 128, 256],  # For P2, P3, P4, P5
            aspect_ratios=[0.5, 1.0, 2.0],
            scales=[0.8, 1.0, 1.2]  # Additional size variations
        )
        
        # Update num_anchors based on generator
        self.num_anchors = self.anchor_generator.num_anchors
        self.num_classes = num_classes
        
        # Shared convolutions
        self.shared_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, 3, padding=1)
            for _ in range(num_convs)
        ])
        
        # Classification branch - outputs logits for each anchor
        self.cls_head = nn.Conv2d(
            in_channels, 
            self.num_anchors * num_classes,  # Output for each anchor
            3, 
            padding=1
        )
        
        # Box regression branch - outputs box deltas for each anchor
        self.box_head = nn.Conv2d(
            in_channels,
            self.num_anchors * 4,  # (dx, dy, dw, dh) for each anchor
            3,
            padding=1
        )
        
        # Loss functions
        self.cls_criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.box_criterion = nn.SmoothL1Loss(reduction='mean', beta=0.1)
        
        # Initialize weights
        self._initialize_weights()
        
        # Initialize box coder
        self.box_coder = BoxCoder()
        
    def _initialize_weights(self):
        """Initialize weights with Xavier/Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward_single(self, x):
        """
        Forward pass for a single feature level.
        
        Args:
            x (torch.Tensor): Feature map from FPN [B, C, H, W]
            
        Returns:
            tuple:
                cls_scores [B, num_anchors * num_classes, H, W]
                bbox_preds [B, num_anchors * 4, H, W]
        """
        # Shared convolutions
        feat = x
        for conv in self.shared_convs:
            feat = F.relu(conv(feat))
        
        # Classification prediction
        cls_scores = self.cls_head(feat)
        
        # Box regression prediction
        bbox_preds = self.box_head(feat)
        
        return cls_scores, bbox_preds
    
    def match_anchors_to_targets(self, anchors, target_boxes, target_labels, 
                               base_threshold=0.4,
                               center_radius=2.5):
        """Improved anchor matching with quality-aware assignment."""
        all_anchors = torch.cat(anchors, dim=0)
        device = all_anchors.device
        
        matched_labels = torch.zeros(len(all_anchors), dtype=torch.long, device=device)
        matched_boxes = torch.zeros_like(all_anchors)
        
        # Get anchor centers
        anchor_centers = (all_anchors[:, :2] + all_anchors[:, 2:]) / 2
        
        for target_idx in range(len(target_boxes)):
            target_box = target_boxes[target_idx]
            target_center = (target_box[:2] + target_box[2:]) / 2
            target_size = target_box[2:] - target_box[:2]
            
            # Calculate IoU between all anchors and current target
            ious = box_iou(all_anchors, target_box.unsqueeze(0)).squeeze(1)
            
            # Get anchors within center region
            center_radius_pixels = center_radius * torch.min(target_size)
            distances = torch.norm(anchor_centers - target_center, dim=1)
            center_mask = distances < center_radius_pixels
            
            # Quality-aware thresholding
            if center_mask.any():
                # Get IoUs for center region anchors
                center_ious = ious[center_mask]
                
                # Calculate adaptive threshold
                mean_iou = center_ious.mean()
                std_iou = center_ious.std()
                dynamic_threshold = mean_iou + std_iou
                
                # Use maximum of base and dynamic threshold
                final_threshold = max(base_threshold, dynamic_threshold.item())
                
                # Select positive anchors
                quality_mask = ious > final_threshold
                positive_mask = center_mask & quality_mask
                
                if positive_mask.any():
                    matched_labels[positive_mask] = target_labels[target_idx]
                    matched_boxes[positive_mask] = target_box
                    
                    if self.training and hasattr(self, 'verbose') and self.verbose:
                        num_positives = positive_mask.sum().item()
                        print(f"\nMatching Stats for target {target_idx}:")
                        print(f"Center anchors: {center_mask.sum().item()}")
                        print(f"Quality anchors: {quality_mask.sum().item()}")
                        print(f"Final positives: {num_positives}")
                        print(f"Mean IoU: {ious[positive_mask].mean().item():.4f}")
        
        return matched_labels, matched_boxes

    def forward(self, features):
        """Forward pass through detection head."""
        feature_list = [features[f'p{i}'] for i in range(2, 6)]  # P2 to P5
        batch_size = feature_list[0].shape[0]
        device = feature_list[0].device
        
        # Generate anchors first to know how many we need per location
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
            # Calculate anchors per grid cell
            anchors_per_cell = len(level_anchors) // (H * W)
            anchor_nums.append(anchors_per_cell)
        
        # Store for loss computation
        self.last_anchors = anchors
        self.total_anchors = sum(len(a) for a in anchors)
        
        # Get predictions using correct number of anchors per level
        cls_scores = []
        bbox_preds = []
        
        for level_id, feature in enumerate(feature_list):
            # Shared features
            feat = feature
            for conv in self.shared_convs:
                feat = F.relu(conv(feat))
            
            # Get raw predictions
            cls_score = self.cls_head(feat)  # [B, num_anchors*num_classes, H, W]
            bbox_pred = self.box_head(feat)  # [B, num_anchors*4, H, W]
            
            # Reshape predictions
            B, _, H, W = cls_score.shape
            cls_score = cls_score.view(B, -1, self.num_classes, H, W)  # [B, num_anchors, num_classes, H, W]
            bbox_pred = bbox_pred.view(B, -1, 4, H, W)  # [B, num_anchors, 4, H, W]
            
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
        
        # Debug info
        if getattr(self, 'verbose', False):
            print("\nAnchor counts per level:")
            for i, a in enumerate(anchors):
                print(f"P{i+2}: {len(a)} anchors ({anchor_nums[i]} per cell)")
            print(f"Total anchors: {self.total_anchors}")
            
            print("\nPrediction shapes:")
            for i, (cls, box) in enumerate(zip(cls_scores, bbox_preds)):
                print(f"P{i+2}: cls {cls.shape}, box {box.shape}")
                pred_anchors = cls.shape[1] * cls.shape[3] * cls.shape[4]
                print(f"Predicted anchors: {pred_anchors}")
        
        return {
            'cls_scores': cls_scores,
            'bbox_preds': bbox_preds,
            'anchors': anchors
        }

    def cls_loss(self, pred_scores, target_boxes, target_labels):
        """Updated classification loss with proper anchor matching."""
        batch_size = pred_scores[0].shape[0]
        device = pred_scores[0].device
        
        # Process and verify predictions
        all_scores = []
        total_preds = 0
        
        for level_scores in pred_scores:
            # level_scores shape: [B, num_anchors, num_classes, H, W]
            B, num_anchors, num_classes, H, W = level_scores.shape
            # Reshape to [B, H*W*num_anchors, num_classes]
            level_scores = level_scores.permute(0, 1, 3, 4, 2).reshape(B, -1, num_classes)
            total_preds += level_scores.shape[1]
            all_scores.append(level_scores)
        
        # Verify sizes match
        assert total_preds == self.total_anchors, \
            f"Mismatch between predictions ({total_preds}) and anchors ({self.total_anchors})"
        
        # Combine predictions
        pred_scores = torch.cat(all_scores, dim=1)  # [B, total_anchors, num_classes]
        
        # Initialize final labels tensor
        final_labels = torch.zeros((batch_size, self.total_anchors), device=device)
        
        # Match anchors for each image
        for b in range(batch_size):
            matched_labels, _ = self.match_anchors_to_targets(
                self.last_anchors,
                target_boxes[b],
                target_labels[b]
            )
            final_labels[b, :len(matched_labels)] = matched_labels
        
        return self.cls_criterion(pred_scores.squeeze(-1), final_labels)

    def box_loss(self, pred_boxes, target_boxes, target_labels):
        """Box regression loss with scale normalization."""
        batch_size = pred_boxes[0].shape[0]
        device = pred_boxes[0].device
        
        # Process predictions
        all_boxes = []
        for level_boxes in pred_boxes:
            B, num_anchors, box_dim, H, W = level_boxes.shape
            level_boxes = level_boxes.permute(0, 1, 3, 4, 2).reshape(B, -1, box_dim)
            all_boxes.append(level_boxes)
        
        pred_boxes = torch.cat(all_boxes, dim=1)
        all_anchors = torch.cat(self.last_anchors, dim=0).to(device)
        
        total_positives = 0
        total_loss = torch.tensor(0.0, device=device)
        
        for b in range(batch_size):
            matched_labels, matched_boxes = self.match_anchors_to_targets(
                self.last_anchors,
                target_boxes[b],
                target_labels[b]
            )
            
            positive_mask = matched_labels > 0
            num_positives = positive_mask.sum().item()
            
            if num_positives > 0:
                # Get positive samples
                pos_pred_boxes = pred_boxes[b, positive_mask]
                pos_target_boxes = matched_boxes[positive_mask]
                pos_anchors = all_anchors[positive_mask]
                
                # Normalize coordinates by image size
                img_size = 800.0  # Assuming square images
                pos_pred_boxes = pos_pred_boxes / img_size
                pos_target_boxes = pos_target_boxes / img_size
                pos_anchors = pos_anchors / img_size
                
                # Compute relative offsets
                pred_ctr_x = (pos_pred_boxes[:, 0] + pos_pred_boxes[:, 2]) * 0.5
                pred_ctr_y = (pos_pred_boxes[:, 1] + pos_pred_boxes[:, 3]) * 0.5
                pred_w = pos_pred_boxes[:, 2] - pos_pred_boxes[:, 0]
                pred_h = pos_pred_boxes[:, 3] - pos_pred_boxes[:, 1]
                
                gt_ctr_x = (pos_target_boxes[:, 0] + pos_target_boxes[:, 2]) * 0.5
                gt_ctr_y = (pos_target_boxes[:, 1] + pos_target_boxes[:, 3]) * 0.5
                gt_w = pos_target_boxes[:, 2] - pos_target_boxes[:, 0]
                gt_h = pos_target_boxes[:, 3] - pos_target_boxes[:, 1]
                
                # Compute loss on normalized coordinates
                loss_x = F.smooth_l1_loss(pred_ctr_x, gt_ctr_x, reduction='sum')
                loss_y = F.smooth_l1_loss(pred_ctr_y, gt_ctr_y, reduction='sum')
                loss_w = F.smooth_l1_loss(pred_w, gt_w, reduction='sum')
                loss_h = F.smooth_l1_loss(pred_h, gt_h, reduction='sum')
                
                box_loss = (loss_x + loss_y + loss_w + loss_h) / 4.0
                
                if self.training and hasattr(self, 'verbose') and self.verbose:
                    print(f"\nBox Loss Components (batch {b}):")
                    print(f"x: {loss_x.item():.4f}, y: {loss_y.item():.4f}")
                    print(f"w: {loss_w.item():.4f}, h: {loss_h.item():.4f}")
                
                total_loss += box_loss
                total_positives += num_positives
        
        # Return normalized loss
        return total_loss / (total_positives + 1e-8)

    def assign_targets(self, anchors, target_boxes, target_labels):
        """Quality-aware target assignment."""
        # Compute IoU matrix
        ious = box_iou(anchors, target_boxes)  # [num_anchors, num_targets]
        
        # Get highest quality match for each anchor
        max_iou_per_anchor, matched_targets = ious.max(dim=1)
        
        # Get highest quality match for each target
        max_iou_per_target, matched_anchors = ious.max(dim=0)
        
        # Ensure each target has at least one positive anchor
        for target_idx in range(len(target_boxes)):
            best_anchor_idx = matched_anchors[target_idx]
            matched_targets[best_anchor_idx] = target_idx
            max_iou_per_anchor[best_anchor_idx] = max_iou_per_target[target_idx]
        
        # Assign labels based on IoU quality
        labels = torch.zeros_like(matched_targets)
        labels[max_iou_per_anchor > self.iou_threshold] = 1
        
        return labels, matched_targets 