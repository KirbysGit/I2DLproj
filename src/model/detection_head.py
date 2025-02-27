import torch
import torch.nn as nn
import torch.nn.functional as F
from .anchor_generator import AnchorGenerator
from typing import List, Tuple
from .box_ops import box_iou

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
    
    def match_anchors_to_targets(self, anchors: List[torch.Tensor], 
                               target_boxes: torch.Tensor,
                               target_labels: torch.Tensor,
                               iou_threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Match anchors to ground truth boxes.
        
        Args:
            anchors: List of anchor boxes for each FPN level [num_anchors, 4]
            target_boxes: Ground truth boxes [num_boxes, 4]
            target_labels: Ground truth labels [num_boxes]
            iou_threshold: IoU threshold for positive matches
            
        Returns:
            matched_labels: Labels for each anchor (0: background, 1: object)
            matched_boxes: Target boxes for each anchor
        """
        # Combine all anchors
        all_anchors = torch.cat(anchors, dim=0)
        device = all_anchors.device
        
        # Initialize labels as background (0)
        matched_labels = torch.zeros(len(all_anchors), device=device)
        matched_boxes = torch.zeros_like(all_anchors)
        
        if len(target_boxes) == 0:
            return matched_labels, matched_boxes
        
        # Calculate IoU between all anchors and all target boxes
        ious = box_iou(all_anchors, target_boxes)  # [num_anchors, num_targets]
        
        # For each anchor, get the best matching target
        best_target_iou, best_target_idx = ious.max(dim=1)
        
        # Assign positive labels where IoU > threshold
        positive_mask = best_target_iou > iou_threshold
        matched_labels[positive_mask] = 1
        
        # Assign corresponding target boxes
        matched_boxes[positive_mask] = target_boxes[best_target_idx[positive_mask]]
        
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
        """Box regression loss with proper anchor matching."""
        batch_size = pred_boxes[0].shape[0]
        device = pred_boxes[0].device
        
        # Process predictions from each FPN level
        all_boxes = []
        for level_boxes in pred_boxes:
            # level_boxes shape: [B, num_anchors, 4, H, W]
            B, num_anchors, box_dim, H, W = level_boxes.shape
            # Reshape to [B, H*W*num_anchors, 4]
            level_boxes = level_boxes.permute(0, 1, 3, 4, 2).reshape(B, -1, box_dim)
            all_boxes.append(level_boxes)
        
        # Combine predictions from all levels
        pred_boxes = torch.cat(all_boxes, dim=1)  # [B, total_anchors, 4]
        
        # Initialize tensors with known size
        final_boxes = torch.zeros((batch_size, self.total_anchors, 4), device=device)
        valid_mask = torch.zeros((batch_size, self.total_anchors), device=device)
        
        # Match anchors for each image
        for b in range(batch_size):
            matched_labels, matched_boxes = self.match_anchors_to_targets(
                self.last_anchors,
                target_boxes[b],
                target_labels[b]
            )
            final_boxes[b, :len(matched_boxes)] = matched_boxes
            valid_mask[b, :len(matched_labels)] = matched_labels
        
        # Calculate loss only for positive anchors
        box_loss = self.box_criterion(pred_boxes, final_boxes)
        box_loss = (box_loss * valid_mask.unsqueeze(-1)).sum() / (valid_mask.sum() + 1e-6)
        
        return box_loss 