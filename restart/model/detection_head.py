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
from restart.utils.box_ops import box_iou
from restart.model.anchor_generator import AnchorGenerator

debug = False
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
                 num_classes=1,
                 anchor_generator=None,    # Anchor Generator.
                 num_convs=4,              # Number of Shared Convolutions.
                 debug=False,
                 min_size=0.01,  # Reduced minimum size to allow smaller anchors
                 max_size=0.4,   # Increased maximum size to allow larger anchors
                 max_delta=2.0): # Maximum allowed box delta
        super().__init__()
        
        # Debug Settings.
        self.debug = debug
        self.min_size = min_size
        self.max_size = max_size
        self.max_delta = max_delta

        self.anchor_generator = anchor_generator
        
        # Define strides for each FPN level (P2 to P5)
        self.strides = [4, 8, 16, 32]  # Keep strides consistent with FPN levels
        
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
        
        # Initialize Weights
        self._initialize_weights()
        
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
        
        # Classification head
        cls_scores = self.cls_head(feat)
        
        # Box regression head
        bbox_preds = self.box_head(feat)
        
        return cls_scores, bbox_preds
    
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
            
            # Take the box predictions for the highest scoring class
            bbox_pred = bbox_pred.permute(0, 1, 3, 4, 2)  # [B, num_anchors, H, W, 4]
            
            # Get scores and corresponding box predictions
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
    
        # Return Predictions.
        return {
            'cls_scores': cls_scores,
            'bbox_preds': bbox_preds,
            'anchors': anchors
        }
    

    def cls_loss(self, pred_scores, gt_boxes, gt_labels, anchors):
        """
        Compute binary classification loss (BCEWithLogitsLoss) with anchor matching.
        """
        B = len(gt_boxes)
        criterion = nn.BCEWithLogitsLoss()
        all_scores = []
        all_targets = []

        # Flatten predictions: [B, A, C, H, W] → [B, total_anchors]
        for level_scores in pred_scores:
            B, A, C, H, W = level_scores.shape
            level_scores = level_scores.permute(0, 1, 3, 4, 2).reshape(B, -1)  # [B, N]
            all_scores.append(level_scores)
        
        pred = torch.cat(all_scores, dim=1)  # [B, total_anchors]

        for b in range(B):
            matched_labels, _ = self.match_anchors_to_targets(anchors, gt_boxes[b], gt_labels[b])
            target = (matched_labels > 0).float()  # 1 for positive, 0 for background
            all_targets.append(target)

        targets = torch.stack(all_targets, dim=0)  # [B, total_anchors]
        return criterion(pred, targets)


    def box_loss(self, bbox_preds, gt_boxes, gt_labels, anchors):
        """
        Compute Smooth L1 loss for matched anchor-GT box pairs.
        """
        B = len(gt_boxes)
        criterion = nn.SmoothL1Loss()
        matched_preds = []
        matched_targets = []

        # Flatten box predictions: [B, A, H, W, 4] → [B, total_anchors, 4]
        all_pred_boxes = []
        for level_boxes in bbox_preds:
            B, A, H, W, _ = level_boxes.shape
            level_boxes = level_boxes.reshape(B, -1, 4)
            all_pred_boxes.append(level_boxes)

        pred_boxes = torch.cat(all_pred_boxes, dim=1)  # [B, total_anchors, 4]

        for b in range(B):
            matched_labels, matched_boxes = self.match_anchors_to_targets(
                anchors, gt_boxes[b], gt_labels[b]
            )
            pos_mask = matched_labels > 0

            #if debug == True:
                #print(f"[DEBUG] Image {b} - Positive anchors: {pos_mask.sum().item()}")

            if pos_mask.sum() == 0:
                continue  # skip image if no positive anchors

            matched_preds.append(pred_boxes[b][pos_mask])
            matched_targets.append(matched_boxes[pos_mask])

        if not matched_preds:
            return torch.tensor(0.0, device=anchors.device)

        return criterion(torch.cat(matched_preds), torch.cat(matched_targets))


    def match_anchors_to_targets(self, anchors, gt_boxes, gt_labels, iou_threshold=0.3):
        """
        Match anchors to GT boxes by IoU.
        Returns:
            matched_labels: Tensor [N_anchors] -> 0 = background, > 0 = class index.
            matched_gt_boxes: Tensor [N_anchors, 4] -> matched GT box per anchor.
        """

        N = anchors.size(0)
        matched_labels = torch.zeros(N, dtype=torch.long, device=anchors.device)
        matched_gt_boxes = torch.zeros((N, 4), dtype=torch.float32, device=anchors.device)

        if gt_boxes.numel() == 0:
            return matched_labels, matched_gt_boxes
            
        ious = box_iou(anchors, gt_boxes)
        
        # For each anchor, get the GT box with highest IoU
        max_iou, max_idx = ious.max(dim=1)
        
        # For each GT box, get the anchor with highest IoU
        gt_max_ious, gt_argmax = ious.max(dim=0)
        
        # Mark positive samples
        pos_mask = max_iou >= iou_threshold
        matched_labels[pos_mask] = gt_labels[max_idx[pos_mask]]
        matched_gt_boxes[pos_mask] = gt_boxes[max_idx[pos_mask]]
        
        # Ensure each GT box has at least one anchor
        matched_labels[gt_argmax] = gt_labels
        matched_gt_boxes[gt_argmax] = gt_boxes
        
        # Add center-based matching: anchors whose centers fall within GT boxes
        anchor_centers = torch.stack([
            (anchors[:, 0] + anchors[:, 2]) / 2,
            (anchors[:, 1] + anchors[:, 3]) / 2
        ], dim=1)
        
        gt_centers = torch.stack([
            (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2,
            (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
        ], dim=1)
        
        # For each GT box
        for gt_idx in range(len(gt_boxes)):
            gt_box = gt_boxes[gt_idx]
            # Check which anchor centers fall within this GT box
            inside_box = (
                (anchor_centers[:, 0] >= gt_box[0]) &
                (anchor_centers[:, 0] <= gt_box[2]) &
                (anchor_centers[:, 1] >= gt_box[1]) &
                (anchor_centers[:, 1] <= gt_box[3])
            )
            matched_labels[inside_box] = gt_labels[gt_idx]
            matched_gt_boxes[inside_box] = gt_box

        return matched_labels, matched_gt_boxes
