# model / detector.py

# -----

# Defines Complete Object Detection Model.
# Combines Backbone, Feature Pyramid Network, & Detection Head.

# ----- 

# Imports.
from .backbone import ResNetBackbone
from .fpn import FeaturePyramidNetwork
from .detection_head import DetectionHead
import torch
import torch.nn as nn
from torchvision.ops import nms

# Object Detector Class.
class ObjectDetector(nn.Module):
    """Complete object detector with backbone, FPN, and detection head."""
    
    # Initialize Object Detector.
    def __init__(self, 
                 pretrained_backbone=True,
                 fpn_out_channels=256,
                 num_classes=3,
                 num_anchors=3,
                 debug=False):
        super().__init__()
        
        # Backbone.
        self.backbone = ResNetBackbone(pretrained=pretrained_backbone)
        
        # FPN
        in_channels_list = [
            self.backbone.out_channels['layer1'],  # 256
            self.backbone.out_channels['layer2'],  # 512
            self.backbone.out_channels['layer3'],  # 1024
            self.backbone.out_channels['layer4'],  # 2048
        ]
        self.fpn = FeaturePyramidNetwork(in_channels_list, fpn_out_channels)
        
        # Detection Head.
        self.detection_head = DetectionHead(
            in_channels=fpn_out_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
            debug=debug
        )
        
        # Set thresholds based on ground truth statistics
        self.confidence_thresholds = [0.15, 0.15, 0.15, 0.15]  # Keep uniform thresholds
        
        # Adjust size ranges to match ground truth statistics
        self.min_sizes = [0.01, 0.02, 0.03, 0.04]  # Minimum sizes matching GT minimum
        self.max_sizes = [0.06, 0.08, 0.10, 0.12]  # Maximum sizes matching GT maximum
        
        # More permissive NMS to avoid removing valid detections
        self.nms_thresholds = [0.5, 0.5, 0.5, 0.5]  # Less aggressive NMS
        
        # Box regression parameters - adjust for better localization
        self.box_scale = [4, 8, 16, 32]  # Reduced scale factors for finer control
        self.box_delta_means = [0.0, 0.0, 0.0, 0.0]
        self.box_delta_stds = [0.2, 0.2, 0.2, 0.2]  # Increased std for more flexibility
        
        # Size scaling factors - adjusted based on GT statistics
        self.width_scales = [0.8, 0.9, 1.0, 1.1]   # Less aggressive width scaling
        self.height_scales = [1.0, 1.2, 1.4, 1.6]  # Less aggressive height scaling
    
    # Forward Pass.
    def forward(self, images, boxes=None, labels=None):
        """Forward pass through the complete detector."""
        # Get Backbone Features.
        backbone_features = self.backbone(images)
        
        # Get Features for FPN (C2 to C5).
        fpn_input_features = {
            k: v for k, v in backbone_features.items() 
            if k in ['layer1', 'layer2', 'layer3', 'layer4']
        }
        
        # Generate FPN Features (P2 to P5).
        fpn_features = self.fpn(fpn_input_features)
        
        # Get Predictions and Anchors from Detection Head.
        head_outputs = self.detection_head(fpn_features)
        
        # Initialize lists to store predictions from all levels
        all_boxes = []
        all_scores = []
        all_labels = []
        
        # Process each FPN level
        for level_idx, (level_scores, level_boxes, level_anchors) in enumerate(zip(
            head_outputs['cls_scores'], 
            head_outputs['bbox_preds'],
            head_outputs['anchors']
        )):
            print(f"\nLevel {level_idx} shapes:")
            print(f"Scores: {level_scores.shape}")
            print(f"Boxes: {level_boxes.shape}")
            print(f"Anchors: {level_anchors.shape}")
            
            # Move tensors to CPU for processing
            level_scores = level_scores.cpu()
            level_boxes = level_boxes.cpu()
            level_anchors = level_anchors.cpu()
            
            B, num_anchors, num_classes, H, W = level_scores.shape
            
            # Reshape predictions
            level_scores = level_scores.permute(0, 1, 3, 4, 2).reshape(B, -1, num_classes)
            level_boxes = level_boxes.permute(0, 1, 3, 4, 2).reshape(B, -1, 4)
            
            # Apply sigmoid to get confidence scores
            level_scores = torch.sigmoid(level_scores)
            
            # Convert box deltas to actual boxes
            anchor_widths = level_anchors[:, 2] - level_anchors[:, 0]
            anchor_heights = level_anchors[:, 3] - level_anchors[:, 1]
            anchor_ctr_x = level_anchors[:, 0] + 0.5 * anchor_widths
            anchor_ctr_y = level_anchors[:, 1] + 0.5 * anchor_heights
            
            # Apply box regression with level-specific bounds and scaling
            dx = level_boxes[..., 0] * self.box_delta_stds[0] + self.box_delta_means[0]
            dy = level_boxes[..., 1] * self.box_delta_stds[1] + self.box_delta_means[1]
            dw = level_boxes[..., 2] * self.box_delta_stds[2] + self.box_delta_means[2]
            dh = level_boxes[..., 3] * self.box_delta_stds[3] + self.box_delta_means[3]
            
            # Get image size for normalization
            image_size = float(images.shape[-1])  # Assuming square images
            
            # Normalize anchor coordinates to [0, 1]
            anchor_widths = anchor_widths / image_size
            anchor_heights = anchor_heights / image_size
            anchor_ctr_x = anchor_ctr_x / image_size
            anchor_ctr_y = anchor_ctr_y / image_size
            
            # Scale deltas based on feature level
            scale = self.box_scale[level_idx]
            feature_scale = 1.0 / scale  # Convert scale to fraction
            
            # Center deltas are relative to feature scale and anchor size
            dx = dx.clamp(-1.0, 1.0) * feature_scale * anchor_widths.unsqueeze(0)
            dy = dy.clamp(-1.0, 1.0) * feature_scale * anchor_heights.unsqueeze(0)
            
            # Calculate predicted centers (normalized)
            pred_ctr_x = anchor_ctr_x.unsqueeze(0) + dx
            pred_ctr_y = anchor_ctr_y.unsqueeze(0) + dy
            
            # Clamp centers to valid range
            pred_ctr_x = pred_ctr_x.clamp(0.0, 1.0)
            pred_ctr_y = pred_ctr_y.clamp(0.0, 1.0)
            
            # Size deltas use wider range for better size variation
            dw = torch.clamp(dw, -2.0, 2.0)  # Allow more size variation
            dh = torch.clamp(dh, -2.0, 2.0)
            
            # Calculate base sizes from anchors with level-specific scaling
            base_w = anchor_widths.unsqueeze(0) * self.width_scales[level_idx]
            base_h = anchor_heights.unsqueeze(0) * self.height_scales[level_idx]
            
            # Apply size deltas with reduced scaling for better control
            width_scale = 0.2 + level_idx * 0.1  # Gentler progressive scaling
            height_scale = 0.4 + level_idx * 0.2  # Gentler progressive scaling
            
            pred_w = base_w * torch.exp(dw * width_scale)
            pred_h = base_h * torch.exp(dh * height_scale)
            
            # Ensure sizes match ground truth ranges
            min_size = self.min_sizes[level_idx]
            max_size = self.max_sizes[level_idx]
            
            # Allow smaller boxes to match GT minimums
            effective_min_w = min_size * 0.8  # Less reduction of minimum width
            effective_min_h = min_size * 0.8  # Less reduction of minimum height
            
            # Limit maximum sizes to avoid oversized predictions
            effective_max_w = max_size * 0.9  # Stricter maximum width
            effective_max_h = max_size * 1.2  # Allow slightly taller boxes
            
            pred_w = torch.clamp(pred_w, effective_min_w, effective_max_w)
            pred_h = torch.clamp(pred_h, effective_min_h, effective_max_h)
            
            # Convert to [x1, y1, x2, y2] format with careful handling of small boxes
            pred_boxes = torch.stack([
                (pred_ctr_x - 0.5 * pred_w).clamp(0.0, 1.0),
                (pred_ctr_y - 0.5 * pred_h).clamp(0.0, 1.0),
                (pred_ctr_x + 0.5 * pred_w).clamp(0.0, 1.0),
                (pred_ctr_y + 0.5 * pred_h).clamp(0.0, 1.0)
            ], dim=-1)
            
            print(f"\nBox regression statistics for level {level_idx}:")
            print(f"Center ranges: x=[{pred_ctr_x.min():.4f}, {pred_ctr_x.max():.4f}], y=[{pred_ctr_y.min():.4f}, {pred_ctr_y.max():.4f}]")
            print(f"Size ranges: w=[{pred_w.min():.4f}, {pred_w.max():.4f}], h=[{pred_h.min():.4f}, {pred_h.max():.4f}]")
            
            # Get confidence scores and filter by level-specific threshold
            max_scores, max_classes = level_scores.max(dim=-1)
            confidence_mask = max_scores > self.confidence_thresholds[level_idx]
            
            print(f"\nChunk statistics:")
            print(f"Max score: {max_scores.max().item():.4f}")
            print(f"Mean score: {max_scores.mean().item():.4f}")
            print(f"Predictions above threshold: {confidence_mask.sum().item()}")
            
            # Initialize lists for this level
            level_filtered_boxes = []
            level_filtered_scores = []
            level_filtered_labels = []
            
            if confidence_mask.any():
                # Process each batch item
                for b in range(B):
                    b_mask = confidence_mask[b]
                    if not b_mask.any():
                        continue
                        
                    b_boxes = pred_boxes[b][b_mask]
                    b_scores = max_scores[b][b_mask]
                    b_labels = max_classes[b][b_mask]
                    
                    print(f"\nInitial predictions for batch {b}:")
                    print(f"Number of boxes: {len(b_boxes)}")
                    print(f"Score range: [{b_scores.min().item():.4f}, {b_scores.max().item():.4f}]")
                    
                    # Filter by box size with level-specific thresholds
                    widths = b_boxes[:, 2] - b_boxes[:, 0]
                    heights = b_boxes[:, 3] - b_boxes[:, 1]
                    
                    print(f"Box dimensions before filtering:")
                    print(f"Width range: [{widths.min().item():.4f}, {widths.max().item():.4f}]")
                    print(f"Height range: [{heights.min().item():.4f}, {heights.max().item():.4f}]")
                    
                    min_size = self.min_sizes[level_idx]
                    max_size = self.max_sizes[level_idx]
                    
                    # More permissive size filtering
                    size_mask = (widths > min_size * 0.5) & (heights > min_size * 0.5) & \
                               (widths < max_size * 2.0) & (heights < max_size * 2.0)
                    
                    print(f"Size filtering:")
                    print(f"Min size threshold: {min_size:.4f}")
                    print(f"Max size threshold: {max_size:.4f}")
                    print(f"Boxes passing size filter: {size_mask.sum().item()}")
                    
                    if not size_mask.any():
                        print("No boxes passed size filtering!")
                        continue
                        
                    b_boxes = b_boxes[size_mask]
                    b_scores = b_scores[size_mask]
                    b_labels = b_labels[size_mask]
                    
                    print(f"\nAfter size filtering:")
                    print(f"Number of boxes: {len(b_boxes)}")
                    print(f"Width range: [{widths[size_mask].min().item():.4f}, {widths[size_mask].max().item():.4f}]")
                    print(f"Height range: [{heights[size_mask].min().item():.4f}, {heights[size_mask].max().item():.4f}]")
                    
                    # Apply NMS with more permissive threshold
                    keep = nms(b_boxes, b_scores, iou_threshold=self.nms_thresholds[level_idx])
                    print(f"Boxes after NMS: {len(keep)}")
                    
                    if len(keep) > 0:
                        level_filtered_boxes.append(b_boxes[keep])
                        level_filtered_scores.append(b_scores[keep])
                        level_filtered_labels.append(b_labels[keep])
                        print(f"Added {len(keep)} boxes from level {level_idx}")
            
            # Add level predictions to overall lists if we found any
            if level_filtered_boxes:
                print(f"\nAdding predictions from level {level_idx}:")
                print(f"Number of predictions: {sum(len(boxes) for boxes in level_filtered_boxes)}")
                all_boxes.extend(level_filtered_boxes)
                all_scores.extend(level_filtered_scores)
                all_labels.extend(level_filtered_labels)
        
        # If we have any valid predictions
        if all_boxes:
            # Stack predictions from all levels
            boxes = torch.cat(all_boxes, dim=0)
            scores = torch.cat(all_scores, dim=0)
            labels = torch.cat(all_labels, dim=0)
            
            # Move back to original device
            boxes = boxes.to(images.device)
            scores = scores.to(images.device)
            labels = labels.to(images.device)
            
            print("\nFinal statistics:")
            print(f"Total boxes: {len(boxes)}")
            print(f"Score range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
            
            return {
                'boxes': boxes.unsqueeze(0),
                'scores': scores.unsqueeze(0),
                'labels': labels.unsqueeze(0) + 1,  # Convert to 1-based indexing
                'all_scores': level_scores.to(images.device)  # Keep original scores for debugging
            }
        else:
            # Return empty predictions
            return {
                'boxes': torch.zeros((1, 0, 4), device=images.device),
                'scores': torch.zeros((1, 0), device=images.device),
                'labels': torch.zeros((1, 0), dtype=torch.long, device=images.device),
                'all_scores': level_scores.to(images.device)  # Keep original scores for debugging
            }

# ----------------------------------------------------------------------------

# Build Detector.   
def build_detector(config):
    """Build the complete detection model."""
    # Create Backbone.
    backbone = ResNetBackbone(
        pretrained=config['pretrained_backbone']
    )
    
    # Create FPN with Proper Channel List.
    in_channels_list = [
        backbone.out_channels['layer1'],  # 256
        backbone.out_channels['layer2'],  # 512
        backbone.out_channels['layer3'],  # 1024
        backbone.out_channels['layer4']   # 2048
    ]
    
    # Create FPN.
    fpn = FeaturePyramidNetwork(
        in_channels_list=in_channels_list,  # Pass the list instead of the dict
        out_channels=config['fpn_out_channels']
    )
    
    # Create Detector.
    detector = ObjectDetector(
        pretrained_backbone=config['pretrained_backbone'],
        fpn_out_channels=config['fpn_out_channels'],
        num_classes=config['num_classes'],
        num_anchors=config['num_anchors'],
        debug=config.get('debug', False)  # Get debug setting from config
    )
    
    return detector

# ----------------------------------------------------------------------------

# Detection Model Class.
class DetectionModel(nn.Module):
    """Complete detection model."""
    
    def __init__(self, backbone, fpn, num_classes, num_anchors):
        super().__init__()
        self.backbone = backbone
        self.fpn = fpn
        # ... might come back to this 

    def forward(self, x, boxes=None, labels=None, debug=False):
        """Forward pass of the detector.
        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W]
            boxes (torch.Tensor, optional): Ground truth boxes of shape [B, N, 4]
            labels (torch.Tensor, optional): Ground truth labels of shape [B, N]
            debug (bool, optional): Whether to print debug information
        Returns:
            dict: Dictionary containing the model outputs
        """
        # Extract features using the backbone
        features = self.backbone(x)
        
        # Generate FPN features
        fpn_features = self.fpn(features)
        
        # Initialize lists to store predictions from each level
        all_scores = []
        all_boxes = []
        all_anchors = []
        
        # Process each FPN level
        for level, feature in enumerate(fpn_features):
            if debug:
                print(f"\nLevel {level} shapes:")
                print(f"Scores: {feature.shape}")
                print(f"Boxes: {feature.shape}")
                print(f"Anchors: {self.anchor_generator.anchors[level].shape}")
            
            # Get predictions for this level
            level_scores, level_boxes = self.detection_head(feature)
            level_anchors = self.anchor_generator.anchors[level]
            
            # Reshape predictions for processing
            B, _, H, W = feature.shape
            level_scores = level_scores.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)
            level_boxes = level_boxes.permute(0, 2, 3, 1).reshape(B, -1, 4)
            
            # Apply sigmoid to get confidence scores
            level_scores = torch.sigmoid(level_scores)
            
            # Convert box deltas to actual boxes
            level_boxes = self.apply_deltas_to_anchors(level_boxes, level_anchors)
            
            # Clip boxes to valid range [0, 1]
            level_boxes = torch.clamp(level_boxes, 0, 1)
            
            all_scores.append(level_scores)
            all_boxes.append(level_boxes)
            all_anchors.append(level_anchors)
        
        # Concatenate predictions from all levels
        scores = torch.cat(all_scores, dim=1)  # [B, N, num_classes]
        boxes = torch.cat(all_boxes, dim=1)    # [B, N, 4]
        anchors = torch.cat(all_anchors, dim=0)  # [N, 4]
        
        if debug:
            print(f"\nTotal predictions: {len(boxes[0])}")
            print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        
        # During training, return the raw predictions
        if boxes is not None and labels is not None:
            return {
                'pred_boxes': boxes,
                'pred_scores': scores,
                'gt_boxes': boxes,
                'gt_labels': labels
            }
        
        # During inference, filter and process predictions
        filtered_boxes = []
        filtered_scores = []
        filtered_labels = []
        
        for b in range(len(boxes)):
            # Get maximum score and corresponding class for each prediction
            max_scores, max_classes = scores[b].max(dim=1)
            
            # Filter by confidence threshold
            mask = max_scores > self.confidence_threshold
            if mask.sum() == 0:
                if debug:
                    print("\nNo predictions above confidence threshold!")
                continue
            
            b_boxes = boxes[b][mask]
            b_scores = max_scores[mask]
            b_labels = max_classes[mask]
            
            # Filter by box size
            widths = b_boxes[:, 2] - b_boxes[:, 0]
            heights = b_boxes[:, 3] - b_boxes[:, 1]
            size_mask = (widths > 0.01) & (heights > 0.01) & (widths < 0.99) & (heights < 0.99)
            
            if size_mask.sum() == 0:
                if debug:
                    print("\nNo valid box sizes!")
                continue
            
            b_boxes = b_boxes[size_mask]
            b_scores = b_scores[size_mask]
            b_labels = b_labels[size_mask]
            
            if debug:
                print(f"\nBox statistics:")
                print(f"Total boxes: {len(b_boxes)}")
                print(f"Width range: [{widths[size_mask].min():.4f}, {widths[size_mask].max():.4f}]")
                print(f"Height range: [{heights[size_mask].min():.4f}, {heights[size_mask].max():.4f}]")
            
            # Apply NMS per class
            keep_indices = nms(b_boxes, b_scores, self.nms_threshold)
            
            filtered_boxes.append(b_boxes[keep_indices])
            filtered_scores.append(b_scores[keep_indices])
            filtered_labels.append(b_labels[keep_indices])
        
        # If no predictions survived filtering
        if not filtered_boxes:
            return {
                'boxes': torch.zeros((0, 4), device=x.device),
                'scores': torch.zeros(0, device=x.device),
                'labels': torch.zeros(0, dtype=torch.long, device=x.device),
                'all_scores': scores  # Keep all class scores for debugging [B, N, num_classes]
            }
        
        return {
            'boxes': torch.stack(filtered_boxes),
            'scores': torch.stack(filtered_scores),
            'labels': torch.stack(filtered_labels),
            'all_scores': scores  # Keep all class scores for debugging [B, N, num_classes]
        }

    def apply_deltas_to_anchors(self, deltas, anchors):
        """Convert box deltas to actual boxes using anchor boxes.
        Args:
            deltas (torch.Tensor): Box deltas of shape [B, N, 4]
            anchors (torch.Tensor): Anchor boxes of shape [N, 4]
        Returns:
            torch.Tensor: Predicted boxes of shape [B, N, 4]
        """
        # Extract anchor coordinates
        anchor_widths = anchors[:, 2] - anchors[:, 0]
        anchor_heights = anchors[:, 3] - anchors[:, 1]
        anchor_ctr_x = anchors[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y = anchors[:, 1] + 0.5 * anchor_heights
        
        # Extract predicted deltas
        dx = deltas[..., 0]
        dy = deltas[..., 1]
        dw = deltas[..., 2]
        dh = deltas[..., 3]
        
        # Prevent extreme deltas
        dw = torch.clamp(dw, max=4.0)
        dh = torch.clamp(dh, max=4.0)
        
        # Apply deltas
        pred_ctr_x = dx * anchor_widths + anchor_ctr_x
        pred_ctr_y = dy * anchor_heights + anchor_ctr_y
        pred_w = torch.exp(dw) * anchor_widths
        pred_h = torch.exp(dh) * anchor_heights
        
        # Convert to [x1, y1, x2, y2] format
        pred_boxes = torch.zeros_like(deltas)
        pred_boxes[..., 0] = pred_ctr_x - 0.5 * pred_w
        pred_boxes[..., 1] = pred_ctr_y - 0.5 * pred_h
        pred_boxes[..., 2] = pred_ctr_x + 0.5 * pred_w
        pred_boxes[..., 3] = pred_ctr_y + 0.5 * pred_h
        
        return pred_boxes 