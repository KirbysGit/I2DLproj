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

# Object Detector Class.
class ObjectDetector(nn.Module):
    """Complete object detector with backbone, FPN, and detection head."""
    
    # Initialize Object Detector.
    def __init__(self, 
                 pretrained_backbone=True,
                 fpn_out_channels=256,
                 num_classes=1,
                 num_anchors=6):
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
            num_classes=num_classes
        )
    
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
        
        # During Training or Validation, if Targets are Provided, Return Losses.
        if boxes is not None and labels is not None:
            losses = {
                'cls_loss': self.detection_head.cls_loss(
                    head_outputs['cls_scores'], 
                    boxes,
                    labels
                ),
                'box_loss': self.detection_head.box_loss(
                    head_outputs['bbox_preds'], 
                    boxes,
                    labels
                )
            }
            return losses
        
        # During Inference, Combine Predictions from All FPN Levels.
        all_boxes = []
        all_scores = []
        
        # Iterate Over Each FPN Level.
        for level_scores, level_boxes in zip(head_outputs['cls_scores'], head_outputs['bbox_preds']):
            # level_scores: [B, num_anchors, num_classes, H, W]
            # level_boxes: [B, num_anchors, 4, H, W]
            B, num_anchors, num_classes, H, W = level_scores.shape
            
            # Reshape Predictions to [B, H*W*num_anchors, num_classes/4].
            scores = level_scores.permute(0, 1, 3, 4, 2).reshape(B, -1, num_classes)
            boxes = level_boxes.permute(0, 1, 3, 4, 2).reshape(B, -1, 4)
            
            # Append to List.
            all_boxes.append(boxes)
            all_scores.append(scores)
        
        # Concatenate Predictions from All Levels.
        boxes = torch.cat(all_boxes, dim=1)    # [B, total_anchors, 4]
        scores = torch.cat(all_scores, dim=1)  # [B, total_anchors, num_classes]
        
        # For Binary Classification, Squeeze the Class Dimension.
        if scores.shape[-1] == 1:
            scores = scores.squeeze(-1)
        
        # Return Outputs.
        return {
            'boxes': boxes,                             # [B, total_anchors, 4]
            'scores': scores,                           # [B, total_anchors, num_classes]
            'labels': torch.ones_like(scores),          # [B, total_anchors]
            'anchors': head_outputs['anchors'],         # [B, total_anchors, 4]
            'backbone_features': backbone_features,     # Dict of Features.
            'fpn_features': fpn_features,               # Dict of Features.
            'cls_scores': head_outputs['cls_scores'],   # [B, total_anchors, num_classes]
            'bbox_preds': head_outputs['bbox_preds']    # [B, total_anchors, 4]
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
        num_anchors=config['num_anchors']
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