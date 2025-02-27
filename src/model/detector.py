from .backbone import ResNetBackbone
from .fpn import FeaturePyramidNetwork
from .detection_head import DetectionHead
import torch
import torch.nn as nn

class ObjectDetector(nn.Module):
    """Complete object detector with backbone, FPN, and detection head."""
    
    def __init__(self, 
                 pretrained_backbone=True,
                 fpn_out_channels=256,
                 num_classes=1,
                 num_anchors=6):
        super().__init__()
        
        # Backbone
        self.backbone = ResNetBackbone(pretrained=pretrained_backbone)
        
        # FPN
        in_channels_list = [
            self.backbone.out_channels['layer1'],  # 256
            self.backbone.out_channels['layer2'],  # 512
            self.backbone.out_channels['layer3'],  # 1024
            self.backbone.out_channels['layer4'],  # 2048
        ]
        self.fpn = FeaturePyramidNetwork(in_channels_list, fpn_out_channels)
        
        # Detection Head
        self.detection_head = DetectionHead(
            in_channels=fpn_out_channels,
            num_anchors=num_anchors,
            num_classes=num_classes
        )
        
    def forward(self, images, boxes=None, labels=None):
        """Forward pass through the complete detector."""
        # Get backbone features
        backbone_features = self.backbone(images)
        
        # Get features for FPN (C2 to C5)
        fpn_input_features = {
            k: v for k, v in backbone_features.items() 
            if k in ['layer1', 'layer2', 'layer3', 'layer4']
        }
        
        # Generate FPN features (P2 to P5)
        fpn_features = self.fpn(fpn_input_features)
        
        # Get predictions and anchors from detection head
        head_outputs = self.detection_head(fpn_features)
        
        # During training or validation, if targets are provided, return losses
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
        
        # During inference, combine predictions from all FPN levels
        all_boxes = []
        all_scores = []
        
        for level_scores, level_boxes in zip(head_outputs['cls_scores'], head_outputs['bbox_preds']):
            # level_scores: [B, num_anchors, num_classes, H, W]
            # level_boxes: [B, num_anchors, 4, H, W]
            B, num_anchors, num_classes, H, W = level_scores.shape
            
            # Reshape predictions to [B, H*W*num_anchors, num_classes/4]
            scores = level_scores.permute(0, 1, 3, 4, 2).reshape(B, -1, num_classes)
            boxes = level_boxes.permute(0, 1, 3, 4, 2).reshape(B, -1, 4)
            
            all_boxes.append(boxes)
            all_scores.append(scores)
        
        # Concatenate predictions from all levels
        boxes = torch.cat(all_boxes, dim=1)    # [B, total_anchors, 4]
        scores = torch.cat(all_scores, dim=1)  # [B, total_anchors, num_classes]
        
        # For binary classification, squeeze the class dimension
        if scores.shape[-1] == 1:
            scores = scores.squeeze(-1)
        
        return {
            'boxes': boxes,
            'scores': scores,
            'labels': torch.ones_like(scores),
            'anchors': head_outputs['anchors']
        }

def build_detector(config):
    """Build the complete detection model."""
    # Create backbone
    backbone = ResNetBackbone(
        pretrained=config['pretrained_backbone']
    )
    
    # Create FPN with proper channel list
    in_channels_list = [
        backbone.out_channels['layer1'],  # 256
        backbone.out_channels['layer2'],  # 512
        backbone.out_channels['layer3'],  # 1024
        backbone.out_channels['layer4']   # 2048
    ]
    
    fpn = FeaturePyramidNetwork(
        in_channels_list=in_channels_list,  # Pass the list instead of the dict
        out_channels=config['fpn_out_channels']
    )
    
    # Create detector
    detector = ObjectDetector(
        pretrained_backbone=config['pretrained_backbone'],
        fpn_out_channels=config['fpn_out_channels'],
        num_classes=config['num_classes'],
        num_anchors=config['num_anchors']
    )
    
    return detector

class DetectionModel(nn.Module):
    """Complete detection model."""
    
    def __init__(self, backbone, fpn, num_classes, num_anchors):
        super().__init__()
        self.backbone = backbone
        self.fpn = fpn
        # We'll add detection heads later 