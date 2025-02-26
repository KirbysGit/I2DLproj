import torch
import torch.nn as nn
from backbone import ResNetBackbone
from fpn import FeaturePyramidNetwork
from detection_head import DetectionHead

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
        
    def forward(self, x):
        """
        Forward pass through the complete detector.
        
        Args:
            x (torch.Tensor): Input image tensor [B, C, H, W]
            
        Returns:
            dict: Dictionary containing:
                - backbone_features: Features from backbone
                - fpn_features: Features from FPN
                - cls_scores: Classification scores
                - bbox_preds: Box predictions
        """
        # Get backbone features
        backbone_features = self.backbone(x)
        
        # Remove layer0 as it's not used in FPN
        fpn_input_features = {
            k: v for k, v in backbone_features.items() 
            if k != 'layer0'
        }
        
        # Generate FPN features
        fpn_features = self.fpn(fpn_input_features)
        
        # Get predictions from detection head
        predictions = self.detection_head(fpn_features)
        
        return {
            'backbone_features': backbone_features,
            'fpn_features': fpn_features,
            'cls_scores': predictions['cls_scores'],
            'bbox_preds': predictions['bbox_preds']
        }

def build_detector(config):
    """Build detector model from config."""
    model = ObjectDetector(
        pretrained_backbone=config['model'].get('pretrained_backbone', True),
        fpn_out_channels=config['model'].get('fpn_out_channels', 256),
        num_classes=config['model'].get('num_classes', 1),
        num_anchors=config['model'].get('num_anchors', 6)
    )
    return model 