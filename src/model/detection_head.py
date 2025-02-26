import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionHead(nn.Module):
    """
    Detection head for object detection.
    Predicts object presence and bounding box refinements for each anchor.
    """
    
    def __init__(self, 
                 in_channels=256,          # FPN channels
                 num_anchors=6,            # Anchors per location (2 scales × 3 ratios)
                 num_classes=1,            # Binary classification (object vs background)
                 num_convs=4):             # Number of shared convolutions
        super().__init__()
        
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        
        # Shared convolutions
        self.shared_convs = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, 3, padding=1)
            for _ in range(num_convs)
        ])
        
        # Classification branch
        self.cls_head = nn.Conv2d(
            in_channels, 
            num_anchors * num_classes,  # Output for each anchor
            3, 
            padding=1
        )
        
        # Box regression branch
        self.box_head = nn.Conv2d(
            in_channels,
            num_anchors * 4,  # (dx, dy, dw, dh) for each anchor
            3,
            padding=1
        )
        
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
    
    def forward(self, features):
        """
        Forward pass through the detection head.
        
        Args:
            features (dict): Dictionary of FPN feature maps {level: tensor}
            
        Returns:
            dict: Dictionary containing:
                - 'cls_scores': List of classification scores for each level
                - 'bbox_preds': List of box predictions for each level
        """
        cls_scores = []
        bbox_preds = []
        
        # Process each feature level
        for level, feature in features.items():
            cls_score, bbox_pred = self.forward_single(feature)
            
            # Reshape predictions
            B, _, H, W = feature.shape
            cls_score = cls_score.view(B, self.num_anchors, self.num_classes, H, W)
            bbox_pred = bbox_pred.view(B, self.num_anchors, 4, H, W)
            
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
        
        return {
            'cls_scores': cls_scores,
            'bbox_preds': bbox_preds
        } 