import torch
import torch.nn as nn
import torch.nn.functional as F

class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network (FPN) as described in FPN paper.
    Creates a multi-scale feature pyramid from single scale features.
    """
    
    def __init__(self, in_channels_list, out_channels):
        """
        Initialize FPN.
        
        Args:
            in_channels_list (list): List of input channels for each scale
            out_channels (int): Number of output channels for each FPN level
        """
        super().__init__()
        
        # Convert to ModuleList instead of ModuleDict for indexed access
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1)
            for in_channels in in_channels_list
        ])
        
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in range(len(in_channels_list))
        ])
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
    
    def _upsample_add(self, x, y):
        """
        Upsample x and add it to y.
        
        Args:
            x (torch.Tensor): Tensor to be upsampled
            y (torch.Tensor): Tensor to be added to upsampled x
        """
        return F.interpolate(x, size=y.shape[-2:], mode='nearest') + y
    
    def forward(self, features):
        """
        Forward pass through FPN.
        
        Args:
            features (dict): Dictionary of features from backbone
                Expected keys: 'layer1', 'layer2', 'layer3', 'layer4'
        """
        # Get input features from each level (bottom-up)
        c2 = features['layer1']  # 1/4
        c3 = features['layer2']  # 1/8
        c4 = features['layer3']  # 1/16
        c5 = features['layer4']  # 1/32
        
        # List of features from bottom to top
        inputs = [c2, c3, c4, c5]
        
        # Lateral connections (bottom-up)
        laterals = []
        for idx, feature in enumerate(reversed(inputs)):  # Process from top down
            lateral = self.lateral_convs[len(inputs) - 1 - idx](feature)
            laterals.append(lateral)
        
        # Top-down pathway
        fpn_features = [laterals[0]]  # Start with topmost feature
        for idx in range(len(laterals) - 1):
            # Upsample current feature
            upsampled = F.interpolate(
                fpn_features[-1],
                size=laterals[idx + 1].shape[-2:],
                mode='nearest'
            )
            # Add lateral connection
            fpn_features.append(upsampled + laterals[idx + 1])
        
        # Apply output convolutions
        for idx in range(len(fpn_features)):
            fpn_features[idx] = self.output_convs[idx](fpn_features[idx])
        
        # Return as dictionary with proper FPN level names
        # P5, P4, P3, P2 (from top to bottom)
        return {
            f'p{5-i}': feature 
            for i, feature in enumerate(fpn_features)
        } 