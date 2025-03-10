# model / fpn.py

# -----

# Implements Feature Pyramid Network (FPN).
# Combines Features from Different Backbone Levels.

# -----

# Imports.
import torch
import torch.nn as nn
import torch.nn.functional as F

# Feature Pyramid Network Class.
class FeaturePyramidNetwork(nn.Module):
    """
    Feature Pyramid Network (FPN) As Described in FPN Paper.
    Creates a Multi-Scale Feature Pyramid from Single Scale Features.
    """

    # Initialize FPN.
    def __init__(self, in_channels_list, out_channels):
        """
        Args:
            in_channels_list (list):    List of input channels for each scale.
            out_channels (int):         Number of output channels for each FPN level.
        """
        super().__init__()

        # Convert To ModuleList Instead of ModuleDict for Indexed Access.
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1)
            for in_channels in in_channels_list
        ])
        
        # Output Convolutions.
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in range(len(in_channels_list))
        ])
        
        # Initialize Weights.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)
    
    # ----------------------------------------------------------------------------
    
    # Upsample & Add.
    def _upsample_add(self, x, y):
        """
        Upsample x and add it to y.
        
        Args:
            x (torch.Tensor): Tensor to be upsampled.
            y (torch.Tensor): Tensor to be added to upsampled x.
        """
        return F.interpolate(x, size=y.shape[-2:], mode='nearest') + y
    
    # ----------------------------------------------------------------------------
    
    # Forward Pass.
    def forward(self, features):
        """
        Forward Pass Through FPN.
        
        Args:
            features (dict): Dictionary of Features from Backbone.
                Expected Keys: 'layer1', 'layer2', 'layer3', 'layer4'.
        """
        # Get Input Features from Each Level (Bottom-Up).
        c2 = features['layer1']  # 1/4
        c3 = features['layer2']  # 1/8
        c4 = features['layer3']  # 1/16
        c5 = features['layer4']  # 1/32
        
        # List of Features from Bottom to Top.
        inputs = [c2, c3, c4, c5]
        
        # Lateral Connections (Bottom-Up).
        laterals = []
        for idx, feature in enumerate(reversed(inputs)):  # Process from top down
            lateral = self.lateral_convs[len(inputs) - 1 - idx](feature)
            laterals.append(lateral)
        
        # Top-Down Pathway.
        fpn_features = [laterals[0]]  # Start with Topmost Feature.
        for idx in range(len(laterals) - 1):
            # Upsample Current Feature.
            upsampled = F.interpolate(
                fpn_features[-1],
                size=laterals[idx + 1].shape[-2:],
                mode='nearest'
            )
            # Add Lateral Connection.
            fpn_features.append(upsampled + laterals[idx + 1])
        
        # Apply Output Convolutions.
        for idx in range(len(fpn_features)):
            fpn_features[idx] = self.output_convs[idx](fpn_features[idx])
        
        # Return as Dictionary with Proper FPN Level Names.
        # P5, P4, P3, P2 (from Top to Bottom).
        return {
            f'p{5-i}': feature 
            for i, feature in enumerate(fpn_features)
        } 