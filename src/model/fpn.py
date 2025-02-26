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
        
        # Lateral connections (1x1 convolutions)
        self.lateral_convs = nn.ModuleDict({
            f'lateral_{i}': nn.Conv2d(in_channels, out_channels, 1)
            for i, in_channels in enumerate(in_channels_list)
        })
        
        # Output connections (3x3 convolutions)
        self.output_convs = nn.ModuleDict({
            f'output_{i}': nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for i in range(len(in_channels_list))
        })
        
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
            features (dict): Dictionary of feature maps at different scales
                           from backbone (bottom-up pathway)
        
        Returns:
            dict: Dictionary of FPN feature maps
        """
        # Get input features from backbone
        names = sorted(features.keys())  # Sort to ensure correct order
        inputs = [features[name] for name in names]
        
        # Build top-down pathway
        laterals = []
        for i, feature in enumerate(inputs):
            lateral = self.lateral_convs[f'lateral_{i}'](feature)
            laterals.append(lateral)
        
        # Top-down pathway with lateral connections
        fpn_features = {}
        prev_features = laterals[-1]
        fpn_features[names[-1]] = self.output_convs[f'output_{len(inputs)-1}'](prev_features)
        
        for i in range(len(inputs)-2, -1, -1):  # Start from second to last
            prev_features = self._upsample_add(prev_features, laterals[i])
            fpn_features[names[i]] = self.output_convs[f'output_{i}'](prev_features)
        
        return fpn_features 