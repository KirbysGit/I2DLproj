import torch
import torch.nn as nn
import torchvision.models as models

class ResNetBackbone(nn.Module):
    """
    Backbone network based on ResNet50 for feature extraction.
    Extracts features at multiple scales for object detection.
    """
    
    def __init__(self, pretrained=True):
        super().__init__()
        # Load pretrained ResNet50 with updated API
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None
        resnet = models.resnet50(weights=weights)
        
        # Remove final layers
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels
        
        # Store output channels for FPN
        self.out_channels = {
            'layer1': 256,
            'layer2': 512,
            'layer3': 1024,
            'layer4': 2048
        }
    
    def forward(self, x):
        """
        Forward pass through the backbone network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            dict: Dictionary of feature maps at different scales
        """
        features = {}
        input_shape = x.shape[-2:]
        
        # Extract features at each scale
        x = self.layer0(x)
        features['layer0'] = x
        
        x = self.layer1(x)
        features['layer1'] = x
        
        x = self.layer2(x)
        features['layer2'] = x
        
        x = self.layer3(x)
        features['layer3'] = x
        
        x = self.layer4(x)
        features['layer4'] = x
        
        # Print shapes in verbose mode
        verbose = getattr(self, 'verbose', False)
        if verbose and not hasattr(self, '_printed_shapes'):
            self._printed_shapes = True
            for name, feat in features.items():
                print(f"{name} shape: {feat.shape}, reduction: {input_shape[0]/feat.shape[-2]}x")
        
        return features 