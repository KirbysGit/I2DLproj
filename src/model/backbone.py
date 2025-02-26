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
        
        # Remove the final layers (avgpool and fc)
        self.backbone = nn.ModuleDict({
            # conv1 and maxpool
            'layer0': nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool
            ),
            'layer1': resnet.layer1,  # 256 channels
            'layer2': resnet.layer2,  # 512 channels
            'layer3': resnet.layer3,  # 1024 channels
            'layer4': resnet.layer4   # 2048 channels
        })
        
        self.out_channels = {
            'layer0': 64,
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
        input_shape = x.shape[-2:]  # Store original input shape
        
        # Extract features at each scale
        x = self.backbone['layer0'](x)
        features['layer0'] = x
        print(f"layer0 shape: {x.shape}, reduction: {input_shape[0]/x.shape[-2]}x")
        
        x = self.backbone['layer1'](x)
        features['layer1'] = x
        print(f"layer1 shape: {x.shape}, reduction: {input_shape[0]/x.shape[-2]}x")
        
        x = self.backbone['layer2'](x)
        features['layer2'] = x
        print(f"layer2 shape: {x.shape}, reduction: {input_shape[0]/x.shape[-2]}x")
        
        x = self.backbone['layer3'](x)
        features['layer3'] = x
        print(f"layer3 shape: {x.shape}, reduction: {input_shape[0]/x.shape[-2]}x")
        
        x = self.backbone['layer4'](x)
        features['layer4'] = x
        print(f"layer4 shape: {x.shape}, reduction: {input_shape[0]/x.shape[-2]}x")
        
        return features 