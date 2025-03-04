# model / backbone.py

# -----

# Defines ResNet Backbone for Object Detection.
# Extracts Features at Multiple Scales for Anchor Generation.

# -----

# Imports.
import torch.nn as nn
import torchvision.models as models

# ResNet Backbone Class.
class ResNetBackbone(nn.Module):
    """
    Backbone Network Based on ResNet50 for Feature Extraction.
    Extracts Features at Multiple Scales for Object Detection.
    """
    
    # Initialize.
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Load Pretrained ResNet50 with Updated API.
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None
        resnet = models.resnet50(weights=weights)
        
        # Remove Final Layers.
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        # Layer 1.
        self.layer1 = resnet.layer1  # 256 Channels.

        # Layer 2.
        self.layer2 = resnet.layer2  # 512 Channels.

        # Layer 3.
        self.layer3 = resnet.layer3  # 1024 Channels.

        # Layer 4.
        self.layer4 = resnet.layer4  # 2048 Channels.
        
        # Store Output Channels for FPN.
        self.out_channels = {
            'layer1': 256,
            'layer2': 512,
            'layer3': 1024,
            'layer4': 2048
        }
    
    def forward(self, x):
        """
        Forward Pass Through the Backbone Network.
        
        Args:
            x (torch.Tensor): Input Tensor of Shape (B, C, H, W).
            
        Returns:
            dict: Dictionary of Feature Maps at Different Scales.
        """
        features = {}
        input_shape = x.shape[-2:]
        
        # Extract Features at Each Scale.
        x = self.layer0(x)
        features['layer0'] = x

        # Layer 1.
        x = self.layer1(x)
        features['layer1'] = x

        # Layer 2.
        x = self.layer2(x)
        features['layer2'] = x

        # Layer 3.
        x = self.layer3(x)
        features['layer3'] = x

        # Layer 4.
        x = self.layer4(x)
        features['layer4'] = x
        
        # Print Shapes in Verbose Mode.
        verbose = getattr(self, 'verbose', False)
        if verbose and not hasattr(self, '_printed_shapes'):
            self._printed_shapes = True
            for name, feat in features.items():
                print(f"{name} shape: {feat.shape}, reduction: {input_shape[0]/feat.shape[-2]}x")
        
        return features 