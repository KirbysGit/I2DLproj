import torch
from fpn import FeaturePyramidNetwork

def test_fpn():
    """Test FPN implementation"""
    print("\nTesting Feature Pyramid Network...")
    
    # Create dummy feature maps (similar to ResNet output)
    batch_size = 2
    feature_shapes = [
        (batch_size, 256, 160, 160),  # layer1
        (batch_size, 512, 80, 80),    # layer2
        (batch_size, 1024, 40, 40),   # layer3
        (batch_size, 2048, 20, 20),   # layer4
    ]
    
    features = {
        f'layer{i+1}': torch.randn(shape)
        for i, shape in enumerate(feature_shapes)
    }
    
    # Initialize FPN
    in_channels_list = [256, 512, 1024, 2048]  # ResNet feature channels
    out_channels = 256  # Standard FPN output channels
    
    fpn = FeaturePyramidNetwork(in_channels_list, out_channels)
    fpn.eval()
    
    # Forward pass
    with torch.no_grad():
        fpn_features = fpn(features)
    
    # Verify output
    print("\nFPN Output Feature Maps:")
    for name, feat in fpn_features.items():
        print(f"{name}:")
        print(f"  Shape: {tuple(feat.shape)}")
        print(f"  Channels: {feat.shape[1]}")
        print(f"  Spatial size: {feat.shape[-2:]}")
        
        # Verify channel dimension
        assert feat.shape[1] == out_channels, \
            f"Expected {out_channels} channels, got {feat.shape[1]}"
        
        # Verify spatial dimensions match input
        assert feat.shape[-2:] == features[name].shape[-2:], \
            f"Spatial dimensions changed for {name}"
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_fpn() 