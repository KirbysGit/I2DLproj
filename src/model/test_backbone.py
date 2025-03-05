import torch
from .backbone import ResNetBackbone

def test_backbone():
    """Test the ResNetBackbone implementation"""
    print("\nTesting ResNetBackbone...")
    
    # Test with different input sizes
    test_sizes = [(640, 640), (800, 800), (1024, 1024)]
    
    for height, width in test_sizes:
        print(f"\nTesting with input size: ({height}, {width})")
        x = torch.randn(2, 3, height, width)
        
        backbone = ResNetBackbone(pretrained=True)
        backbone.eval()
        
        with torch.no_grad():
            features = backbone(x)
        
        # Verify feature map sizes
        expected_reductions = {
            'layer0': 4,   # Initial conv + maxpool = 4x reduction
            'layer1': 4,   # Same as layer0
            'layer2': 8,   # Additional 2x reduction
            'layer3': 16,  # Additional 2x reduction
            'layer4': 32   # Additional 2x reduction
        }
        
        for name, feat in features.items():
            reduction = height / feat.shape[-2]
            expected = expected_reductions[name]
            assert abs(reduction - expected) < 1e-5, \
                f"Unexpected reduction in {name}. Got {reduction}x, expected {expected}x"
            
            print(f"{name}:")
            print(f"  Shape: {tuple(feat.shape)}")
            print(f"  Reduction: {reduction}x")
            print(f"  Channels: {feat.shape[1]}")
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_backbone() 