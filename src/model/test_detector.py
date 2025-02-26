import torch
import yaml
from pathlib import Path
from detector import build_detector

def verify_feature_maps(features, input_size):
    """Verify feature maps are suitable for object detection"""
    min_size = min(input_size) // 32  # Maximum reduction should be 32x
    
    for name, feat_dict in features.items():
        if isinstance(feat_dict, dict):
            for level, feat in feat_dict.items():
                feat_size = min(feat.shape[-2:])
                assert feat_size >= min_size, \
                    f"Feature map {level} too small: {feat_size} < {min_size}"
                print(f"{level}:")
                print(f"  Shape: {tuple(feat.shape)}")
                print(f"  Min size: {feat_size}")

def test_detector():
    """Test the complete ObjectDetector implementation"""
    print("\nTesting Complete Object Detector...")
    
    # Load config
    config_path = Path('config/config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Test with different input sizes
    test_sizes = [(640, 640), (800, 800)]
    
    for height, width in test_sizes:
        print(f"\nTesting with input size: ({height}, {width})")
        x = torch.randn(2, 3, height, width)
        
        model = build_detector(config)
        model.eval()
        
        with torch.no_grad():
            output = model(x)
        
        # Verify backbone features
        print("\nBackbone Features:")
        verify_feature_maps({'backbone': output['backbone_features']}, (height, width))
        
        # Verify FPN features
        print("\nFPN Features:")
        verify_feature_maps({'fpn': output['fpn_features']}, (height, width))
        
        # Verify detection head outputs
        print("\nDetection Head Outputs:")
        for i, (cls_score, bbox_pred) in enumerate(zip(
            output['cls_scores'],
            output['bbox_preds']
        )):
            print(f"\nFeature Level {i}:")
            print(f"  Classification shape: {tuple(cls_score.shape)}")
            print(f"  Box prediction shape: {tuple(bbox_pred.shape)}")
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_detector() 