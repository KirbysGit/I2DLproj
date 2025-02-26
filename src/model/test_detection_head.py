import torch
from detection_head import DetectionHead

def test_detection_head():
    """Test DetectionHead implementation"""
    print("\nTesting Detection Head...")
    
    # Create dummy FPN features
    batch_size = 2
    fpn_channels = 256
    feature_sizes = [(80, 80), (40, 40), (20, 20), (10, 10)]
    
    features = {
        f'p{i}': torch.randn(batch_size, fpn_channels, h, w)
        for i, (h, w) in enumerate(feature_sizes)
    }
    
    # Initialize detection head
    head = DetectionHead(
        in_channels=fpn_channels,
        num_anchors=6,  # 2 scales × 3 ratios
        num_classes=1   # Binary classification
    )
    head.eval()
    
    # Forward pass
    with torch.no_grad():
        predictions = head(features)
    
    # Verify output
    print("\nPrediction Shapes:")
    for i, (cls_score, bbox_pred) in enumerate(zip(
        predictions['cls_scores'], 
        predictions['bbox_preds']
    )):
        print(f"\nFeature Level {i}:")
        print(f"  Feature map size: {feature_sizes[i]}")
        print(f"  Classification shape: {tuple(cls_score.shape)}")
        print(f"  Box prediction shape: {tuple(bbox_pred.shape)}")
        
        # Verify shapes
        B, A, C, H, W = cls_score.shape
        assert B == batch_size, f"Wrong batch size: {B}"
        assert A == head.num_anchors, f"Wrong number of anchors: {A}"
        assert C == head.num_classes, f"Wrong number of classes: {C}"
        assert (H, W) == feature_sizes[i], f"Wrong feature size: {(H, W)}"
        
        B, A, D, H, W = bbox_pred.shape
        assert D == 4, f"Wrong box dimension: {D}"
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_detection_head() 