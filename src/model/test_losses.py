import torch
from losses import FocalLoss, IoULoss, DetectionLoss

def test_focal_loss():
    """Test Focal Loss implementation"""
    print("\nTesting Focal Loss...")
    
    # Create dummy data
    batch_size = 2
    num_boxes = 100
    num_classes = 1
    
    pred = torch.randn(batch_size, num_boxes, num_classes)
    target = torch.randint(0, num_classes, (batch_size, num_boxes))
    
    # Test loss
    focal_loss = FocalLoss()
    loss = focal_loss(pred, target)
    
    print(f"Focal Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"

def test_iou_loss():
    """Test IoU Loss implementation"""
    print("\nTesting IoU Loss...")
    
    # Create dummy boxes
    batch_size = 2
    num_boxes = 100
    
    pred_boxes = torch.rand(batch_size, num_boxes, 4)
    target_boxes = torch.rand(batch_size, num_boxes, 4)
    
    # Ensure valid boxes (x1 < x2, y1 < y2)
    pred_boxes[..., 2:] += pred_boxes[..., :2]
    target_boxes[..., 2:] += target_boxes[..., :2]
    
    # Test loss
    iou_loss = IoULoss()
    loss = iou_loss(pred_boxes, target_boxes)
    
    print(f"IoU Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is Inf"

def test_detection_loss():
    """Test combined Detection Loss"""
    print("\nTesting Detection Loss...")
    
    # Create dummy predictions and targets
    batch_size = 2
    feature_sizes = [(80, 80), (40, 40), (20, 20), (10, 10)]
    num_anchors = 6
    num_classes = 1
    
    predictions = {
        'cls_scores': [
            torch.randn(batch_size, num_anchors, num_classes, h, w)
            for h, w in feature_sizes
        ],
        'bbox_preds': [
            torch.randn(batch_size, num_anchors, 4, h, w)
            for h, w in feature_sizes
        ]
    }
    
    targets = {
        'cls_targets': [
            torch.randint(0, num_classes, (batch_size, num_anchors * h * w))
            for h, w in feature_sizes
        ],
        'box_targets': [
            torch.rand(batch_size, num_anchors * h * w, 4)
            for h, w in feature_sizes
        ]
    }
    
    # Test loss
    detection_loss = DetectionLoss()
    losses = detection_loss(predictions, targets)
    
    print("\nDetection Losses:")
    for k, v in losses.items():
        print(f"{k}: {v.item():.4f}")
        assert not torch.isnan(v), f"{k} is NaN"
        assert not torch.isinf(v), f"{k} is Inf"

def run_all_tests():
    """Run all loss function tests"""
    test_focal_loss()
    test_iou_loss()
    test_detection_loss()
    print("\nAll loss tests passed!")

if __name__ == "__main__":
    run_all_tests() 