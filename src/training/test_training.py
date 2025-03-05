# Imports
import torch
import pytest
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Import our training modules
from src.training.optimizer import OptimizerBuilder
from src.training.scheduler import WarmupCosineScheduler
from src.training.trainer import Trainer
from src.model.losses import DetectionLoss
from src.utils.metrics import DetectionMetrics
from src.data.dataset import SKU110KDataset
from src.utils.augmentation import DetectionAugmentation
from src.model.detector import ObjectDetector, build_detector  # Import build_detector as well


# ----------------------------
# Test Optimizer Functionality
# ----------------------------
def test_optimizer():
    """Test optimizer building and weight decay handling."""

    # Create Dummy Model.
    model = nn.Sequential(
        nn.Conv2d(3, 64, 3),
        nn.BatchNorm2d(64),
        nn.ReLU()
    )
    
    # Define Config.
    config = {
        'optimizer_type': 'adam',
        'learning_rate': 1e-4,
        'weight_decay': 1e-4
    }

    # Build Optimizer.
    optimizer = OptimizerBuilder.build(model, config)
    
    # Verify Optimizer Properties.
    assert len(optimizer.param_groups) == 2  # Weight decay & no weight decay groups
    assert optimizer.param_groups[0]['weight_decay'] == config['weight_decay']
    assert optimizer.param_groups[1]['weight_decay'] == 0.0  # No decay for biases


# ----------------------------
# Test Learning Rate Scheduler
# ----------------------------
def test_scheduler():
    """Test learning rate scheduler with warmup and cosine decay."""
    
    # Create Dummy Optimizer
    model = nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Define Scheduler
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=5,
        max_epochs=100
    )

    # Check Warmup Phase
    for epoch in range(5):
        # Create dummy loss and compute gradients
        output = model(torch.randn(5, 10))
        loss = output.mean()
        loss.backward()
        
        # Update parameters and learning rate in correct order
        optimizer.step()
        scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        assert lr > scheduler.warmup_start_lr  # Ensure LR is increasing

    # Check Cosine Decay Phase
    prev_lr = optimizer.param_groups[0]['lr']
    for epoch in range(95):
        # Create dummy loss and compute gradients
        output = model(torch.randn(5, 10))
        loss = output.mean()
        loss.backward()
        
        # Update parameters and learning rate in correct order
        optimizer.step()
        scheduler.step()
        
        lr = optimizer.param_groups[0]['lr']
        assert lr <= prev_lr  # Learning rate should decrease
        prev_lr = lr


# ----------------------------
# Dummy Dataset for Training
# ----------------------------
class DummyDataset(torch.utils.data.Dataset):
    """A simple dataset to simulate images, boxes, and labels for detection."""

    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            'images': torch.randn(3, 640, 640),
            'boxes': torch.randn(6, 4),
            'labels': torch.randint(0, 2, (6,))
        }

    @staticmethod
    def collate_fn(batch):
        images = torch.stack([b['images'] for b in batch])

        # Find max boxes in batch
        max_boxes = max([b['boxes'].shape[0] for b in batch])
        batch_size = len(batch)

        # Pad boxes & labels
        padded_boxes = torch.zeros((batch_size, max_boxes, 4))
        padded_labels = torch.zeros((batch_size, max_boxes), dtype=torch.long)

        for i, b in enumerate(batch):
            num_boxes = b['boxes'].shape[0]
            padded_boxes[i, :num_boxes, :] = b['boxes']
            padded_labels[i, :num_boxes] = b['labels']

        return {'images': images, 'boxes': padded_boxes, 'labels': padded_labels}


# ----------------------------
# Dummy Model for Trainer Test
# ----------------------------
class DummyModel(nn.Module):
    """A simple convolutional model that mimics a detection model."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.cls_conv = nn.Conv2d(64, 6 * 1, 3, padding=1)  # 6 anchors * 1 class
        self.box_conv = nn.Conv2d(64, 6 * 4, 3, padding=1)  # 6 anchors * 4 bbox coords

    def forward(self, x, boxes=None, labels=None):
        feat = self.conv(x)
        cls_pred = self.cls_conv(feat)
        box_pred = self.box_conv(feat)

        # Resize predictions to match FPN levels
        cls_scores = []
        bbox_preds = []

        for h, w in [(80, 80), (40, 40), (20, 20), (10, 10)]:
            cls_scores.append(F.interpolate(cls_pred, size=(h, w)).reshape(x.size(0), 6, 1, h, w))
            bbox_preds.append(F.interpolate(box_pred, size=(h, w)).reshape(x.size(0), 6, 4, h, w))

        # Return losses if training, else return predictions
        if boxes is not None and labels is not None:
            cls_loss = torch.sigmoid(cls_pred).mean()  # Example differentiable loss
            box_loss = torch.abs(box_pred).mean()  # Example differentiable loss
            return {
                'cls_loss': cls_loss,
                'box_loss': box_loss
            }

        
        return {'cls_scores': cls_scores, 'bbox_preds': bbox_preds}


# ----------------------------
# Test Full Training Process
# ----------------------------
def test_trainer(tmp_path):
    """Tests the full training pipeline, including loss functions and metrics."""

    batch_size = 2
    num_classes = 1

    # Create dummy datasets
    train_dataset = DummyDataset(size=100)
    val_dataset = DummyDataset(size=20)

    # Define Model
    model = DummyModel()
    criterion = DetectionLoss(num_classes=num_classes)

    # Define Training Config
    config = {
        'optimizer_type': 'adam',
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'epochs': 2,
        'save_freq': 1,
        'save_dir': str(tmp_path),
        'grad_clip': 1.0,
        'batch_size': batch_size,
        'num_workers': 0
    }

    # Build Optimizer & Scheduler
    optimizer = OptimizerBuilder.build(model, config)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=1, max_epochs=config['epochs'])

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device='cpu'
    )

    # Run Training
    trainer.train()

    # Verify checkpoint files were created
    assert (tmp_path / 'checkpoint_epoch_0.pth').exists()
    assert (tmp_path / 'checkpoint_epoch_1.pth').exists()
    assert (tmp_path / 'best_model.pth').exists()


# ----------------------------
# Test Detection Metrics
# ----------------------------
def test_detection_metrics():
    """Ensure anchor matching metrics are computed correctly."""

    # Simulated labels and IoUs
    matched_labels = torch.tensor([1, 1, 0, 0, 1, 0])
    ious = torch.tensor([0.8, 0.5, 0.0, 0.1, 0.7, 0.0])

    metrics = DetectionMetrics.compute_matching_quality(matched_labels, ious)

    assert metrics['mean_iou'] > 0  # Should be positive
    assert metrics['num_positive'] == 3  # Only three positive matches
    assert 0.0 <= metrics['positive_ratio'] <= 1.0  # Must be within range


# ----------------------------
# Test Training Pipeline
# ----------------------------
def test_training_pipeline(tmp_path):
    """Test the complete training pipeline with a small subset of data."""
    print("\nTesting Training Pipeline...")
    
    # Initialize datasets with augmentation
    augmentation = DetectionAugmentation(height=800, width=800)
    
    train_dataset = SKU110KDataset(
        data_dir='datasets/SKU-110K',
        split='train',
        transform=augmentation.train_transform
    )
    
    val_dataset = SKU110KDataset(
        data_dir='datasets/SKU-110K',
        split='val',
        transform=augmentation.val_transform
    )
    
    # Create a small subset for testing
    train_dataset.image_ids = train_dataset.image_ids[:10]  # Use only 10 images
    val_dataset.image_ids = val_dataset.image_ids[:5]      # Use only 5 images
    
    # Define model configuration
    model_config = {
        'pretrained_backbone': True,
        'fpn_out_channels': 256,
        'num_classes': 1,  # Binary classification (object vs background)
        'num_anchors': 6   # 2 aspect ratios * 3 scales
    }
    
    # Initialize model using build_detector
    model = build_detector(model_config)
    
    # Training configuration
    config = {
        'batch_size': 2,
        'num_workers': 0,
        'learning_rate': '1e-4',
        'weight_decay': '1e-4',
        'epochs': 2,
        'save_freq': 1,
        'save_dir': str(tmp_path),
        'grad_clip': '1.0',
        'verbose': True  # Enable verbose mode for debugging
    }
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device='cpu'  # Use CPU for testing
    )
    
    try:
        # Run training
        print("\nStarting test training...")
        trainer.train()
        
        # Verify checkpoint files were created
        assert (tmp_path / 'checkpoint_epoch_0.pth').exists(), "Epoch 0 checkpoint not found"
        assert (tmp_path / 'checkpoint_epoch_1.pth').exists(), "Epoch 1 checkpoint not found"
        assert (tmp_path / 'best_model.pth').exists(), "Best model checkpoint not found"
        
        print("\nTraining completed successfully!")
        print("Checkpoint files created:")
        print(f"- {tmp_path / 'checkpoint_epoch_0.pth'}")
        print(f"- {tmp_path / 'checkpoint_epoch_1.pth'}")
        print(f"- {tmp_path / 'best_model.pth'}")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise
    
    return trainer  # Return trainer for additional testing if needed


# ----------------------------
# Run All Tests
# ----------------------------
if __name__ == "__main__":
    pytest.main([__file__])
