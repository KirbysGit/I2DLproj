import torch
import pytest
import sys
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F

# Add src to Python path
src_path = str(Path(__file__).parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from training.optimizer import OptimizerBuilder
from training.scheduler import WarmupCosineScheduler
from training.trainer import Trainer
from model.losses import DetectionLoss

def test_optimizer():
    """Test optimizer building"""
    # Create dummy model
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU()
    )
    
    config = {
        'optimizer_type': 'adam',
        'learning_rate': 1e-4,
        'weight_decay': 1e-4
    }
    
    optimizer = OptimizerBuilder.build(model, config)
    
    # Verify parameter groups
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]['weight_decay'] == config['weight_decay']
    assert optimizer.param_groups[1]['weight_decay'] == 0.0

def test_scheduler():
    """Test learning rate scheduler"""
    model = torch.nn.Linear(10, 10)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=5,
        max_epochs=100
    )
    
    # Test warmup with proper ordering
    for epoch in range(5):
        # Simulate training step
        dummy_loss = torch.randn(1, requires_grad=True)
        dummy_loss.backward()
        optimizer.step()  # First optimizer step
        optimizer.zero_grad()
        
        # Then scheduler step
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        assert lr > scheduler.warmup_start_lr
    
    # Test cosine decay
    final_lr = optimizer.param_groups[0]['lr']
    
    # Simulate one more step
    dummy_loss = torch.randn(1, requires_grad=True)
    dummy_loss.backward()
    optimizer.step()
    scheduler.step()
    
    assert optimizer.param_groups[0]['lr'] < final_lr

def test_trainer(tmp_path):
    """Test trainer functionality"""
    batch_size = 2
    num_classes = 1
    
    # Create proper data structure matching our detector output
    train_loader = [
        {
            'image': torch.randn(batch_size, 3, 640, 640),
            'cls_targets': [
                torch.randint(0, num_classes, (batch_size, 6, h, w), dtype=torch.long)  # Use long instead of float
                for h, w in [(80, 80), (40, 40), (20, 20), (10, 10)]
            ],
            'box_targets': [
                torch.randn(batch_size, 6, 4, h, w).float()
                for h, w in [(80, 80), (40, 40), (20, 20), (10, 10)]
            ]
        }
        for _ in range(5)
    ]
    
    val_loader = [
        {
            'image': torch.randn(batch_size, 3, 640, 640),
            'cls_targets': [
                torch.randint(0, num_classes, (batch_size, 6, h, w), dtype=torch.long)  # Use long instead of float
                for h, w in [(80, 80), (40, 40), (20, 20), (10, 10)]
            ],
            'box_targets': [
                torch.randn(batch_size, 6, 4, h, w).float()
                for h, w in [(80, 80), (40, 40), (20, 20), (10, 10)]
            ]
        }
        for _ in range(2)
    ]
    
    # Create a simple model that outputs the expected format
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_anchors = 6
            self.num_classes = 1
            
            # Adjust output channels to match desired reshape
            self.conv = torch.nn.Conv2d(3, 64, 3, padding=1)
            self.cls_conv = nn.Conv2d(64, self.num_anchors * self.num_classes, 3, padding=1)  # 6 * 1 = 6 channels
            self.box_conv = nn.Conv2d(64, self.num_anchors * 4, 3, padding=1)  # 6 * 4 = 24 channels
            
        def forward(self, x):
            # Initial feature extraction
            feat = self.conv(x)
            
            # Get predictions
            cls_feat = self.cls_conv(feat)  # [B, 6, H, W]
            box_feat = self.box_conv(feat)  # [B, 24, H, W]
            
            batch_size = x.size(0)
            cls_scores = []
            bbox_preds = []
            
            # For each feature level
            for h, w in [(80, 80), (40, 40), (20, 20), (10, 10)]:
                # Resize features to current level
                cls_score = F.interpolate(cls_feat, size=(h, w))
                box_pred = F.interpolate(box_feat, size=(h, w))
                
                # Reshape to expected format
                cls_score = cls_score.reshape(batch_size, self.num_anchors, self.num_classes, h, w).float()
                box_pred = box_pred.reshape(batch_size, self.num_anchors, 4, h, w).float()
                
                cls_scores.append(cls_score)
                bbox_preds.append(box_pred)
            
            return {
                'cls_scores': cls_scores,
                'bbox_preds': bbox_preds
            }
    
    model = DummyModel()
    
    # Use our DetectionLoss instead of MSELoss
    criterion = DetectionLoss(num_classes=num_classes)
    
    config = {
        'optimizer_type': 'adam',
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'epochs': 2,
        'save_freq': 1,
        'save_dir': str(tmp_path),
        'grad_clip': 1.0
    }
    
    optimizer = OptimizerBuilder.build(model, config)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=1,
        max_epochs=config['epochs']
    )
    
    # Explicitly use CPU for testing
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device='cpu'
    )
    
    # Test training loop
    trainer.train()
    
    # Check if checkpoints were saved
    assert (tmp_path / 'checkpoint_epoch_0.pth').exists()
    assert (tmp_path / 'checkpoint_epoch_1.pth').exists()
    assert (tmp_path / 'best_model.pth').exists()

if __name__ == "__main__":
    pytest.main([__file__]) 