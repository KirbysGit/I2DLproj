import torch
from pathlib import Path
import yaml
from model.detector import build_detector
from data.dataset import SKU110KDataset
from data.augmentation import DataAugmentation
from training.trainer import Trainer

# Move class to top level
class LimitedDataset(SKU110KDataset):
    """Dataset wrapper that limits the number of samples for testing."""
    def __init__(self, *args, max_samples=None, **kwargs):
        super().__init__(*args, **kwargs)
        if max_samples:
            self.image_ids = self.image_ids[:max_samples]

def test_training():
    """Test the complete training pipeline."""
    print("\nTesting Training Pipeline...")
    
    # Load test config
    config_path = Path('config/test_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create datasets with size limit for testing
    transform = DataAugmentation(
        min_size=800,
        max_size=1333,
        flip_prob=0.5
    )
    
    train_dataset = LimitedDataset(
        data_dir='datasets/SKU-110K',
        split='train',
        transform=transform,
        max_samples=config['training']['max_samples']
    )
    
    val_dataset = LimitedDataset(
        data_dir='datasets/SKU-110K',
        split='val',
        transform=None,
        max_samples=config['training']['max_samples']//4
    )
    
    print(f"\nTest Dataset sizes:")
    print(f"Training: {len(train_dataset)} (limited from full size)")
    print(f"Validation: {len(val_dataset)} (limited from full size)")
    
    # Build model
    model = build_detector(config['model'])
    print("\nModel created successfully")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config['training']
    )
    print("\nTrainer initialized")
    
    # Test training
    print("\nRunning test training...")
    try:
        for epoch in range(config['training']['epochs']):
            print(f"\nEpoch {epoch}:")
            train_loss = trainer.train_epoch(epoch)
            print(f"Training loss: {train_loss:.4f}")
            
            val_loss = trainer.validate()
            print(f"Validation loss: {val_loss:.4f}")
        
        print("\nAll training tests passed!")
        
    except Exception as e:
        print(f"\nTraining test failed: {str(e)}")
        raise e

if __name__ == "__main__":
    test_training() 