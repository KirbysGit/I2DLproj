import os
import logging
# Suppress warnings
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # Suppress albumentations update warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Suppress other warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import torch
from torch.utils.data import DataLoader
from .dataset_loader import SKU110KDataset
from .model import CNNViTHybrid
from .trainer import Trainer
from .utils import load_config, custom_collate_fn

def main(config=None):
    if config is None:
        config = load_config('config/config.yaml')
    
    # Create and verify datasets
    train_dataset = SKU110KDataset(config, split='train')
    val_dataset = SKU110KDataset(config, split='val')
    
    print(f"\nDataset sizes:")
    print(f"Train: {len(train_dataset)} images")
    print(f"Val: {len(val_dataset)} images")
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Empty datasets!")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['preprocessing']['batch_size'],
        shuffle=True,
        num_workers=1,  # Reduced for testing
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['preprocessing']['batch_size'],
        shuffle=False,
        num_workers=1,
        collate_fn=custom_collate_fn
    )
    
    # Initialize model and trainer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNViTHybrid(config).to(device)
    trainer = Trainer(model, config)
    
    # Run training
    metrics = trainer.train(train_loader, val_loader)
    
    return metrics

if __name__ == '__main__':
    main() 