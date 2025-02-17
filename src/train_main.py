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

def main():
    print("\n=== Starting Retail Product Detection Training ===\n")
    
    # Load configuration
    config = load_config('config/config.yaml')
    print("Configuration loaded successfully")
    
    # Create datasets
    print("\nInitializing datasets:")
    train_dataset = SKU110KDataset(config, split='train')
    val_dataset = SKU110KDataset(config, split='val')
    
    # Create dataloaders with optimized settings
    print("\nCreating data loaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['preprocessing']['batch_size'],
        shuffle=True,
        num_workers=min(4, os.cpu_count()),  # Optimize worker count
        pin_memory=True,  # Speed up CPU to GPU transfer
        persistent_workers=True,  # Keep workers alive between epochs
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['preprocessing']['batch_size'],
        shuffle=False,
        num_workers=config['preprocessing']['num_workers'],
        collate_fn=custom_collate_fn
    )
    print("Data loaders created successfully")
    
    # Initialize model with GPU support
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f"\nUsing GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("\nUsing CPU - training will be slower")
    
    model = CNNViTHybrid(config).to(device)
    
    # Enable GPU optimization if available
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    
    # Create trainer
    trainer = Trainer(model, config)
    
    # Start training
    print("\nStarting training...\n")
    trainer.train(train_loader, val_loader)

if __name__ == '__main__':
    main() 