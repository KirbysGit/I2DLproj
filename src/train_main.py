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
from .model import SimpleDetector
from .trainer import Trainer
from .utils import load_config, custom_collate_fn
from .early_stopping import EarlyStopping

def main(config):
    # Create model
    model = SimpleDetector()  # Using simpler model first
    
    # Create datasets
    train_dataset = SKU110KDataset(config, split='train')
    val_dataset = SKU110KDataset(config, split='val')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn
    )
    
    # Add early stopping
    early_stopper = EarlyStopping(
        patience=5,
        min_delta=0.01,
        mode='max'
    )
    
    trainer = Trainer(model, config)
    metrics = trainer.train(train_loader, val_loader, early_stopper)
    
    return metrics

if __name__ == '__main__':
    main() 