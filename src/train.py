import torch
from pathlib import Path
from model.detector import build_detector
from data.dataset import DetectionDataset
from training.trainer import Trainer
from config.train_config import get_training_config
from utils.augmentation import DetectionAugmentation

def main():
    # Get config
    config = get_training_config()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create augmentation
    augmentation = DetectionAugmentation(
        height=config['image_size'],
        width=config['image_size']
    )
    
    # Create datasets
    train_dataset = DetectionDataset(
        data_dir=config['data_dir'],
        csv_file=config['train_csv'],
        transform=augmentation.train_transform
    )
    
    val_dataset = DetectionDataset(
        data_dir=config['data_dir'],
        csv_file=config['val_csv'],
        transform=augmentation.val_transform
    )
    
    # Build model
    model = build_detector(
        backbone=config['backbone'],
        pretrained=config['pretrained'],
        num_classes=config['num_classes']
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        device=device
    )
    
    # Train
    trainer.train()

if __name__ == '__main__':
    main() 