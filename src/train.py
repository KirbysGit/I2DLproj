import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .dataset_loader import SKU110KDataset
from .utils import load_config, save_checkpoint
from .model import CNNViTHybrid
from .evaluate import evaluate
import numpy as np
from tqdm import tqdm

def train_step(model, batch, optimizer, device):
    """
    Execute a single training step
    Args:
        model: The neural network model
        batch: Dictionary containing batch data
        optimizer: The optimizer for updating weights
        device: Device to run computations on (CPU/GPU)
    Returns:
        dict: Dictionary of loss values for logging
    """
    # Move batch data to appropriate device
    images = batch['image'].to(device)
    target_boxes = batch['boxes'].to(device)
    target_labels = batch['labels'].to(device)
    
    # Clear gradients from previous step
    optimizer.zero_grad()
    
    # Forward pass through the model
    predictions = model(images)
    
    # Calculate losses:
    # - box_loss: L1 loss for bounding box coordinates
    # - obj_loss: Binary cross-entropy for objectness score
    box_loss = nn.SmoothL1Loss()(predictions[..., :4], target_boxes)
    obj_loss = nn.BCEWithLogitsLoss()(predictions[..., 4], target_labels)
    
    # Combine losses
    loss = box_loss + obj_loss
    
    # Backward pass and optimizer step
    loss.backward()
    optimizer.step()
    
    # Return losses for logging
    return {
        'total_loss': loss.item(),
        'box_loss': box_loss.item(),
        'obj_loss': obj_loss.item()
    }

def train(config_path='config/config.yaml'):
    """
    Main training loop
    Args:
        config_path: Path to configuration YAML file
    """
    # Load configuration and setup
    config = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize datasets
    train_dataset = SKU110KDataset(config, split='train')
    val_dataset = SKU110KDataset(config, split='val')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['preprocessing']['batch_size'],
        shuffle=True,  # Shuffle training data
        num_workers=config['preprocessing']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['preprocessing']['batch_size'],
        shuffle=False,  # Don't shuffle validation data
        num_workers=config['preprocessing']['num_workers']
    )
    
    # Initialize model and move to device
    model = CNNViTHybrid(config).to(device)
    
    # Setup optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Training loop
    best_val_f1 = 0  # Track best validation F1 score
    for epoch in range(config['training']['epochs']):
        model.train()  # Set model to training mode
        epoch_losses = []
        
        # Training epoch
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['epochs']}")):
            losses = train_step(model, batch, optimizer, device)
            epoch_losses.append(losses)
        
        # Calculate and display average epoch losses
        avg_losses = {k: np.mean([loss[k] for loss in epoch_losses]) 
                     for k in epoch_losses[0].keys()}
        
        print(f"\nEpoch {epoch+1} Losses:")
        for k, v in avg_losses.items():
            print(f"{k}: {v:.4f}")
        
        # Validation phase
        if epoch % config['logging']['save_frequency'] == 0:
            model.eval()  # Set model to evaluation mode
            metrics = evaluate(model, val_loader, device, config)
            
            print("\nValidation Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
            
            # Save best model based on F1 score
            if metrics['f1_score'] > best_val_f1:
                best_val_f1 = metrics['f1_score']
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    os.path.join(config['training']['save_dir'], 'best_model.pt')
                )
            
            # Save regular checkpoint
            save_checkpoint(
                model,
                optimizer,
                epoch,
                os.path.join(config['training']['save_dir'], f'checkpoint_{epoch}.pt')
            )

if __name__ == '__main__':
    train()
