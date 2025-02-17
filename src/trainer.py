import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import logging
from torch.utils.tensorboard import SummaryWriter
from .evaluate import evaluate
import time
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from .utils import ColorLogger
from colorama import Fore, Style

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training parameters
        self.epochs = config['training']['epochs']
        self.save_dir = Path(config['training']['save_dir'])
        self.save_dir.mkdir(exist_ok=True)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Initialize loss functions
        self.box_loss = nn.SmoothL1Loss()
        self.obj_loss = nn.BCEWithLogitsLoss()
        
        # Setup logging
        self.writer = SummaryWriter(config['logging']['log_dir'])
        self.best_val_f1 = 0.0
        
        # Use single logger instance
        self.logger = ColorLogger()
        
        # Load checkpoint if needed
        if config.get('training', {}).get('resume_from_checkpoint'):
            self._load_checkpoint()
        
        # Add checkpoint tracking
        self.start_epoch = 0
        self.training_history = {
            'train_losses': [],
            'val_metrics': [],
            'learning_rates': [],
            'best_f1': 0.0
        }
        
        # Add visualization setup
        self.viz_dir = Path('results/visualizations')
        self.viz_dir.mkdir(exist_ok=True)
        
        # Initialize metrics history
        self.metrics_history = {
            'train_loss': [],
            'val_f1': [],
            'learning_rates': []
        }

        # Add self.current_batch counter
        self.current_batch = 0

        # Initialize weights tracking
        self.init_weights = {}
        for name, param in self.model.named_parameters():
            self.init_weights[name] = param.data.clone()
        
        # Add _log_weight_updates flag
        self._log_weight_updates = True

    def _load_checkpoint(self):
        """Load model checkpoint only"""
        checkpoint_path = self.save_dir / 'latest_checkpoint.pt'
        if checkpoint_path.exists():
            self.logger.info(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_f1 = checkpoint.get('best_f1', 0.0)

    def plot_metrics(self, epoch):
        """Plot training metrics"""
        plt.figure(figsize=(12, 4))
        
        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(self.metrics_history['train_loss'], label='Training Loss')
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot validation F1 score
        plt.subplot(1, 2, 2)
        plt.plot(self.metrics_history['val_f1'], label='Validation F1')
        plt.title('Validation F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / f'metrics_epoch_{epoch}.png')
        plt.close()

    def train_epoch(self, dataloader):
        self.model.train()
        epoch_losses = {'total': 0, 'box': 0, 'obj': 0}
        
        processed_batches = 0
        total_samples = 0
        
        try:
            for batch in tqdm(dataloader, desc='Training', leave=False):
                # Validate batch data
                if not self._validate_batch(batch):
                    continue
                
                # Process batch
                loss = self._process_batch(batch)
                epoch_losses['total'] += loss
                
                # Verify model is learning
                if processed_batches == 0:  # Check first batch
                    self._verify_model_outputs(batch)
                
                processed_batches += 1
                total_samples += len(batch['image'])
                
        except Exception as e:
            self.logger.error(f"Error in training epoch: {str(e)}")
            raise e
        
        # Average losses
        for k in epoch_losses:
            epoch_losses[k] /= processed_batches
        
        return epoch_losses

    def _validate_batch(self, batch):
        """Validate batch data"""
        required_keys = ['image', 'boxes', 'labels']
        for key in required_keys:
            if key not in batch:
                self.logger.error(f"Missing {key} in batch")
                return False
        return True

    def _process_batch(self, batch):
        """Process a batch and compute loss"""
        # Move batch to device
        images = batch['image'].to(self.device)
        obj_targets = batch['obj_targets'].to(self.device)
        box_targets = batch['box_targets'].to(self.device)
        
        # Forward pass
        outputs = self.model(images)
        
        # Calculate losses
        obj_loss = self.obj_loss(outputs[..., 4], obj_targets)
        box_loss = self.box_loss(outputs[..., :4], box_targets)
        loss = obj_loss + box_loss
        
        # Backward pass
        loss.backward()
        
        # Log weight updates (only if enabled)
        if self._log_weight_updates:
            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        weight_diff = (param.data - self.init_weights[name]).norm().item()
                        if weight_diff > 0.001:  # Only log significant changes
                            self.logger.info(f"Weight update for {name}: {weight_diff:.4f}")
        
        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()

    def _verify_model_outputs(self, batch):
        """Verify model predictions are reasonable"""
        with torch.no_grad():
            images = batch['image'].to(self.device)
            predictions = self.model(images)
            
            # Check prediction ranges
            pred_boxes = predictions[..., :4]
            pred_obj = predictions[..., 4]
            
            self.logger.info("\nModel Output Verification:")
            self.logger.info(f"Prediction shapes - Boxes: {pred_boxes.shape}, Obj: {pred_obj.shape}")
            self.logger.info(f"Box predictions range: [{pred_boxes.min():.4f}, {pred_boxes.max():.4f}]")
            self.logger.info(f"Objectness range: [{pred_obj.min():.4f}, {pred_obj.max():.4f}]")

    def _visualize_batch(self, batch, predictions, epoch, batch_idx):
        """Visualize a batch of predictions"""
        try:
            # Get first image from batch and denormalize
            image = batch['image'][0].cpu().numpy().transpose(1, 2, 0)
            # Denormalize image
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image = std * image + mean
            image = np.clip(image, 0, 1)  # Clip values to valid range
            
            # Get predictions and ground truth boxes
            pred_boxes = predictions[0, ..., :4].detach().cpu().numpy()
            true_boxes = batch['boxes'][0].cpu().numpy()
            
            plt.figure(figsize=(10, 5))
            
            # Plot original image with true boxes
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            for box in true_boxes:
                # Check if box is valid (non-zero)
                if np.any(box):  # Only plot non-zero boxes
                    try:
                        x1, y1, x2, y2 = box[:4]  # Take only first 4 values
                        plt.plot([x1, x2, x2, x1, x1], 
                                [y1, y1, y2, y2, y1], 
                                'g-', linewidth=2, label='Ground Truth')
                    except Exception as e:
                        self.logger.warning(f"Skipping invalid ground truth box: {box}")
            plt.title('Ground Truth')
            
            # Plot predictions
            plt.subplot(1, 2, 2)
            plt.imshow(image)
            for box in pred_boxes:
                # Filter valid predictions
                if np.any(box):  # Only plot non-zero boxes
                    try:
                        x1, y1, x2, y2 = box[:4]  # Take only first 4 values
                        plt.plot([x1, x2, x2, x1, x1], 
                                [y1, y1, y2, y2, y1], 
                                'r-', linewidth=2, label='Prediction')
                    except Exception as e:
                        self.logger.warning(f"Skipping invalid predicted box: {box}")
            plt.title('Predictions')
            
            # Save with error handling
            try:
                save_path = self.viz_dir / f'batch_viz_epoch_{epoch}_batch_{batch_idx}.png'
                plt.savefig(save_path)
                self.logger.info(f"Saved visualization to {save_path}")
            except Exception as e:
                self.logger.error(f"Failed to save visualization: {str(e)}")
            finally:
                plt.close()
            
        except Exception as e:
            self.logger.error(f"Visualization failed: {str(e)}")
            # Don't raise the error - allow training to continue

    def validate(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            val_bar = tqdm(dataloader, 
                          desc='Validating',
                          ncols=100,
                          leave=False,
                          position=1)
            metrics = evaluate(self.model, val_bar, self.device, self.config)
            val_bar.close()
        return metrics

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """Enhanced checkpoint saving"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'training_history': self.training_history,
            'config': self.config
        }
        
        # Save latest checkpoint
        latest_path = self.save_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)
        
        # Save epoch checkpoint
        if (epoch + 1) % self.config['logging']['checkpoint_frequency'] == 0:
            epoch_path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, epoch_path)
        
        # Save best model
        if is_best:
            best_path = self.save_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)

    def train(self, train_loader, val_loader):
        if len(train_loader) == 0:
            self.logger.error("No training data available!")
            return
        
        stage = self.config['training'].get('current_stage', 1)
        start_epoch = self.config['training']['start_epoch']
        end_epoch = self.config['training']['epochs']
        
        self.logger.info(f"Stage {stage}: Processing {len(train_loader)} batches per epoch")
        
        for epoch in range(start_epoch, end_epoch):
            # Training phase
            epoch_loss = self.train_epoch(train_loader)
            
            # Log progress
            self.logger.info(f"Epoch {epoch + 1}: Loss={epoch_loss['total']:.4f}")
            
            # Validation phase
            if (epoch + 1) % self.config['logging']['save_frequency'] == 0:
                metrics = self.validate(val_loader)
                self.logger.info(f"Validation F1: {metrics['f1_score']:.4f}")
        
        return {
            'f1_score': self.best_val_f1,
            'total_epochs': end_epoch,
            'final_loss': epoch_loss['total']
        }
