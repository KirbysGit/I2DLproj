import torch
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import logging
from torch.utils.tensorboard import SummaryWriter
from .evaluate import evaluate
import time

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
        
        # Setup logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def train_epoch(self, dataloader):
        self.model.train()
        epoch_losses = {
            'total': 0,
            'box': 0,
            'obj': 0
        }
        
        start_time = time.time()
        total_samples = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Training')):
            batch_start = time.time()
            batch_size = len(batch['image'])
            total_samples += batch_size
            
            images = batch['image'].to(self.device)
            obj_targets = batch['obj_targets'].to(self.device)
            box_targets = batch['box_targets'].to(self.device)
            num_boxes = batch['num_boxes'].to(self.device)
            
            # Forward pass
            predictions = self.model(images)
            pred_boxes = predictions[..., :4]
            pred_obj = predictions[..., 4]
            
            # Calculate losses (only on valid boxes)
            box_loss = 0
            obj_loss = 0
            
            for i in range(batch_size):
                valid_boxes = num_boxes[i]
                box_loss += self.box_loss(
                    pred_boxes[i, :valid_boxes],
                    box_targets[i, :valid_boxes]
                )
                obj_loss += self.obj_loss(
                    pred_obj[i],
                    obj_targets[i]
                )
            
            # Average losses over batch
            box_loss /= batch_size
            obj_loss /= batch_size
            loss = box_loss + obj_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update losses
            epoch_losses['total'] += loss.item()
            epoch_losses['box'] += box_loss.item()
            epoch_losses['obj'] += obj_loss.item()
            
            # Log batch statistics
            batch_time = time.time() - batch_start
            images_per_second = batch_size / batch_time
            
            if batch_idx % 10 == 0:  # Log every 10 batches
                self.logger.info(
                    f"Batch {batch_idx}: {images_per_second:.2f} img/s, "
                    f"Loss: {loss.item():.4f}"
                )
        
        epoch_time = time.time() - start_time
        avg_time_per_sample = epoch_time / total_samples
        
        self.logger.info(
            f"\nEpoch Statistics:\n"
            f"Total time: {epoch_time:.2f}s\n"
            f"Samples/second: {total_samples/epoch_time:.2f}\n"
            f"Time/sample: {avg_time_per_sample*1000:.2f}ms"
        )
        
        # Average losses over epochs
        num_batches = len(dataloader)
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
            
        return epoch_losses

    def validate(self, dataloader):
        self.model.eval()
        val_metrics = None
        
        with torch.no_grad():
            val_metrics = evaluate(self.model, dataloader, self.device, self.config)
        
        return val_metrics

    def save_checkpoint(self, epoch, metrics, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if needed
        if is_best:
            best_path = self.save_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)

    def train(self, train_loader, val_loader):
        self.logger.info(f"Starting training on device: {self.device}")
        
        for epoch in range(self.epochs):
            self.logger.info(f"\nEpoch {epoch+1}/{self.epochs}")
            
            # Training phase
            train_losses = self.train_epoch(train_loader)
            
            # Log training metrics
            self.writer.add_scalar('Loss/train_total', train_losses['total'], epoch)
            self.writer.add_scalar('Loss/train_box', train_losses['box'], epoch)
            self.writer.add_scalar('Loss/train_obj', train_losses['obj'], epoch)
            
            # Validation phase
            if epoch % self.config['logging']['save_frequency'] == 0:
                val_metrics = self.validate(val_loader)
                
                # Log validation metrics
                for metric, value in val_metrics.items():
                    self.writer.add_scalar(f'Metrics/{metric}', value, epoch)
                
                # Save checkpoint if best model
                if val_metrics['f1_score'] > self.best_val_f1:
                    self.best_val_f1 = val_metrics['f1_score']
                    self.save_checkpoint(epoch, val_metrics, is_best=True)
                    self.logger.info(f"New best model saved! F1: {self.best_val_f1:.4f}")
                
                # Print metrics
                self.logger.info("Validation Metrics:")
                for metric, value in val_metrics.items():
                    self.logger.info(f"{metric}: {value:.4f}")
            
            # Save regular checkpoint
            if (epoch + 1) % self.config['logging']['save_frequency'] == 0:
                self.save_checkpoint(epoch, val_metrics)
                
        self.writer.close()
        self.logger.info("Training completed!")
