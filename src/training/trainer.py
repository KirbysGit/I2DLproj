import torch
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm
import wandb  # For experiment tracking
from utils.visualization import DetectionVisualizer

class Trainer:
    """Handles the training process including validation and checkpointing."""
    
    def __init__(self,
                 model,
                 train_dataset,
                 val_dataset,
                 config,
                 device=None):
        """
        Initialize trainer.
        
        Args:
            model: Detection model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Training configuration
            device: Device to train on
        """
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert string numbers to float where needed
        config['learning_rate'] = float(config['learning_rate'])
        config['weight_decay'] = float(config['weight_decay'])
        config['grad_clip'] = float(config['grad_clip'])
        
        # Model
        self.model = model.to(self.device)
        
        # Datasets
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            collate_fn=train_dataset.collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            collate_fn=val_dataset.collate_fn
        )
        
        # Optimization
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['learning_rate'],
            epochs=config['epochs'],
            steps_per_epoch=len(self.train_loader)
        )
        
        # Setup logging
        self.save_dir = Path(config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb
        if config['use_wandb']:
            wandb.init(project=config['project_name'])
            wandb.config.update(config)
        
        self.visualizer = DetectionVisualizer()
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_cls_loss = 0
        total_box_loss = 0
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc=f'Epoch {epoch}') as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                images = batch['images'].to(self.device)
                boxes = batch['boxes'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Debug info
                if self.config.get('verbose', False) and batch_idx == 0:
                    print(f"\nBatch shapes:")
                    print(f"Images: {images.shape}")
                    print(f"Boxes: {boxes.shape}")
                    print(f"Labels: {labels.shape}")
                
                # Forward pass
                self.optimizer.zero_grad()
                loss_dict = self.model(images, boxes, labels)
                
                # Compute losses
                losses = sum(loss.mean() if isinstance(loss, (list, tuple)) 
                            else loss for loss in loss_dict.values())
                
                # Track individual losses
                cls_loss = loss_dict['cls_loss'].mean() if isinstance(loss_dict['cls_loss'], (list, tuple)) \
                          else loss_dict['cls_loss']
                box_loss = loss_dict['box_loss'].mean() if isinstance(loss_dict['box_loss'], (list, tuple)) \
                          else loss_dict['box_loss']
                
                total_cls_loss += cls_loss.item()
                total_box_loss += box_loss.item()
                total_loss += losses.item()
                
                # Debug info
                if self.config.get('verbose', False) and batch_idx == 0:
                    print("\nLosses:")
                    for k, v in loss_dict.items():
                        print(f"{k}: {v.item():.4f}")
                
                # Backward pass
                losses.backward()
                
                if self.config['grad_clip']:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['grad_clip']
                    )
                
                self.optimizer.step()
                self.scheduler.step()
                
                # Update progress bar with more details
                avg_loss = total_loss / (batch_idx + 1)
                avg_cls_loss = total_cls_loss / (batch_idx + 1)
                avg_box_loss = total_box_loss / (batch_idx + 1)
                
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'cls_loss': f'{avg_cls_loss:.4f}',
                    'box_loss': f'{avg_box_loss:.4f}',
                    'lr': f'{self.scheduler.get_last_lr()[0]:.6f}'
                })
                
                # Log to wandb
                if self.config['use_wandb']:
                    wandb.log({
                        'train_loss': losses.item(),
                        'learning_rate': self.scheduler.get_last_lr()[0]
                    })
        
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def validate(self):
        """Run validation with visualization."""
        self.model.eval()
        total_loss = 0
        total_cls_loss = 0
        total_box_loss = 0
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            # Move data to device
            images = batch['images'].to(self.device)
            boxes = batch['boxes'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass (force loss computation by providing targets)
            loss_dict = self.model(images, boxes, labels)
            
            # Calculate losses properly
            cls_loss = loss_dict['cls_loss'].mean() if isinstance(loss_dict['cls_loss'], (list, tuple)) \
                      else loss_dict['cls_loss']
            box_loss = loss_dict['box_loss'].mean() if isinstance(loss_dict['box_loss'], (list, tuple)) \
                      else loss_dict['box_loss']
            
            batch_loss = cls_loss + box_loss
            
            total_cls_loss += cls_loss.item()
            total_box_loss += box_loss.item()
            total_loss += batch_loss.item()
        
        # Calculate averages
        num_batches = len(self.val_loader)
        avg_loss = total_loss / num_batches
        avg_cls_loss = total_cls_loss / num_batches
        avg_box_loss = total_box_loss / num_batches
        
        print(f"\nValidation losses:")
        print(f"Total: {avg_loss:.4f}")
        print(f"Classification: {avg_cls_loss:.4f}")
        print(f"Box Regression: {avg_box_loss:.4f}")
        
        if self.config['use_wandb']:
            wandb.log({'val_loss': avg_loss})
        
        # Visualize predictions and matches if requested
        if self.config.get('visualize', False):
            with torch.no_grad():
                batch = next(iter(self.val_loader))
                images = batch['images'].to(self.device)
                targets = {
                    'boxes': batch['boxes'].to(self.device),
                    'labels': batch['labels'].to(self.device)
                }
                
                # Get predictions and matches
                outputs = self.model(images)
                matched_labels, matched_boxes = self.model.detection_head.match_anchors_to_targets(
                    outputs['anchors'],
                    targets['boxes'][0],
                    targets['labels'][0]
                )
                
                # Visualize matches
                self.visualizer.visualize_matched_anchors(
                    images[0],
                    outputs['anchors'],
                    targets['boxes'][0],
                    matched_labels,
                    matched_boxes
                )
        
        return avg_loss
    
    def train(self):
        """Main training loop."""
        best_loss = float('inf')
        
        for epoch in range(self.config['epochs']):
            # Training phase
            train_loss = self.train_epoch(epoch)
            
            # Validation phase
            val_loss = self.validate()
            
            # Save checkpoint
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True)
            
            if epoch % self.config['save_freq'] == 0:
                self.save_checkpoint(epoch, val_loss)
            
            print(f'Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}')
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        if is_best:
            path = self.save_dir / 'best_model.pth'
        else:
            path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, path) 