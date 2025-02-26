import torch
from tqdm import tqdm
from pathlib import Path
import logging

class Trainer:
    """Handles the training process including validation and checkpointing."""
    
    def __init__(self, 
                 model,
                 train_loader,
                 val_loader,
                 criterion,
                 optimizer,
                 scheduler,
                 config,
                 device=None):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            config: Training configuration
            device: Device to train on (will auto-detect if None)
        """
        # Auto-detect device if not specified
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        print(f"Using device: {device}")
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.save_dir = Path(config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc=f'Epoch {epoch}') as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                images = batch['image'].to(self.device)
                targets = {
                    'cls_targets': [t.to(self.device) for t in batch['cls_targets']],
                    'box_targets': [t.to(self.device) for t in batch['box_targets']]
                }
                
                # Forward pass
                self.optimizer.zero_grad()
                predictions = self.model(images)
                
                # Compute loss
                losses = self.criterion(predictions, targets)
                loss = losses['loss']
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.get('grad_clip'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['grad_clip']
                    )
                
                # Update weights
                self.optimizer.step()
                
                # Update progress bar
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
                })
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        """Run validation."""
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(self.val_loader, desc='Validation'):
            # Move data to device
            images = batch['image'].to(self.device)
            targets = {
                'cls_targets': [t.to(self.device) for t in batch['cls_targets']],
                'box_targets': [t.to(self.device) for t in batch['box_targets']]
            }
            
            # Forward pass
            predictions = self.model(images)
            
            # Compute loss
            losses = self.criterion(predictions, targets)
            total_loss += losses['loss'].item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def train(self):
        """Main training loop."""
        best_loss = float('inf')
        
        for epoch in range(self.config['epochs']):
            # Training phase
            train_loss = self.train_epoch(epoch)
            
            # Validation phase
            val_loss = self.validate()
            
            # Logging
            self.logger.info(
                f'Epoch {epoch}: train_loss={train_loss:.4f}, '
                f'val_loss={val_loss:.4f}, '
                f'lr={self.optimizer.param_groups[0]["lr"]:.6f}'
            )
            
            # Save checkpoint
            if val_loss < best_loss:
                best_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True)
            
            if epoch % self.config['save_freq'] == 0:
                self.save_checkpoint(epoch, val_loss)
            
            # Update learning rate after everything else
            self.scheduler.step()
    
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
        
        # Save regular checkpoint
        save_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, save_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path) 