# training/scheduler.py

# -----

# Modifies Learning Rate Dynamically During Training.
# Consists of Warmup Phase & Cosine Decay Phase.

# -----

# Imports.
import math
from torch.optim.lr_scheduler import _LRScheduler
import torch

# Warmup Cosine Scheduler Class.
class WarmupCosineScheduler(_LRScheduler):
    """
    Learning Rate Scheduler w/ Warmup & Cosine Decay.
    """
    
    def __init__(self, 
                 optimizer,
                 warmup_epochs,
                 max_epochs,
                 warmup_start_lr=1e-8,
                 eta_min=1e-8):
        """
        Args:
            optimizer:          Optimizer to schedule.
            warmup_epochs:      Number of epochs for warmup.
            max_epochs:         Total number of epochs.
            warmup_start_lr:    Initial learning rate for warmup.
            eta_min:            Minimum learning rate.
        """

        # Initialize Scheduler.
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        self.warmup_progress_shown = False
        
        super().__init__(optimizer, -1)
    
    # Get Learning Rate.
    def get_lr(self):
        """Calculate Learning Rates Based on Current Epoch."""
        
        if self.last_epoch < self.warmup_epochs:
            # Linear Warmup.
            alpha = self.last_epoch / self.warmup_epochs
            factor = alpha
            
            # Show warmup progress
            progress = int(alpha * 20)  # 20 steps progress bar
            if not self.warmup_progress_shown:
                print("\nWarmup Phase:")
                self.warmup_progress_shown = True
            print(f"\r[{'=' * progress}{' ' * (20-progress)}] {int(alpha*100)}% ", end='')
            
            if self.last_epoch == self.warmup_epochs - 1:
                print("\nWarmup completed!")
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            factor = 0.5 * (1 + math.cos(math.pi * progress))
        
        # Return Learning Rates.
        return [
            self.warmup_start_lr + (base_lr - self.warmup_start_lr) * factor
            for base_lr in self.base_lrs
        ] 

def build_scheduler(optimizer, config):
    """Build learning rate scheduler.
    
    Args:
        optimizer: The optimizer to schedule
        config: Dictionary containing scheduler configuration
            - scheduler_type: Type of scheduler (default: 'onecycle')
            - epochs: Number of epochs
            - learning_rate: Base learning rate
            - steps_per_epoch: Number of steps per epoch
            - warmup_epochs: Number of warmup epochs (default: 1)
    
    Returns:
        torch.optim.lr_scheduler._LRScheduler: The configured scheduler
    """
    scheduler_type = config.get('scheduler_type', 'onecycle').lower()
    epochs = int(config.get('epochs', 10))
    warmup_epochs = int(config.get('warmup_epochs', 1))
    
    if scheduler_type == 'onecycle':
        # OneCycleLR is good for training from scratch
        steps_per_epoch = config.get('steps_per_epoch', 100)
        max_lr = float(config.get('learning_rate', 1e-3))
        
        # Calculate pct_start based on warmup_epochs
        pct_start = warmup_epochs / epochs
        
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,  # Use warmup_epochs to determine warmup percentage
            div_factor=25,  # Initial learning rate is max_lr/25
            final_div_factor=1e4  # Final learning rate is max_lr/10000
        )
    
    elif scheduler_type == 'cosine':
        # Use our custom WarmupCosineScheduler for warmup + cosine decay
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=epochs,
            warmup_start_lr=float(config.get('learning_rate', 1e-3)) / 25,  # Similar to OneCycleLR
            eta_min=float(config.get('learning_rate', 1e-3)) / 1e4  # Similar to OneCycleLR
        )
    
    elif scheduler_type == 'step':
        # Step decay for more controlled decay
        step_size = int(config.get('step_size', epochs // 3))
        gamma = float(config.get('gamma', 0.1))
        
        # Wrap StepLR with linear warmup
        base_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
        
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=epochs,
            warmup_start_lr=float(config.get('learning_rate', 1e-3)) / 25
        )
    
    elif scheduler_type == 'plateau':
        # ReduceLROnPlateau for adaptive learning rate
        patience = int(config.get('patience', 5))
        factor = float(config.get('factor', 0.1))
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=patience,
            factor=factor,
            verbose=True
        )
    
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
    
    return scheduler 