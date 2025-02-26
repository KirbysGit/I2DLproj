import math
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineScheduler(_LRScheduler):
    """
    Learning rate scheduler with warmup and cosine decay.
    """
    
    def __init__(self, 
                 optimizer,
                 warmup_epochs,
                 max_epochs,
                 warmup_start_lr=1e-8,
                 eta_min=1e-8):
        """
        Args:
            optimizer: Optimizer to schedule
            warmup_epochs: Number of epochs for warmup
            max_epochs: Total number of epochs
            warmup_start_lr: Initial learning rate for warmup
            eta_min: Minimum learning rate
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        
        super().__init__(optimizer, -1)
    
    def get_lr(self):
        """Calculate learning rates based on current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            factor = alpha
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            factor = 0.5 * (1 + math.cos(math.pi * progress))
        
        return [
            self.warmup_start_lr + (base_lr - self.warmup_start_lr) * factor
            for base_lr in self.base_lrs
        ] 