import torch
import torch.nn as nn
import torch.nn.functional as F
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

class FocalLoss(nn.Module):
    """Focal Loss for better handling of class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        # Apply focal scaling
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        return focal_loss.mean()

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
        self.obj_loss = FocalLoss(alpha=0.25, gamma=2.0)
        
        # Get loss weights from config
        self.box_weight = config['training']['debug'].get('box_loss_weight', 5.0)
        self.obj_weight = config['training']['debug'].get('obj_loss_weight', 1.0)
        
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

        # Add self.current_epoch attribute
        self.current_epoch = 0

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

    def _track_gradients(self, named_parameters):
        """Track gradient statistics for debugging"""
        grad_stats = {}
        for name, param in named_parameters:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()
                
                grad_stats[name] = {
                    'norm': grad_norm,
                    'mean': grad_mean,
                    'std': grad_std
                }
                
                # Log significant gradient issues
                if grad_norm > self.config['training']['debug']['grad_clip_value']:
                    self.logger.warning(f"Large gradient in {name}: {grad_norm:.4f}")
                elif grad_norm < 1e-8:
                    self.logger.warning(f"Vanishing gradient in {name}: {grad_norm:.4f}")
        
        return grad_stats

    def _track_activations(self, model_output, batch_idx):
        """Track activation statistics for debugging"""
        with torch.no_grad():
            # Track various activation statistics
            act_stats = {
                'output_mean': model_output.mean().item(),
                'output_std': model_output.std().item(),
                'output_range': (model_output.min().item(), model_output.max().item())
            }
            
            # Check for activation issues
            if abs(act_stats['output_mean']) < 1e-8:
                self.logger.warning(f"Near-zero activations in batch {batch_idx}")
            elif abs(act_stats['output_std']) < 1e-4:
                self.logger.warning(f"Low activation variance in batch {batch_idx}")
        
        return act_stats

    def _visualize_predictions(self, images, predictions, targets, batch_idx, epoch):
        """Visualize model predictions vs ground truth"""
        import matplotlib.pyplot as plt
        import numpy as np
        import torch.nn.functional as F
        
        viz_path = Path(self.config['training']['debug']['visualization']['save_path'])
        viz_path.mkdir(parents=True, exist_ok=True)
        
        conf_threshold = self.config['training']['debug']['visualization']['confidence_threshold']
        max_images = min(self.config['training']['debug']['visualization']['max_images'], len(images))
        
        for i in range(max_images):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            
            # Process image
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = np.clip(img, 0, 1)
            
            # Ground truth
            ax1.imshow(img)
            ax1.set_title('Ground Truth')
            for box in targets[i]:
                x1, y1, x2, y2 = box.detach().cpu().numpy()
                ax1.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                          fill=False, color='green', linewidth=2))
            
            # Predictions
            ax2.imshow(img)
            pred = predictions[i].detach()
            
            # Apply sigmoid to objectness scores
            obj_scores = F.sigmoid(pred[..., 4])
            conf_mask = obj_scores > conf_threshold
            
            # Log prediction stats
            self.logger.info(f"\nPrediction Stats (Image {i}):")
            self.logger.info(f"Max confidence: {obj_scores.max().item():.4f}")
            self.logger.info(f"Mean confidence: {obj_scores.mean().item():.4f}")
            self.logger.info(f"Boxes above threshold: {conf_mask.sum().item()}")
            
            # Get boxes above threshold
            boxes = pred[conf_mask][..., :4].cpu().numpy()
            scores = obj_scores[conf_mask].cpu().numpy()
            
            # Set title with max confidence (handle empty case)
            max_conf = scores.max() if len(scores) > 0 else 0.0
            ax2.set_title(f'Predictions (max conf: {max_conf:.2f})')
            
            # Only try to plot boxes if we have any
            if len(boxes) > 0:
                for box, score in zip(boxes, scores):
                    x1, y1, x2, y2 = box
                    ax2.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                              fill=False, color='red', linewidth=2))
                    ax2.text(x1, y1, f'{score:.2f}', color='red')
            else:
                ax2.text(0.5, 0.5, 'No detections', 
                        horizontalalignment='center',
                        transform=ax2.transAxes)
            
            # Add grid visualization
            ax3.imshow(obj_scores.detach().cpu().numpy(), cmap='hot')
            ax3.set_title('Objectness Heatmap')
            
            plt.savefig(viz_path / f'epoch_{epoch}_batch_{batch_idx}_img_{i}.png')
            plt.close()

    def _process_batch(self, batch):
        """Process a batch and compute loss with debugging"""
        images = batch['image'].to(self.device)
        obj_targets = batch['obj_targets'].to(self.device)
        box_targets = batch['box_targets'].to(self.device)
        
        # Forward pass
        outputs = self.model(images)
        
        # Separate losses with different weights
        box_loss = F.mse_loss(outputs[..., :4], box_targets) * 50.0  # Higher box weight
        obj_loss = F.binary_cross_entropy_with_logits(
            outputs[..., 4], 
            obj_targets,
            pos_weight=torch.tensor([10.0]).to(self.device)  # Weight positive examples more
        )
        
        # Total loss
        loss = box_loss + obj_loss
        
        # Accumulate gradients over multiple batches
        loss = loss / self.config['training'].get('gradient_accumulation_steps', 1)
        loss.backward()
        
        self.current_batch += 1
        
        # Only step optimizer after accumulating gradients
        if self.current_batch % self.config['training'].get('gradient_accumulation_steps', 1) == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # Debug mode tracking
        if self.config['training'].get('debug_mode', False):
            # Track gradients
            if self.config['training']['debug']['track_gradients']:
                grad_stats = self._track_gradients(self.model.named_parameters())
                self.logger.debug(f"Gradient stats: {grad_stats}")
            
            # Track activations
            if self.config['training']['debug']['track_activations']:
                act_stats = self._track_activations(outputs, self.current_batch)
                self.logger.debug(f"Activation stats: {act_stats}")
            
            # Visualize predictions
            if self.current_batch % self.config['dataset']['visualization_frequency'] == 0:
                self._visualize_predictions(
                    images, outputs, batch['boxes'], 
                    self.current_batch, self.current_epoch
                )
        
        # Debug info
        if self.current_batch % 10 == 0:
            self.logger.info(f"\nLoss components:")
            self.logger.info(f"Box loss: {box_loss.item():.4f}")
            self.logger.info(f"Objectness loss: {obj_loss.item():.4f}")
        
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

    def train(self, train_loader, val_loader, early_stopper):
        if len(train_loader) == 0:
            self.logger.error("No training data available!")
            return
        
        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # Training phase
            epoch_loss = self.train_epoch(train_loader)
            self.logger.info(f"Epoch {epoch + 1}: Loss={epoch_loss:.4f}")
            
            # Validation and visualization
            if (epoch + 1) % self.config['logging']['save_frequency'] == 0:
                metrics = self.validate(val_loader)
                self.logger.info(f"Validation F1: {metrics['f1_score']:.4f}")
                
                # Early stopping check
                if early_stopper(metrics['f1_score']):
                    self.logger.info("Early stopping triggered!")
                    break
                
                # Save best model
                if metrics['f1_score'] > self.best_val_f1:
                    self.best_val_f1 = metrics['f1_score']
                    self.save_checkpoint('best_model.pt')
        
        return {
            'f1_score': self.best_val_f1,
            'total_epochs': epoch + 1,
            'final_loss': epoch_loss
        }

    def calculate_iou_loss(self, pred_boxes, target_boxes):
        """Calculate IoU loss between predicted and target boxes"""
        # Convert predictions to corners format if needed
        pred_x1 = pred_boxes[..., 0]
        pred_y1 = pred_boxes[..., 1]
        pred_x2 = pred_boxes[..., 2]
        pred_y2 = pred_boxes[..., 3]
        
        # Get target corners
        target_x1 = target_boxes[..., 0]
        target_y1 = target_boxes[..., 1]
        target_x2 = target_boxes[..., 2]
        target_y2 = target_boxes[..., 3]
        
        # Calculate intersection areas
        x1 = torch.max(pred_x1, target_x1)
        y1 = torch.max(pred_y1, target_y1)
        x2 = torch.min(pred_x2, target_x2)
        y2 = torch.min(pred_y2, target_y2)
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        
        # Calculate union areas
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union = pred_area + target_area - intersection
        
        # Calculate IoU
        iou = intersection / (union + 1e-6)
        
        # Return loss
        return 1 - iou.mean()
