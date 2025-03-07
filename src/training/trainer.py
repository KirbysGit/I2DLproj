import torch
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm
from src.utils.visualization import DetectionVisualizer
from src.utils.box_ops import box_iou
from src.utils.metrics import DetectionMetrics
import numpy as np

from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler


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
            collate_fn=self.collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            collate_fn=self.collate_fn
        )
        
        # Optimization
        self.optimizer = build_optimizer(model, config)
        self.scheduler = build_scheduler(self.optimizer, config)
        
        # Setup logging
        self.save_dir = Path(config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.visualizer = DetectionVisualizer()
        
        # Get metrics tracker from config
        self.metrics_tracker = config.get("metrics_tracker", None)
        self.run_dir = Path(config.get("run_dir", "training_runs/default"))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Debug options
        self.debug_options = config.get("debug_options", {})
    
    def train_epoch(self, epoch):
        """Train for one epoch with better formatting and error handling."""
        self.model.train()
        total_loss = 0
        total_map = 0
        total_f1 = 0
        num_batches = 0
        
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{self.config['epochs']}")
        print(f"{'='*80}")
        
        with tqdm(self.train_loader, desc=f'Training') as pbar:
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Move data to device
                    images = batch['images'].to(self.device)  # This matches the collate_fn output
                    boxes = [b.to(self.device) for b in batch['boxes']]
                    labels = [l.to(self.device) for l in batch['labels']]
                    
                    # Debug info
                    if self.config.get('verbose', False) and batch_idx == 0:
                        print(f"\nBatch shapes:")
                        print(f"Images: {images.shape}")
                        print(f"Boxes: {[b.shape for b in boxes]}")
                        print(f"Labels: {[l.shape for l in labels]}")
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    
                    try:
                        predictions = self.model(images)
                    except RuntimeError as e:
                        print(f"\nError in forward pass: {str(e)}")
                        if self.config.get('verbose', False):
                            print(f"Input shapes that caused error:")
                            print(f"Images: {images.shape}")
                            print(f"Boxes: {[b.shape for b in boxes]}")
                            print(f"Labels: {[l.shape for l in labels]}")
                        raise
                    
                    # Calculate loss
                    loss_dict = self.compute_loss(predictions, boxes, labels)
                    total_loss += loss_dict['total_loss'].item()
                    
                    # Calculate metrics
                    batch_map = self.compute_map(predictions, boxes)
                    batch_f1 = self.compute_f1(predictions, boxes)
                    total_map += batch_map
                    total_f1 += batch_f1
                    
                    # Backward pass
                    loss_dict['total_loss'].backward()
                    
                    # Gradient clipping
                    if self.config['grad_clip']:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config['grad_clip']
                        )
                    
                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    
                    # Update progress bar
                    num_batches += 1
                    avg_loss = total_loss / num_batches
                    avg_map = total_map / num_batches
                    avg_f1 = total_f1 / num_batches
                    
                    pbar.set_postfix({
                        'Loss': f'{avg_loss:.4f}',
                        'mAP': f'{avg_map:.4f}',
                        'F1': f'{avg_f1:.4f}',
                        'LR': f'{self.scheduler.get_last_lr()[0]:.6f}'
                    })
                    
                except Exception as e:
                    print(f"\nError processing batch {batch_idx}: {str(e)}")
                    if self.config.get('verbose', False):
                        print("Batch contents:")
                        for k, v in batch.items():
                            if isinstance(v, torch.Tensor):
                                print(f"{k}: shape={v.shape}, dtype={v.dtype}")
                            else:
                                print(f"{k}: {type(v)}")
                    raise
        
        # Calculate epoch metrics
        epoch_loss = total_loss / num_batches
        epoch_map = total_map / num_batches
        epoch_f1 = total_f1 / num_batches
        
        # Return metrics dictionary
        metrics = {
            'loss': epoch_loss,
            'mAP': epoch_map,
            'f1': epoch_f1
        }
        
        return metrics
    
    @torch.no_grad()
    def validate(self):
        """Validation with detailed metrics."""
        self.model.eval()
        total_loss = 0
        total_map = 0
        total_f1 = 0
        num_batches = 0
        
        print(f"\n{'='*80}")
        print("Validation")
        print(f"{'='*80}")
        
        all_metrics = []
        
        for batch in tqdm(self.val_loader, desc='Validating'):
            # Move data to device
            images = batch['images'].to(self.device)  # This matches the collate_fn output
            boxes = [b.to(self.device) for b in batch['boxes']]
            labels = [l.to(self.device) for l in batch['labels']]
            
            # Forward pass
            predictions = self.model(images)
            
            # Calculate loss
            loss_dict = self.compute_loss(predictions, boxes, labels)
            total_loss += loss_dict['total_loss'].item()
            
            # Calculate metrics
            batch_map = self.compute_map(predictions, boxes)
            batch_f1 = self.compute_f1(predictions, boxes)
            total_map += batch_map
            total_f1 += batch_f1
            
            num_batches += 1
        
        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_map = total_map / num_batches
        avg_f1 = total_f1 / num_batches
        
        # Return metrics dictionary
        metrics = {
            'loss': avg_loss,
            'mAP': avg_map,
            'f1': avg_f1
        }
        
        return metrics
    
    def train(self):
        """Main training loop."""
        best_val_map = 0
        
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
            
            # Training phase
            train_metrics = self.train_epoch(epoch)
            print(f"\nTraining - Loss: {train_metrics['loss']:.4f}, mAP: {train_metrics['mAP']:.4f}, F1: {train_metrics['f1']:.4f}")
            
            # Validation phase
            val_metrics = self.validate()
            print(f"Validation - Loss: {val_metrics['loss']:.4f}")
            
            # Track metrics
            if self.metrics_tracker is not None:
                self.metrics_tracker.update(epoch, train_metrics, val_metrics)
            
            # Save checkpoint
            if train_metrics['mAP'] > best_val_map:
                best_val_map = train_metrics['mAP']
                self.save_checkpoint(epoch, train_metrics['mAP'], is_best=True)
            
            if epoch % self.config['save_freq'] == 0:
                self.save_checkpoint(epoch, train_metrics['mAP'])
            
            print(f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, "
                f"mAP={train_metrics['mAP']:.4f}, F1={train_metrics['f1']:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}")

    
    def save_checkpoint(self, epoch, val_map, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': val_map,
            'config': self.config
        }
        
        if is_best:
            path = self.save_dir / 'best_model.pth'
        else:
            path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, path)
        
        # Save latest checkpoint
        latest_path = self.run_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint if this is the best validation mAP
        if val_map == self.metrics_tracker.val_maps[-1]:
            best_path = self.run_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for the data loader."""
        images = torch.stack([item['image'] for item in batch])
        boxes = [item['boxes'] for item in batch]
        labels = [item['labels'] for item in batch]
        
        return {
            'images': images,
            'boxes': boxes,
            'labels': labels
        }
    
    def compute_loss(self, predictions, gt_boxes, gt_labels):
        """Compute the total loss for training.
        
        Args:
            predictions (dict): Model predictions containing boxes, scores, etc.
            gt_boxes (list): List of ground truth boxes tensors
            gt_labels (list): List of ground truth label tensors
            
        Returns:
            dict: Dictionary containing the total loss and individual loss components
        """
        # Initialize losses
        cls_loss = 0
        box_loss = 0
        
        # Calculate classification loss (focal loss)
        pred_scores = predictions['scores']
        pred_boxes = predictions['boxes']
        
        # Handle case where predictions are not batched
        if len(pred_boxes.shape) == 2:  # [N, 4] -> [1, N, 4]
            pred_boxes = pred_boxes.unsqueeze(0)
        if len(pred_scores.shape) == 2:  # [N, C] -> [1, N, C]
            pred_scores = pred_scores.unsqueeze(0)
            
        for i in range(len(gt_boxes)):
            # Get IoU between predictions and ground truth
            ious = box_iou(pred_boxes[min(i, len(pred_boxes)-1)], gt_boxes[i])
            
            # Assign positive/negative samples based on IoU
            max_ious, gt_indices = ious.max(dim=1)
            pos_mask = max_ious >= 0.5
            neg_mask = max_ious < 0.4
            
            # Get target labels - ensure float type for BCE loss
            target_labels = torch.zeros_like(pred_scores[min(i, len(pred_scores)-1)], dtype=torch.float32)
            if pos_mask.any():
                # Convert labels to float and ensure proper shape
                matched_labels = gt_labels[i][gt_indices[pos_mask]].float()
                # Handle multi-class case
                if len(matched_labels.shape) == 1:
                    matched_labels = matched_labels.unsqueeze(-1)
                target_labels[pos_mask] = matched_labels
            
            # Calculate focal loss
            alpha = 0.25
            gamma = 2.0
            
            p = torch.sigmoid(pred_scores[min(i, len(pred_scores)-1)])
            
            # Use F.binary_cross_entropy_with_logits instead
            ce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                pred_scores[min(i, len(pred_scores)-1)], 
                target_labels,
                reduction='none'
            )
            
            p_t = p * target_labels + (1 - p) * (1 - target_labels)
            focal_weight = (1 - p_t) ** gamma
            
            if alpha >= 0:
                alpha_t = alpha * target_labels + (1 - alpha) * (1 - target_labels)
                focal_weight = alpha_t * focal_weight
            
            cls_loss += (focal_weight * ce_loss).mean()
        
        # Calculate box regression loss (smooth L1)
        for i in range(len(gt_boxes)):
            # Get IoU between predictions and ground truth
            ious = box_iou(pred_boxes[min(i, len(pred_boxes)-1)], gt_boxes[i])
            
            # Assign positive samples based on IoU
            max_ious, gt_indices = ious.max(dim=1)
            pos_mask = max_ious >= 0.5
            
            if pos_mask.sum() > 0:
                # Get matched ground truth boxes
                matched_gt_boxes = gt_boxes[i][gt_indices[pos_mask]]
                pred_boxes_i = pred_boxes[min(i, len(pred_boxes)-1)][pos_mask]
                
                # Calculate smooth L1 loss
                box_loss += torch.nn.functional.smooth_l1_loss(
                    pred_boxes_i,
                    matched_gt_boxes,
                    reduction='mean',
                    beta=0.1
                )
        
        # Normalize losses by batch size
        cls_loss = cls_loss / len(gt_boxes)
        box_loss = box_loss / len(gt_boxes)
        
        # Combine losses with weighting
        total_loss = cls_loss + box_loss
        
        return {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'box_loss': box_loss
        }
    
    def compute_map(self, predictions, gt_boxes, iou_threshold=0.5):
        """Compute mean Average Precision."""
        # Check if predictions or ground truth are empty using proper tensor operations
        if len(predictions['boxes']) == 0 or len(gt_boxes) == 0 or predictions['boxes'][0].numel() == 0 or gt_boxes[0].numel() == 0:
            return 0.0
        
        # Get predictions
        pred_boxes = predictions['boxes']
        pred_scores = predictions['scores']
        
        # Calculate IoU between predictions and ground truth
        ious = box_iou(pred_boxes[0], gt_boxes[0])
        
        # For each prediction, get the best matching ground truth
        max_ious, _ = ious.max(dim=1)
        
        # Calculate precision at different recall points
        sorted_indices = torch.argsort(pred_scores[0], descending=True)
        precisions = []
        recalls = []
        
        num_gt = len(gt_boxes[0])
        true_positives = 0
        
        for idx in sorted_indices:
            if max_ious[idx] >= iou_threshold:
                true_positives += 1
            
            precision = true_positives / (len(precisions) + 1)
            recall = true_positives / num_gt if num_gt > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Calculate AP using 11-point interpolation
        ap = 0
        for recall_threshold in np.arange(0, 1.1, 0.1):
            max_precision = 0
            for precision, recall in zip(precisions, recalls):
                if recall >= recall_threshold:
                    max_precision = max(max_precision, precision)
            ap += max_precision / 11
        
        return ap
    
    def compute_f1(self, predictions, gt_boxes, iou_threshold=0.5):
        """Compute F1 score."""
        # Check if predictions or ground truth are empty using proper tensor operations
        if len(predictions['boxes']) == 0 or len(gt_boxes) == 0 or predictions['boxes'][0].numel() == 0 or gt_boxes[0].numel() == 0:
            return 0.0
        
        # Get predictions
        pred_boxes = predictions['boxes']
        pred_scores = predictions['scores']
        
        # Calculate IoU between predictions and ground truth
        ious = box_iou(pred_boxes[0], gt_boxes[0])
        
        # For each prediction, get the best matching ground truth
        max_ious, _ = ious.max(dim=1)
        
        # Calculate true positives, false positives, and false negatives
        true_positives = (max_ious >= iou_threshold).sum().item()
        false_positives = len(pred_boxes[0]) - true_positives
        false_negatives = len(gt_boxes[0]) - true_positives
        
        # Calculate precision and recall
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        
        return f1 