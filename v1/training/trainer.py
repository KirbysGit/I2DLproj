# training/trainer.py

# -----

# Handles the training process including validation and checkpointing.

# -----

# Imports.
import torch
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from v1.utils.box_ops import box_iou
from torch.utils.data import DataLoader
from v1.utils.metrics import DetectionMetrics
from v1.training.optimizer import build_optimizer
from v1.training.scheduler import build_scheduler
from v1.utils.visualization import DetectionVisualizer
from torch.nn import functional as F
from v1.evaluate import visualize_detections
import torchvision.ops
import matplotlib.pyplot as plt
from v1.utils.visualization import visualize_anchors_and_gt, analyze_anchor_coverage
import json


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
        # GPU Setup and Memory Management
        if device is None:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
                torch.backends.cudnn.benchmark = True
                torch.cuda.empty_cache()
                print(f"\nUsing GPU: {torch.cuda.get_device_name(device)}")
                print(f"GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f}GB")
            else:
                device = torch.device('cpu')
                print("\nNo GPU available, using CPU")
        
        self.device = device
        self.config = config
        
        # Set stricter default thresholds
        self.config.setdefault('score_threshold', 0.5)  # Increased from 0.1
        self.config.setdefault('nms_threshold', 0.3)   # Keep at 0.3
        self.config.setdefault('top_k', 100)          # Keep at 100
        self.config.setdefault('min_box_size', 1e-4)  # Keep minimum box dimension
        self.config.setdefault('max_predictions_per_image', 100)  # Keep prediction cap
        self.config.setdefault('iou_threshold', 0.3)   # Keep IoU threshold
        self.config.setdefault('neg_iou_threshold', 0.2)  # Slightly lower negative threshold
        
        # Learning rate setup with warmup
        self.initial_lr = float(config['learning_rate'])
        self.warmup_epochs = config.get('warmup_epochs', 1)
        self.warmup_factor = config.get('warmup_factor', 0.001)
        self.current_lr = self.initial_lr * self.warmup_factor
        
        # Gradient Accumulation Setup
        self.accumulate_grad_batches = config.get('accumulate_grad_batches', 1)
        self.global_step = 0
        
        # Loss weights
        self.config.setdefault('loss_weights', {
            'cls_loss': 2.0,  # Increased classification weight
            'box_loss': 1.0   # Keep box regression weight
        })
        
        # Send Model to Device and Convert to Half Precision if on GPU
        self.model = model.to(device)
        if device.type == 'cuda':
            # Optional: Use mixed precision training
            self.scaler = torch.cuda.amp.GradScaler()
            # Pin memory for faster data transfer to GPU
            self.pin_memory = True
        else:
            self.scaler = None
            self.pin_memory = False
        
        # Data Loaders with GPU Optimizations
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory
        )
        
        # Optimization & Learning Rate Scheduler.   
        self.optimizer = build_optimizer(model, config)
        self.scheduler = build_scheduler(self.optimizer, config)
        
        # Save Directory.
        self.save_dir = Path(config['save_dir'])
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Visualizer.
        self.visualizer = DetectionVisualizer()
        
        # Get Metrics Tracker.
        self.metrics_tracker = config.get("metrics_tracker", None)
        
        # Run Directory.
        self.run_dir = Path(config.get("run_dir", "training_runs/default"))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Debug Options.
        self.debug_options = config.get("debug_options", {})
    
    def train_one_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {
            'loss': 0.0,
            'cls_loss': 0.0,
            'box_loss': 0.0,
            'pos_ratio': 0.0,
            'score_mean': 0.0,
            'score_std': 0.0,
            'num_pos': 0,
            'num_samples': 0,
            'mAP': 0.0,  # Initialize mAP
            'f1': 0.0    # Initialize F1
        }
        
        # Track per-batch metrics for better averaging
        batch_maps = []
        batch_f1s = []
        
        pbar = tqdm(self.train_loader, desc=f'Training Epoch {epoch}')
        optimizer_step = False
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device, handling both tensors and lists
            processed_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    processed_batch[k] = v.to(self.device, non_blocking=True)
                elif isinstance(v, list):
                    # Handle lists of tensors (e.g., boxes and labels)
                    processed_batch[k] = [item.to(self.device, non_blocking=True) if isinstance(item, torch.Tensor) else item for item in v]
                else:
                    processed_batch[k] = v
            
            # Only zero gradients at the start of accumulation
            if self.global_step % self.accumulate_grad_batches == 0:
            self.optimizer.zero_grad()
            
            if self.device.type == 'cuda':
                # Use mixed precision training on GPU
                with torch.cuda.amp.autocast():
                    # Forward pass with ground truth for loss computation
                    outputs = self.model(
                        processed_batch['images'],
                        boxes=processed_batch['bboxes'],
                        labels=processed_batch['labels']
                    )
                    
                    # Check if model computed losses
                    if 'cls_loss' not in outputs or 'box_loss' not in outputs:
                        # If missing, compute manually
                        loss_dict = self.compute_loss(outputs, processed_batch['bboxes'], processed_batch['labels'])
                        loss = loss_dict['loss']
                        cls_loss = loss_dict['cls_loss']
                        box_loss = loss_dict['box_loss']
                    else:
                        cls_loss = outputs['cls_loss']
                        box_loss = outputs['box_loss']
                        loss = cls_loss + box_loss
                    
                    # Scale loss for gradient accumulation
                    loss = loss / self.accumulate_grad_batches
                
                # Scale loss and backward pass
                self.scaler.scale(loss).backward()
                
                # Only update weights after accumulating enough gradients
                if (self.global_step + 1) % self.accumulate_grad_batches == 0:
                # Unscale gradients and clip
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    # print(f"[🧮] Gradient norm: {total_grad_norm:.4f}")
                
                # Update weights with scaled gradients
                self.scaler.step(self.optimizer)
                self.scaler.update()
                    optimizer_step = True
                    
                    # Clear GPU cache periodically
                    if self.global_step % (self.accumulate_grad_batches * 10) == 0:
                        torch.cuda.empty_cache()
            else:
                # Regular training on CPU
                outputs = self.model(
                    processed_batch['images'],
                    boxes=processed_batch['bboxes'],
                    labels=processed_batch['labels']
                )
                
                # Check if model computed losses
                if 'cls_loss' not in outputs or 'box_loss' not in outputs:
                    # If missing, compute manually
                    loss_dict = self.compute_loss(outputs, processed_batch['bboxes'], processed_batch['labels'])
                    loss = loss_dict['loss']
                    cls_loss = loss_dict['cls_loss']
                    box_loss = loss_dict['box_loss']
                else:
                    cls_loss = outputs['cls_loss']
                    box_loss = outputs['box_loss']
                    loss = cls_loss + box_loss
                
                # Scale loss for gradient accumulation
                loss = loss / self.accumulate_grad_batches
                
                # Backward pass
                loss.backward()
                
                # Only update weights after accumulating enough gradients
                if (self.global_step + 1) % self.accumulate_grad_batches == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                
                self.optimizer.step()
                    optimizer_step = True
            
            # Update metrics
            with torch.no_grad():
                batch_size = processed_batch['images'].size(0)
                epoch_metrics['loss'] += loss.item() * batch_size * self.accumulate_grad_batches
                epoch_metrics['cls_loss'] += cls_loss.item() * batch_size * self.accumulate_grad_batches
                epoch_metrics['box_loss'] += box_loss.item() * batch_size * self.accumulate_grad_batches
                epoch_metrics['num_samples'] += batch_size
                
                # Compute score statistics
                if 'detections' in outputs:
                    scores = []
                    for detection in outputs['detections']:
                        if len(detection['scores']) > 0:
                            scores.append(detection['scores'])
                    if scores:
                        scores = torch.cat(scores)
                        epoch_metrics['score_mean'] += scores.mean().item() * batch_size
                        epoch_metrics['score_std'] += scores.std().item() * batch_size
                
                # Count positive predictions
                if 'detections' in outputs:
                    pos_preds = sum(len(detection['boxes']) for detection in outputs['detections'])
                    epoch_metrics['num_pos'] += pos_preds
                    epoch_metrics['pos_ratio'] += (pos_preds / batch_size)
                
                # Compute approximate training mAP and F1 (less frequently to save compute)
                if batch_idx % 10 == 0:  # Every 10 batches
                    try:
                        batch_map = self.compute_map(outputs, processed_batch['bboxes'])
                        batch_f1 = self.compute_f1(outputs, processed_batch['bboxes'])
                        batch_maps.append(batch_map)
                        batch_f1s.append(batch_f1)
                    except Exception as e:
                        if self.debug_options.get('print_metric_errors', False):
                            print(f"Error computing training metrics: {str(e)}")
            
            # Update progress bar
            avg_loss = epoch_metrics['loss'] / epoch_metrics['num_samples']
            avg_cls = epoch_metrics['cls_loss'] / epoch_metrics['num_samples']
            avg_box = epoch_metrics['box_loss'] / epoch_metrics['num_samples']
            avg_pos_ratio = epoch_metrics['pos_ratio'] / (batch_idx + 1)
            
            # Add GPU memory info to progress bar if using CUDA
            if self.device.type == 'cuda':
                mem_used = torch.cuda.memory_reserved() / 1024**2
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'cls': f'{avg_cls:.4f}',
                    'box': f'{avg_box:.4f}',
                    'pos_ratio': f'{avg_pos_ratio:.2%}',
                    'GPU_MB': f'{mem_used:.0f}',
                    'acc_step': f'{(self.global_step % self.accumulate_grad_batches) + 1}/{self.accumulate_grad_batches}'
                })
            else:
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'cls': f'{avg_cls:.4f}',
                    'box': f'{avg_box:.4f}',
                    'pos_ratio': f'{avg_pos_ratio:.2%}',
                    'acc_step': f'{(self.global_step % self.accumulate_grad_batches) + 1}/{self.accumulate_grad_batches}'
                })
            
            # Update global step
            self.global_step += 1
            
            # Step scheduler if optimizer step was taken
            if optimizer_step and self.scheduler is not None:
                self.scheduler.step()
                optimizer_step = False
            
            # Add analysis after forward pass
            self.analyze_training_iteration(outputs, batch, batch_idx, epoch)
        
        # Compute final metrics
        for k in ['loss', 'cls_loss', 'box_loss', 'score_mean', 'score_std']:
            epoch_metrics[k] /= epoch_metrics['num_samples']
        epoch_metrics['pos_ratio'] /= len(self.train_loader)
        
        # Compute final mAP and F1 from batch statistics
        if batch_maps:
            epoch_metrics['mAP'] = sum(batch_maps) / len(batch_maps)
        if batch_f1s:
            epoch_metrics['f1'] = sum(batch_f1s) / len(batch_f1s)
        
        return epoch_metrics
    
    @torch.no_grad()
    def validate(self):
        """Validation with detailed metrics."""
        self.model.eval()
        
        # Initialize Metrics
        val_metrics = {
            'loss': 0.0,
            'cls_loss': 0.0,
            'box_loss': 0.0,
            'mAP': 0.0,
            'f1': 0.0,
            'num_samples': 0,
            'valid_batches': 0,
            'error_batches': 0,
            'total_predictions': 0,
            'total_matches': 0
        }
        
        # Use more lenient thresholds for validation (match training)
        validation_score_threshold = 0.2  # Lower than default 0.7
        validation_iou_threshold = 0.3    # Match training threshold
        
        # Create visualization directory for validation
        vis_dir = self.run_dir / f"epoch_{self.config.get('current_epoch', 0)+1}" / "val_vis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*80}")
        print("Validation")
        print(f"{'='*80}")
        
        # Validation Loop
        pbar = tqdm(self.val_loader, desc='Validating')
        for batch_idx, batch in enumerate(pbar):
            try:
                # Sanity check batch contents
                if 'images' not in batch or 'bboxes' not in batch or 'labels' not in batch:
                    print(f"\nWarning: Batch {batch_idx} missing required keys: {batch.keys()}")
                    val_metrics['error_batches'] += 1
                    continue
                
                # Move Data to Device
                images = batch['images'].to(self.device)
                boxes = [b.to(self.device) for b in batch['bboxes']]
                labels = [l.to(self.device) for l in batch['labels']]
                
                # Validate box format
                for i, box_tensor in enumerate(boxes):
                    if box_tensor.numel() == 0:
                        print(f"\nWarning: Empty boxes in batch {batch_idx}, sample {i}")
                        continue
                    if not (0 <= box_tensor.min() and box_tensor.max() <= 1):
                        print(f"\nWarning: Boxes not normalized [0,1] in batch {batch_idx}, sample {i}")
                        print(f"Range: [{box_tensor.min():.3f}, {box_tensor.max():.3f}]")
                
                # Forward Pass
                predictions = self.model(images)
                
                if 'detections' in predictions:
                    for det in predictions['detections']:
                        val_metrics['total_predictions'] += len(det['boxes'])
                        #if len(det['boxes']) > 0:
                            #print(f"\nBatch {batch_idx} predictions:")
                            #print(f"- Number of predictions: {len(det['boxes'])}")
                            #print(f"- Score range: [{det['scores'].min():.3f}, {det['scores'].max():.3f}]")
                            #print(f"- Mean score: {det['scores'].mean():.3f}")
                
                # Calculate Loss with error handling
                try:
                    loss_dict = self.compute_loss(predictions, boxes, labels)
                    batch_loss = loss_dict['loss'].item()
                    if not np.isfinite(batch_loss):
                        print(f"\nWarning: Non-finite loss in batch {batch_idx}: {batch_loss}")
                        print(f"Loss components: cls={loss_dict['cls_loss']:.4f}, box={loss_dict['box_loss']:.4f}")
                        val_metrics['error_batches'] += 1
                        continue
                    
                    val_metrics['loss'] += batch_loss
                    val_metrics['cls_loss'] += loss_dict['cls_loss'].item()
                    val_metrics['box_loss'] += loss_dict['box_loss'].item()
                except Exception as e:
                    print(f"\nError computing loss in batch {batch_idx}: {str(e)}")
                    val_metrics['error_batches'] += 1
                    continue
                
                # Calculate Metrics with error handling and more lenient thresholds
                try:
                    batch_map = self.compute_map(predictions, boxes, iou_threshold=validation_iou_threshold)
                    batch_f1 = self.compute_f1(predictions, boxes, iou_threshold=validation_iou_threshold)
                    
                    if np.isfinite(batch_map):
                        val_metrics['mAP'] += batch_map
                    if np.isfinite(batch_f1):
                        val_metrics['f1'] += batch_f1
                    
                    # Count matches
                    if 'detections' in predictions:
                        for det, gt_box in zip(predictions['detections'], boxes):
                            if len(det['boxes']) > 0 and len(gt_box) > 0:
                                ious = box_iou(det['boxes'], gt_box)
                                matches = (ious >= validation_iou_threshold).sum().item()
                                val_metrics['total_matches'] += matches
                except Exception as e:
                    print(f"\nError computing metrics in batch {batch_idx}: {str(e)}")
                    val_metrics['error_batches'] += 1
                    continue
                
                # Save visualizations periodically
                if batch_idx % 20 == 0 or batch_idx == len(self.val_loader) - 1:
                    for i in range(min(2, len(images))):
                        try:
                        img = images[i].cpu()
                            detection = predictions['detections'][i]
                            pred_boxes = detection['boxes'].detach().cpu()
                            pred_scores = detection['scores'].detach().cpu()
                        gt_boxes = boxes[i].cpu()
                        
                        vis_path = vis_dir / f"batch_{batch_idx}_sample_{i}.png"
                            self.visualize_predictions(
                            image=img,
                            pred_boxes=pred_boxes,
                            pred_scores=pred_scores,
                            gt_boxes=gt_boxes,
                                output_path=vis_path,
                            title=f"Epoch {self.config.get('current_epoch', 0)+1} - Batch {batch_idx} - Val"
                        )
                        except Exception as e:
                            print(f"\nError visualizing batch {batch_idx}, sample {i}: {str(e)}")
                            continue
                
                # Update batch statistics
                val_metrics['num_samples'] += len(images)
                val_metrics['valid_batches'] += 1
                
                # Update progress bar
                avg_loss = val_metrics['loss'] / val_metrics['valid_batches']
                avg_map = val_metrics['mAP'] / val_metrics['valid_batches']
                avg_f1 = val_metrics['f1'] / val_metrics['valid_batches']
                pbar.set_postfix({
                    'loss': f"{avg_loss:.4f}",
                    'mAP': f"{avg_map:.4f}",
                    'f1': f"{avg_f1:.4f}",
                    'errors': val_metrics['error_batches']
                })
                
            except Exception as e:
                print(f"\nError in validation batch {batch_idx}: {str(e)}")
                val_metrics['error_batches'] += 1
                continue
        
        # Compute final metrics
        if val_metrics['valid_batches'] > 0:
            for k in ['loss', 'cls_loss', 'box_loss', 'mAP', 'f1']:
                val_metrics[k] /= val_metrics['valid_batches']
        
        # Print validation summary
        print("\nValidation Summary:")
        print(f"Processed {val_metrics['num_samples']} samples in {val_metrics['valid_batches']} batches")
        print(f"Error batches: {val_metrics['error_batches']}")
        print(f"Loss: {val_metrics['loss']:.4f}")
        print(f"mAP: {val_metrics['mAP']:.4f}")
        print(f"F1: {val_metrics['f1']:.4f}")
        
        return val_metrics

    def visualize_predictions(self, image, pred_boxes, pred_scores, gt_boxes, output_path, title=""):
        """Helper method to visualize predictions with improved error handling."""
        try:
            # Validate inputs
            if image is None or pred_boxes is None or gt_boxes is None:
                print(f"Warning: Missing required inputs for visualization")
                return
            
        # Get dimensions from the transformed image
        _, H, W = image.shape
        resize_size = (H, W)
        
            # Ensure boxes are on CPU and in the right format
            pred_boxes = pred_boxes.detach().cpu()
            gt_boxes = gt_boxes.detach().cpu()
            
            # Validate and clip boxes
            pred_boxes = pred_boxes.clamp(0.0, 1.0)
            gt_boxes = gt_boxes.clamp(0.0, 1.0)
            
            # Filter out invalid boxes
            valid_pred_mask = (pred_boxes[:, 2] > pred_boxes[:, 0]) & \
                            (pred_boxes[:, 3] > pred_boxes[:, 1]) & \
                            ((pred_boxes[:, 2] - pred_boxes[:, 0]) >= 1e-4) & \
                            ((pred_boxes[:, 3] - pred_boxes[:, 1]) >= 1e-4)
            pred_boxes = pred_boxes[valid_pred_mask]
            if len(pred_scores) > 0:
                pred_scores = pred_scores[valid_pred_mask]
            
            valid_gt_mask = (gt_boxes[:, 2] > gt_boxes[:, 0]) & \
                          (gt_boxes[:, 3] > gt_boxes[:, 1]) & \
                          ((gt_boxes[:, 2] - gt_boxes[:, 0]) >= 1e-4) & \
                          ((gt_boxes[:, 3] - gt_boxes[:, 1]) >= 1e-4)
            gt_boxes = gt_boxes[valid_gt_mask]
            
            # Apply score threshold if scores are provided
            if len(pred_scores) > 0:
                score_threshold = self.config.get('score_threshold', 0.7)
                score_mask = pred_scores >= score_threshold
                pred_boxes = pred_boxes[score_mask]
                pred_scores = pred_scores[score_mask]
            
            # Limit number of boxes to visualize
            max_boxes = self.config.get('vis_max_boxes', 50)
            if len(pred_boxes) > max_boxes:
                if len(pred_scores) > 0:
                    # Keep highest scoring boxes
                    _, top_k = pred_scores.topk(max_boxes)
                    pred_boxes = pred_boxes[top_k]
                    pred_scores = pred_scores[top_k]
                else:
                    # Just take first max_boxes
                    pred_boxes = pred_boxes[:max_boxes]
            
            # Unnormalize image for visualization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image = image * std + mean
            image = image.clamp(0, 1)
            
            # Create figure and axis
            plt.figure(figsize=(10, 10))
            plt.imshow(image.permute(1, 2, 0))
            
            # Plot predicted boxes
            for i in range(len(pred_boxes)):
                box = pred_boxes[i]
                score = pred_scores[i] if len(pred_scores) > 0 else None
                
                # Convert normalized coordinates to pixel coordinates
                x1, y1, x2, y2 = box * torch.tensor([W, H, W, H])
                
                # Create rectangle patch
                rect = plt.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    fill=False, edgecolor='red', linewidth=2
                )
                plt.gca().add_patch(rect)
                
                # Add score if available
                if score is not None:
                    plt.text(
                        x1, y1-2,
                        f'{score:.2f}',
                        color='red',
                        fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.7)
                    )
            
            # Plot ground truth boxes
            for box in gt_boxes:
                # Convert normalized coordinates to pixel coordinates
                x1, y1, x2, y2 = box * torch.tensor([W, H, W, H])
                
                # Create rectangle patch
                rect = plt.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    fill=False, edgecolor='green', linewidth=2
                )
                plt.gca().add_patch(rect)
            
            # Add title with box counts
            plt.title(f"{title}\nPred: {len(pred_boxes)}, GT: {len(gt_boxes)}")
            plt.axis('off')
            
            # Save figure
            plt.savefig(output_path, bbox_inches='tight', dpi=100)
            plt.close()
            
        except Exception as e:
            print(f"Error in visualization: {str(e)}")
            # Try to save a simple version without boxes if possible
            try:
                if image is not None:
                    plt.figure(figsize=(10, 10))
                    plt.imshow(image.permute(1, 2, 0))
                    plt.title(f"{title} (Visualization Error)")
                    plt.axis('off')
                    plt.savefig(output_path)
                    plt.close()
            except Exception as e2:
                print(f"Error saving simple visualization: {str(e2)}")
    
    def train(self):
        """Main Training Loop with checkpoint support and collapse detection."""
        # Initialize Best Validation mAP
        best_val_map = 0
        start_epoch = self.config.get('start_epoch', 0)
        
        # Track metrics for collapse detection
        consecutive_zero_preds = 0
        max_zero_pred_epochs = 3  # Maximum number of epochs with zero predictions before stopping
        
        # Resume from checkpoint if specified
        if self.config.get('resume', False):
            checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob('*.pth'))
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                    start_epoch = self.load_checkpoint(latest_checkpoint)
                    print(f"Resuming training from epoch {start_epoch}")
        
        for epoch in range(start_epoch, self.config['epochs']):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch + 1}/{self.config['epochs']}")
            print(f"{'='*80}")
            
            # Training Phase
            train_metrics = self.train_one_epoch(epoch)
            print(f"\nTraining - Loss: {train_metrics['loss']:.4f}, mAP: {train_metrics['mAP']:.4f}, F1: {train_metrics['f1']:.4f}")
            
            # Check for training collapse
            if train_metrics['mAP'] == 0 and train_metrics['loss'] < 0.001:
                consecutive_zero_preds += 1
                print(f"\n⚠️ WARNING: Possible model collapse detected! ({consecutive_zero_preds}/{max_zero_pred_epochs})")
                if consecutive_zero_preds >= max_zero_pred_epochs:
                    print("\n🛑 Training stopped due to suspected model collapse!")
                    print("Suggestions:")
                    print("1. Check the last good checkpoint (before collapse)")
                    print("2. Review learning rate and gradient values")
                    print("3. Verify loss function components")
                    print("4. Consider reducing learning rate or adjusting confidence thresholds")
                    break
            else:
                consecutive_zero_preds = 0
            
            # Validation Phase
            val_metrics = self.validate()
            print(f"Validation - Loss: {val_metrics['loss']:.4f}, mAP: {val_metrics['mAP']:.4f}, F1: {val_metrics['f1']:.4f}")
            
            # Track Metrics
            if self.metrics_tracker is not None:
                self.metrics_tracker.update(epoch, train_metrics, val_metrics)
                # Plot and save metrics after each epoch
                self.metrics_tracker.plot_metrics(self.run_dir)
                
                # Print improvement metrics
                if epoch > start_epoch:
                    print("\nMetric Changes:")
                    print(f"📉 Loss Δ: {self.metrics_tracker.train_losses[-1] - self.metrics_tracker.train_losses[-2]:.4f}")
                    print(f"📈 mAP Δ: {self.metrics_tracker.train_maps[-1] - self.metrics_tracker.train_maps[-2]:.4f}")
                    print(f"📈 F1 Δ: {self.metrics_tracker.train_f1s[-1] - self.metrics_tracker.train_f1s[-2]:.4f}")
            
            # Save Checkpoint
            if val_metrics['mAP'] > best_val_map:
                best_val_map = val_metrics['mAP']
                self.save_checkpoint(epoch, val_metrics['mAP'], is_best=True)
                print(f"\n📈 New best mAP: {best_val_map:.4f}")
            
            if epoch % self.config['save_freq'] == 0:
                self.save_checkpoint(epoch, val_metrics['mAP'])
            
            # Print Epoch Summary
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"Training   - Loss: {train_metrics['loss']:.4f}, mAP: {train_metrics['mAP']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"Validation - Loss: {val_metrics['loss']:.4f}, mAP: {val_metrics['mAP']:.4f}, F1: {val_metrics['f1']:.4f}")
            print(f"Best mAP: {best_val_map:.4f}")
            print(f"{'='*80}\n")
    
    def save_checkpoint(self, epoch, val_map, is_best=False):
        """Save model checkpoint with full state."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_map': val_map,
            'metrics_tracker': self.metrics_tracker
        }
        
        # Save checkpoint
        if is_best:
            path = self.save_dir / 'best_model.pth'
        else:
            path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, path)
        
        # Save latest checkpoint
        latest_path = self.run_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint if this is the best validation mAP
        if is_best:
            best_path = self.run_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and restore model state."""
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state if available
        if self.scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load metrics tracker if available
        if 'metrics_tracker' in checkpoint and checkpoint['metrics_tracker']:
            self.metrics_tracker = checkpoint['metrics_tracker']
        
        return checkpoint.get('epoch', -1) + 1  # Return next epoch number
    
    @staticmethod
    def collate_fn(batch):
        """Custom collate function for the data loader."""
        # Handle both 'images' and 'image' keys
        images = None
        try:
            images = torch.stack([item['images'] for item in batch])
        except KeyError:
            try:
                images = torch.stack([item['image'] for item in batch])
            except KeyError:
                raise KeyError("Neither 'images' nor 'image' key found in batch data")

        # Stack Boxes - handle both keys
        boxes = None
        try:
            boxes = [item['bboxes'] for item in batch]
        except KeyError:
            try:
                boxes = [item['boxes'] for item in batch]
            except KeyError:
                raise KeyError("Neither 'bboxes' nor 'boxes' key found in batch data")

        # Stack Labels
        labels = [item['labels'] for item in batch]
        
        return {
            'images': images,
            'bboxes': boxes,
            'labels': labels
        }
    
    def compute_loss(self, predictions, gt_boxes, gt_labels):
        """Compute the total loss with improved numerical stability and edge case handling."""
        
        # Initialize losses with stability
        cls_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        box_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Early exit if no detections
        if 'detections' not in predictions or not predictions['detections']:
            print("Warning: No detections in predictions")
            return {
                'loss': torch.tensor(0.0, device=self.device, requires_grad=True),
                'cls_loss': torch.tensor(0.0, device=self.device),
                'box_loss': torch.tensor(0.0, device=self.device)
            }
        
        batch_size = len(gt_boxes)
        valid_batches = 0
        total_positives = 0
        total_predictions = 0
        
        # Validate batch sizes match
        if len(predictions['detections']) != batch_size:
            print(f"Warning: Mismatch between detections ({len(predictions['detections'])}) and batch size ({batch_size})")
            # Use the smaller size to avoid index errors
            batch_size = min(len(predictions['detections']), batch_size)
        
        # Get thresholds from config
        score_threshold = self.config.get('score_threshold', 0.05)
        nms_threshold = self.config.get('nms_threshold', 0.5)
        top_k = self.config.get('top_k', 200)
        min_box_size = self.config.get('min_box_size', 1e-4)
        max_predictions = self.config.get('max_predictions_per_image', 100)
        iou_threshold = self.config.get('iou_threshold', 0.5)
        neg_iou_threshold = self.config.get('neg_iou_threshold', 0.4)
        
        for i in range(batch_size):
            try:
                # Validate indices
                if i >= len(gt_boxes) or i >= len(predictions['detections']):
                    print(f"Warning: Skipping batch {i} due to index mismatch")
                    continue
                
                # Get batch items safely
                try:
                batch_gt_boxes = gt_boxes[i]
                batch_gt_labels = gt_labels[i]
                except IndexError as e:
                    print(f"Error accessing ground truth for batch {i}: {str(e)}")
                    continue
                
                # Skip if no ground truth
                if len(batch_gt_boxes) == 0:
                    if self.config.get('verbose', False):
                        print(f"Skipping batch {i}: No ground truth boxes")
                    continue
                    
                # Validate and clip ground truth boxes
                batch_gt_boxes = batch_gt_boxes.clamp(0.0, 1.0)
                
                # Filter invalid ground truth boxes
                gt_widths = batch_gt_boxes[:, 2] - batch_gt_boxes[:, 0]
                gt_heights = batch_gt_boxes[:, 3] - batch_gt_boxes[:, 1]
                valid_gt_mask = (gt_widths > min_box_size) & (gt_heights > min_box_size)
                
                if not valid_gt_mask.any():
                    if self.config.get('verbose', False):
                        print(f"Skipping batch {i}: No valid ground truth boxes")
                    continue
                
                batch_gt_boxes = batch_gt_boxes[valid_gt_mask]
                batch_gt_labels = batch_gt_labels[valid_gt_mask]
                
                # Get predictions safely
                try:
                    detection = predictions['detections'][i]
                    batch_pred_boxes = detection.get('boxes', torch.empty(0, 4, device=self.device))
                    batch_pred_scores = detection.get('scores', torch.empty(0, device=self.device))
                except (IndexError, KeyError, AttributeError) as e:
                    print(f"Error accessing predictions for batch {i}: {str(e)}")
                    continue
                
                # Skip if no predictions
                if len(batch_pred_boxes) == 0:
                    if self.config.get('verbose', False):
                        print(f"Skipping batch {i}: No predicted boxes")
                    continue
                
                # Validate prediction shapes
                if batch_pred_boxes.shape[-1] != 4:
                    print(f"Warning: Invalid prediction box shape in batch {i}: {batch_pred_boxes.shape}")
                    continue
                
                # Validate and clip predicted boxes
                batch_pred_boxes = batch_pred_boxes.clamp(0.0, 1.0)
                
                # Filter out invalid boxes (zero width/height)
                pred_widths = batch_pred_boxes[:, 2] - batch_pred_boxes[:, 0]
                pred_heights = batch_pred_boxes[:, 3] - batch_pred_boxes[:, 1]
                valid_box_mask = (pred_widths > min_box_size) & (pred_heights > min_box_size)
                
                batch_pred_boxes = batch_pred_boxes[valid_box_mask]
                batch_pred_scores = batch_pred_scores[valid_box_mask]
                
                if len(batch_pred_boxes) == 0:
                    if self.config.get('verbose', False):
                        print(f"Skipping batch {i}: No valid predicted boxes")
                    continue
                
                # Apply score threshold
                score_mask = batch_pred_scores >= score_threshold
                batch_pred_boxes = batch_pred_boxes[score_mask]
                batch_pred_scores = batch_pred_scores[score_mask]
                
                if len(batch_pred_boxes) == 0:
                    if self.config.get('verbose', False):
                        print(f"Skipping batch {i}: No predictions above score threshold")
                    continue
                
                # Apply NMS
                try:
                    keep = torchvision.ops.nms(
                        batch_pred_boxes,
                        batch_pred_scores,
                        nms_threshold
                    )
                except RuntimeError as e:
                    print(f"NMS error in batch {i}: {str(e)}")
                    continue
                
                # Apply top-k filtering
                if len(keep) > top_k:
                    scores_for_topk = batch_pred_scores[keep]
                    _, topk_indices = scores_for_topk.topk(min(top_k, len(scores_for_topk)))
                    keep = keep[topk_indices]
                
                batch_pred_boxes = batch_pred_boxes[keep]
                batch_pred_scores = batch_pred_scores[keep]
                
                # Enforce maximum predictions per image
                if len(batch_pred_boxes) > max_predictions:
                    _, top_indices = batch_pred_scores.topk(max_predictions)
                    batch_pred_boxes = batch_pred_boxes[top_indices]
                    batch_pred_scores = batch_pred_scores[top_indices]
                
                # Track total predictions after filtering
                total_predictions += len(batch_pred_boxes)
                
                # Compute IoU matrix
                try:
                ious = box_iou(batch_pred_boxes, batch_gt_boxes)
                except RuntimeError as e:
                    print(f"IoU computation error in batch {i}: {str(e)}")
                    continue
                
                # Check for NaN values
                if torch.isnan(ious).any():
                    print(f"Warning: NaN values in IoU matrix for batch {i}")
                    continue
                
                # Assign targets with stricter thresholds
                max_ious, gt_indices = ious.max(dim=1)
                pos_mask = max_ious >= iou_threshold
                neg_mask = max_ious < neg_iou_threshold
                
                # Skip if no positive matches
                #if not pos_mask.any():
                    #if self.config.get('verbose', False):
                        #print(f"Skipping batch {i}: No positive matches")
                    #continue
                
                # Initialize target scores
                target_scores = torch.zeros_like(batch_pred_scores)
                
                # Assign positive samples safely
                try:
                matched_labels = batch_gt_labels[gt_indices[pos_mask]]
                if matched_labels.ndim > 1:
                    matched_labels = matched_labels.squeeze(-1)
                target_scores[pos_mask] = matched_labels.float()
                except IndexError as e:
                    print(f"Error assigning labels in batch {i}: {str(e)}")
                    continue
                
                # Compute classification loss with stability
                valid_mask = pos_mask | neg_mask
                if valid_mask.any():
                    # Add small epsilon for numerical stability
                    eps = 1e-7
                    pred_probs = torch.sigmoid(batch_pred_scores[valid_mask])
                    pred_probs = torch.clamp(pred_probs, eps, 1.0 - eps)
                    
                    try:
                    batch_cls_loss = F.binary_cross_entropy(
                        pred_probs,
                        target_scores[valid_mask],
                        reduction='mean'
                    )
                    
                    if torch.isfinite(batch_cls_loss):
                        cls_loss = cls_loss + batch_cls_loss
                        else:
                            print(f"Warning: Non-finite classification loss in batch {i}")
                    except RuntimeError as e:
                        print(f"Classification loss error in batch {i}: {str(e)}")
                        continue
                
                # Compute box regression loss for positive samples
                if pos_mask.any():
                    try:
                    matched_boxes = batch_gt_boxes[gt_indices[pos_mask]]
                    batch_box_loss = F.smooth_l1_loss(
                        batch_pred_boxes[pos_mask],
                        matched_boxes,
                        reduction='mean',
                        beta=1.0
                    )
                    
                    if torch.isfinite(batch_box_loss):
                        box_loss = box_loss + batch_box_loss
                        else:
                            print(f"Warning: Non-finite box loss in batch {i}")
                    except RuntimeError as e:
                        print(f"Box loss error in batch {i}: {str(e)}")
                        continue
                
                valid_batches += 1
                total_positives += pos_mask.sum().item()
                
            except Exception as e:
                print(f"Error in batch {i}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Normalize losses by valid batches
        if valid_batches > 0:
            cls_loss = cls_loss / valid_batches
            box_loss = box_loss / valid_batches
        else:
            print("Warning: No valid batches for loss computation")
        
        # Ensure losses are finite
        if not torch.isfinite(cls_loss):
            print("Warning: Classification loss is not finite, resetting to zero")
            cls_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        if not torch.isfinite(box_loss):
            print("Warning: Box loss is not finite, resetting to zero")
            box_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Compute total loss with weighting
        cls_weight = self.config.get('loss_weights', {}).get('cls_loss', 1.0)
        box_weight = self.config.get('loss_weights', {}).get('box_loss', 1.0)
        total_loss = cls_weight * cls_loss + box_weight * box_loss
        
        # Log statistics if verbose
        #if self.config.get('verbose', False) and valid_batches > 0:
            #print(f"\nLoss Statistics:")
            #print(f"Valid batches: {valid_batches}/{batch_size}")
            #print(f"Total predictions after filtering: {total_predictions}")
            #print(f"Total positive matches: {total_positives}")
            #print(f"Classification loss: {cls_loss.item():.4f}")
            #print(f"Box regression loss: {box_loss.item():.4f}")
            #print(f"Total loss: {total_loss.item():.4f}")
            #print(f"Average positives per batch: {total_positives/valid_batches:.1f}")
            #print(f"Average predictions per batch: {total_predictions/valid_batches:.1f}")
        
        return {
            'loss': total_loss,
            'cls_loss': cls_loss.detach(),
            'box_loss': box_loss.detach()
        }
    
    def compute_map(self, predictions, gt_boxes, iou_threshold=None):
        """Compute mean Average Precision with improved filtering."""
        if 'detections' not in predictions:
            return 0.0
        
        # Use more lenient thresholds
        score_threshold = 0.2  # Lower than default
        nms_threshold = 0.5   # More lenient NMS
        top_k = 200          # Allow more predictions
        min_box_size = 1e-4  # Smaller minimum box size
        max_predictions = 200  # Allow more predictions
        iou_threshold = iou_threshold or 0.3  # Match training threshold
        
        batch_aps = []
        total_predictions = 0
        total_matches = 0
        
        for i, detection in enumerate(predictions['detections']):
            try:
                # Skip if no predictions or ground truth
                if i >= len(gt_boxes) or len(detection['boxes']) == 0 or len(gt_boxes[i]) == 0:
                    continue
                
                # Get predictions and ground truth for this sample
                pred_boxes = detection['boxes']
                pred_scores = detection['scores']
                batch_gt_boxes = gt_boxes[i]
                
                # Print detailed stats for debugging
                #if self.config.get('verbose', False):
                #    print(f"\nBatch {i} Statistics:")
                #    print(f"- GT boxes: {len(batch_gt_boxes)}")
                #    print(f"- Initial predictions: {len(pred_boxes)}")
                #    if len(pred_scores) > 0:
                #        print(f"- Score range: [{pred_scores.min():.3f}, {pred_scores.max():.3f}]")
                #        print(f"- Mean score: {pred_scores.mean():.3f}")
                
                # Validate and clip boxes
                pred_boxes = pred_boxes.clamp(0.0, 1.0)
                batch_gt_boxes = batch_gt_boxes.clamp(0.0, 1.0)
                
                # Filter invalid boxes with more lenient size thresholds
                gt_widths = batch_gt_boxes[:, 2] - batch_gt_boxes[:, 0]
                gt_heights = batch_gt_boxes[:, 3] - batch_gt_boxes[:, 1]
                valid_gt_mask = (gt_widths > min_box_size) & (gt_heights > min_box_size)
                
                if not valid_gt_mask.any():
                    if self.config.get('verbose', False):
                        print("No valid GT boxes after size filtering")
                    continue
                
                batch_gt_boxes = batch_gt_boxes[valid_gt_mask]
                
                # Filter predicted boxes with same lenient thresholds
                pred_widths = pred_boxes[:, 2] - pred_boxes[:, 0]
                pred_heights = pred_boxes[:, 3] - pred_boxes[:, 1]
                valid_box_mask = (pred_widths > min_box_size) & (pred_heights > min_box_size)
                
                pred_boxes = pred_boxes[valid_box_mask]
                pred_scores = pred_scores[valid_box_mask]
                
                if len(pred_boxes) == 0:
                    if self.config.get('verbose', False):
                        print("No valid predicted boxes after size filtering")
                    continue
                
                # Apply score threshold
                score_mask = pred_scores >= score_threshold
                pred_boxes = pred_boxes[score_mask]
                pred_scores = pred_scores[score_mask]
                
                if len(pred_boxes) == 0:
                    if self.config.get('verbose', False):
                        print("No predictions above score threshold")
                    continue
                
                # Apply NMS
                keep = torchvision.ops.nms(
                    pred_boxes,
                    pred_scores,
                    nms_threshold
                )
                
                # Apply top-k filtering
                if len(keep) > top_k:
                    # Keep highest scoring boxes
                    scores_for_topk = pred_scores[keep]
                    _, topk_indices = scores_for_topk.topk(min(top_k, len(scores_for_topk)))
                    keep = keep[topk_indices]
                
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]
                
                #if self.config.get('verbose', False):
                #    print(f"After filtering:")
                #    print(f"- Predictions remaining: {len(pred_boxes)}")
                
                # Track total predictions
                total_predictions += len(pred_boxes)
                
                # Calculate IoU between predictions & ground truth
                ious = box_iou(pred_boxes, batch_gt_boxes)
                
                # Check for NaN values
                if torch.isnan(ious).any():
                    print(f"Warning: NaN values in IoU matrix for batch {i}")
                    continue
                
                # For each prediction, get the best matching ground truth
        max_ious, _ = ious.max(dim=1)
        
                # Calculate precision at different recall points
                sorted_indices = torch.argsort(pred_scores, descending=True)
        precisions = []
        recalls = []
        
                # Get number of ground truth boxes
                num_gt = len(batch_gt_boxes)
        true_positives = 0
        
                # Track matches above threshold
                matches = (max_ious >= iou_threshold).sum().item()
                total_matches += matches
                
                # Iterate over predictions in order of confidence
        for idx in sorted_indices:
            if max_ious[idx] >= iou_threshold:
                true_positives += 1
            
                    # Calculate precision & recall
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
        
                batch_aps.append(ap)
                
            except Exception as e:
                print(f"Error computing mAP for batch {i}: {str(e)}")
                continue
        
        # Log statistics
        #if len(batch_aps) > 0:
        #    print(f"\nMAP Statistics:")
        #    print(f"Processed batches: {len(batch_aps)}")
        #    print(f"Total predictions after filtering: {total_predictions}")
        #    print(f"Total matches above IoU threshold: {total_matches}")
        #    print(f"Average predictions per batch: {total_predictions/len(batch_aps):.1f}")
        #    print(f"Average matches per batch: {total_matches/len(batch_aps):.1f}")
        #    print(f"Average AP: {np.mean(batch_aps):.4f}")
        
        # Return mean AP across batch
        return np.mean(batch_aps) if batch_aps else 0.0
    
    def compute_f1(self, predictions, gt_boxes, iou_threshold=None):
        """Compute F1 score with improved filtering."""
        if 'detections' not in predictions:
            return 0.0
        
        # Get thresholds from config
        score_threshold = self.config['score_threshold']
        nms_threshold = self.config['nms_threshold']
        top_k = self.config['top_k']
        min_box_size = self.config['min_box_size']
        max_predictions = self.config['max_predictions_per_image']
        iou_threshold = iou_threshold or self.config['iou_threshold']
        
        batch_f1s = []
        total_predictions = 0
        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0
        
        for i, detection in enumerate(predictions['detections']):
            try:
                # Skip if no predictions or ground truth
                if i >= len(gt_boxes) or len(detection['boxes']) == 0 or len(gt_boxes[i]) == 0:
                    continue
                
                # Get predictions and ground truth for this sample
                pred_boxes = detection['boxes']
                pred_scores = detection['scores']
                batch_gt_boxes = gt_boxes[i]
                
                # Validate and clip boxes
                pred_boxes = pred_boxes.clamp(0.0, 1.0)
                batch_gt_boxes = batch_gt_boxes.clamp(0.0, 1.0)
                
                # Filter invalid ground truth boxes
                gt_widths = batch_gt_boxes[:, 2] - batch_gt_boxes[:, 0]
                gt_heights = batch_gt_boxes[:, 3] - batch_gt_boxes[:, 1]
                valid_gt_mask = (gt_widths > min_box_size) & (gt_heights > min_box_size)
                
                if not valid_gt_mask.any():
                    continue
                
                batch_gt_boxes = batch_gt_boxes[valid_gt_mask]
                num_gt = len(batch_gt_boxes)
                
                # Filter out invalid predicted boxes
                pred_widths = pred_boxes[:, 2] - pred_boxes[:, 0]
                pred_heights = pred_boxes[:, 3] - pred_boxes[:, 1]
                valid_box_mask = (pred_widths > min_box_size) & (pred_heights > min_box_size)
                
                pred_boxes = pred_boxes[valid_box_mask]
                pred_scores = pred_scores[valid_box_mask]
                
                if len(pred_boxes) == 0:
                    # Count all valid ground truth as false negatives
                    total_false_negatives += num_gt
                    continue
                
                # Apply score threshold
                score_mask = pred_scores >= score_threshold
                pred_boxes = pred_boxes[score_mask]
                pred_scores = pred_scores[score_mask]
                
                if len(pred_boxes) == 0:
                    # Count all valid ground truth as false negatives
                    total_false_negatives += num_gt
                    continue
                
                # Apply NMS
                keep = torchvision.ops.nms(
                    pred_boxes,
                    pred_scores,
                    nms_threshold
                )
                
                # Apply top-k filtering
                if len(keep) > top_k:
                    # Keep highest scoring boxes
                    scores_for_topk = pred_scores[keep]
                    _, topk_indices = scores_for_topk.topk(min(top_k, len(scores_for_topk)))
                    keep = keep[topk_indices]
                
                pred_boxes = pred_boxes[keep]
                pred_scores = pred_scores[keep]
                
                # Enforce maximum predictions per image
                if len(pred_boxes) > max_predictions:
                    _, top_indices = pred_scores.topk(max_predictions)
                    pred_boxes = pred_boxes[top_indices]
                    pred_scores = pred_scores[top_indices]
                
                # Track total predictions
                total_predictions += len(pred_boxes)
                
                # Calculate IoU between predictions & ground truth
                ious = box_iou(pred_boxes, batch_gt_boxes)
                
                # Check for NaN values
                if torch.isnan(ious).any():
                    print(f"Warning: NaN values in IoU matrix for batch {i}")
                    continue
                
                # For each prediction, get the best matching ground truth
        max_ious, _ = ious.max(dim=1)
        
                # Calculate metrics
        true_positives = (max_ious >= iou_threshold).sum().item()
                false_positives = len(pred_boxes) - true_positives
                false_negatives = num_gt - true_positives
                
                # Update totals
                total_true_positives += true_positives
                total_false_positives += false_positives
                total_false_negatives += false_negatives
                
                # Calculate precision & recall
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        
                # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
                batch_f1s.append(f1)
                
            except Exception as e:
                print(f"Error computing F1 for batch {i}: {str(e)}")
                continue
        
        # Log statistics if verbose
        #if self.config.get('verbose', False) and len(batch_f1s) > 0:
            #print(f"\nF1 Statistics:")
            #print(f"Processed batches: {len(batch_f1s)}")
            #print(f"Total predictions after filtering: {total_predictions}")
            #print(f"True positives: {total_true_positives}")
            #print(f"False positives: {total_false_positives}")
            #print(f"False negatives: {total_false_negatives}")
            #print(f"Average predictions per batch: {total_predictions/len(batch_f1s):.1f}")
            #if total_true_positives + total_false_positives > 0:
            #    print(f"Precision: {total_true_positives/(total_true_positives + total_false_positives):.4f}")
            #if total_true_positives + total_false_negatives > 0:
            #    print(f"Recall: {total_true_positives/(total_true_positives + total_false_negatives):.4f}")
            #print(f"Average F1: {np.mean(batch_f1s):.4f}")
        
        # Return mean F1 across batch
        return np.mean(batch_f1s) if batch_f1s else 0.0 

    def analyze_training_iteration(self, outputs, batch, iteration, epoch):
        """Analyze model behavior during training iterations."""
        if iteration % self.config.get('analysis_interval', 50) != 0:  # Every 50 iterations by default
            return
            
        try:
            # Create analysis directory
            analysis_dir = self.run_dir / f"epoch_{epoch}" / f"iter_{iteration}"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            
            # Get predictions and ground truth - keep on GPU until needed
            detections = outputs['detections'][0]  # First batch item
            pred_boxes = detections['boxes'].detach().to(self.device)  # Keep on GPU
            pred_scores = detections['scores'].detach().to(self.device)
            image = batch['images'][0]  # Take first image from batch
            gt_boxes = batch['bboxes'][0].to(self.device)  # Ensure on GPU
            
            # NEW: Anchor Analysis
            # Get anchors from detection head
            anchors_by_level = self.model.detection_head.get_anchors(
                image_size=torch.tensor([image.shape[2], image.shape[1]]).to(image.device)
            )
            
            # Visualize anchors and ground truth
            vis_path = analysis_dir / 'anchor_coverage.png'
            visualize_anchors_and_gt(
                image=image,
                gt_boxes=gt_boxes,
                anchors_by_level=anchors_by_level,
                output_path=str(vis_path)
            )
            
            # Analyze anchor coverage
            coverage_stats = analyze_anchor_coverage(gt_boxes, anchors_by_level)
            
            # Print anchor analysis
            print("\n📊 Anchor Coverage Analysis:")
            print("\nGround Truth Statistics:")
            for k, v in coverage_stats['gt_stats'].items():
                print(f"- {k}: {v}")
            
            print("\nAnchor Statistics by Level:")
            for level, stats in coverage_stats['anchor_stats'].items():
                print(f"\nLevel {level}:")
                for k, v in stats.items():
                    print(f"- {k}: {v}")
            
            print("\nMatching Statistics by Level:")
            for level, stats in coverage_stats['matching_stats'].items():
                print(f"\nLevel {level}:")
                for k, v in stats.items():
                    print(f"- {k}: {v}")
            
            # Save anchor statistics
            with open(analysis_dir / 'anchor_stats.json', 'w') as f:
                json.dump(coverage_stats, f, indent=4)
            
            # EXISTING: Box Delta Analysis
            dx = outputs['bbox_preds'][0][..., 0].flatten().detach().to(self.device)
            dy = outputs['bbox_preds'][0][..., 1].flatten().detach().to(self.device)
            dw = outputs['bbox_preds'][0][..., 2].flatten().detach().to(self.device)
            dh = outputs['bbox_preds'][0][..., 3].flatten().detach().to(self.device)
            
            plt.figure(figsize=(15, 10))
            
            # Move to CPU only when needed for plotting
            dx_np = dx.cpu().numpy()
            dy_np = dy.cpu().numpy()
            dw_np = dw.cpu().numpy()
            dh_np = dh.cpu().numpy()
            
            # Plot delta distributions
            plt.subplot(2, 2, 1)
            plt.hist2d(dx_np, dy_np, bins=30, range=[[-2, 2], [-2, 2]])
            plt.colorbar()
            plt.title("Center Shift Distribution (dx vs dy)")
            
            plt.subplot(2, 2, 2)
            plt.hist2d(dw_np, dh_np, bins=30, range=[[-2, 2], [-2, 2]])
            plt.colorbar()
            plt.title("Size Change Distribution (dw vs dh)")
            
            plt.subplot(2, 2, 3)
            plt.hist(dx_np, bins=50, alpha=0.5, label='dx', range=[-2, 2])
            plt.hist(dy_np, bins=50, alpha=0.5, label='dy', range=[-2, 2])
            plt.title("Center Shift Histograms")
            plt.legend()
            
            plt.subplot(2, 2, 4)
            plt.hist(dw_np, bins=50, alpha=0.5, label='dw', range=[-2, 2])
            plt.hist(dh_np, bins=50, alpha=0.5, label='dh', range=[-2, 2])
            plt.title("Size Change Histograms")
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(analysis_dir / 'box_deltas.png')
            plt.close()
            
            # EXISTING: Confidence Score Analysis
            plt.figure(figsize=(15, 5))
            
            pred_scores_np = pred_scores.cpu().numpy()
            plt.subplot(1, 2, 1)
            plt.hist(pred_scores_np, bins=50, range=(0, 1))
            plt.title("Confidence Score Distribution")
            plt.xlabel("Score")
            plt.ylabel("Count")
            
            # Score vs Box Size correlation
            box_sizes = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
            box_sizes_np = box_sizes.cpu().numpy()
            plt.subplot(1, 2, 2)
            plt.scatter(pred_scores_np, box_sizes_np, alpha=0.5)
            plt.title("Score vs Box Size")
            plt.xlabel("Confidence Score")
            plt.ylabel("Box Size (normalized)")
            
            plt.tight_layout()
            plt.savefig(analysis_dir / 'confidence_analysis.png')
            plt.close()
            
            # EXISTING: Spatial Distribution Analysis
            plt.figure(figsize=(15, 5))
            
            # Prediction centers
            pred_centers_x = ((pred_boxes[:, 0] + pred_boxes[:, 2]) / 2).detach()
            pred_centers_y = ((pred_boxes[:, 1] + pred_boxes[:, 3]) / 2).detach()
            pred_centers_x_np = pred_centers_x.cpu().numpy()
            pred_centers_y_np = pred_centers_y.cpu().numpy()
            
            plt.subplot(1, 2, 1)
            plt.hist2d(pred_centers_x_np, pred_centers_y_np, 
                      bins=30, range=[[0, 1], [0, 1]])
            plt.colorbar()
            plt.title("Prediction Center Distribution")
            
            # Ground truth centers
            if len(gt_boxes) > 0:
                gt_centers_x = ((gt_boxes[:, 0] + gt_boxes[:, 2]) / 2).detach()
                gt_centers_y = ((gt_boxes[:, 1] + gt_boxes[:, 3]) / 2).detach()
                gt_centers_x_np = gt_centers_x.cpu().numpy()
                gt_centers_y_np = gt_centers_y.cpu().numpy()
                
                plt.subplot(1, 2, 2)
                plt.hist2d(gt_centers_x_np, gt_centers_y_np, 
                          bins=30, range=[[0, 1], [0, 1]])
                plt.colorbar()
                plt.title("Ground Truth Center Distribution")
            
            plt.tight_layout()
            plt.savefig(analysis_dir / 'spatial_distribution.png')
            plt.close()
            
            # EXISTING: IoU Analysis
            if len(gt_boxes) > 0:
                # Compute IoUs on GPU
                ious = box_iou(pred_boxes, gt_boxes)
                max_ious, _ = ious.max(dim=1)
                max_ious = max_ious.detach()
                
                # Move to CPU for plotting
                max_ious_np = max_ious.cpu().numpy()
                pred_scores_np = pred_scores.cpu().numpy()
                
                plt.figure(figsize=(10, 5))
                
                plt.subplot(1, 2, 1)
                plt.hist(max_ious_np, bins=50, range=(0, 1))
                plt.title("Max IoU Distribution")
                plt.xlabel("IoU")
                plt.ylabel("Count")
                
                plt.subplot(1, 2, 2)
                plt.scatter(pred_scores_np, max_ious_np, alpha=0.5)
                plt.title("IoU vs Confidence")
                plt.xlabel("Confidence Score")
                plt.ylabel("Max IoU")
                
                plt.tight_layout()
                plt.savefig(analysis_dir / 'iou_analysis.png')
                plt.close()
            
            # EXISTING: Summary Statistics
            stats = {
                'mean_dx': dx.mean().item(),
                'mean_dy': dy.mean().item(),
                'mean_dw': dw.mean().item(),
                'mean_dh': dh.mean().item(),
                'std_dx': dx.std().item(),
                'std_dy': dy.std().item(),
                'std_dw': dw.std().item(),
                'std_dh': dh.std().item(),
                'mean_score': pred_scores.mean().item(),
                'num_predictions': len(pred_boxes),
                'mean_iou': max_ious.mean().item() if len(gt_boxes) > 0 else 0.0
            }
            
            # Add additional statistics
            stats.update({
                'box_ranges': {
                    'pred_x_range': (
                        [float(pred_boxes[:, 0].min().cpu()), float(pred_boxes[:, 0].max().cpu())]
                        if pred_boxes.numel() > 0 else [None, None]
                    ),
                    'pred_y_range': (
                        [float(pred_boxes[:, 1].min().cpu()), float(pred_boxes[:, 1].max().cpu())]
                        if pred_boxes.numel() > 0 else [None, None]
                    ),
                    'pred_width_range': (
                        [float((pred_boxes[:, 2] - pred_boxes[:, 0]).min().cpu()),
                        float((pred_boxes[:, 2] - pred_boxes[:, 0]).max().cpu())]
                        if pred_boxes.numel() > 0 else [None, None]
                    ),
                    'pred_height_range': (
                        [float((pred_boxes[:, 3] - pred_boxes[:, 1]).min().cpu()),
                        float((pred_boxes[:, 3] - pred_boxes[:, 1]).max().cpu())]
                        if pred_boxes.numel() > 0 else [None, None]
                    )
                },
                'score_stats': {
                    'min_score': float(pred_scores.min().cpu()) if pred_scores.numel() > 0 else None,
                    'max_score': float(pred_scores.max().cpu()) if pred_scores.numel() > 0 else None,
                    'median_score': float(pred_scores.median().cpu()) if pred_scores.numel() > 0 else None,
                    'high_conf_ratio': float((pred_scores > 0.5).float().mean().cpu()) if pred_scores.numel() > 0 else 0.0
                },
                'delta_stats': {
                    'dx_range': [float(dx.min().cpu()), float(dx.max().cpu())],
                    'dy_range': [float(dy.min().cpu()), float(dy.max().cpu())],
                    'dw_range': [float(dw.min().cpu()), float(dw.max().cpu())],
                    'dh_range': [float(dh.min().cpu()), float(dh.max().cpu())]
                },
                'anchor_coverage': coverage_stats  # Add anchor coverage stats to main stats
            })
            
            # Save all statistics
            with open(analysis_dir / 'stats.json', 'w') as f:
                json.dump(stats, f, indent=4)
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            import traceback
            traceback.print_exc() 