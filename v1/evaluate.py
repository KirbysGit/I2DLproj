# evaluate.py

import torch
import argparse
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
from PIL import Image, ImageDraw
from torch.serialization import add_safe_globals

from v1.model.detector import ObjectDetector
from v1.data.dataset import SKU110KDataset
from v1.utils.box_ops import box_iou
from torchvision.ops import nms

debug = True

# Define MetricsTracker class to match what was saved in checkpoint
class MetricsTracker:
    def __init__(self):
        self.metrics = {}
        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        self.train_maps = []
        self.val_maps = []
        self.train_f1s = []
        self.val_f1s = []
    
    def update(self, epoch, train_metrics, val_metrics):
        """Update metrics with new epoch data."""
        self.epochs.append(epoch)
        
        # Update training metrics
        self.train_losses.append(train_metrics.get('loss', 0))
        self.train_maps.append(train_metrics.get('mAP', 0))
        self.train_f1s.append(train_metrics.get('f1', 0))
        
        # Update validation metrics
        self.val_losses.append(val_metrics.get('loss', 0))
        self.val_maps.append(val_metrics.get('mAP', 0))
        self.val_f1s.append(val_metrics.get('f1', 0))
        
        # Store in metrics dict for backward compatibility
        for key, value in train_metrics.items():
            if key not in self.metrics:
                self.metrics[f'train_{key}'] = []
            self.metrics[f'train_{key}'].append(value)
        
        for key, value in val_metrics.items():
            if key not in self.metrics:
                self.metrics[f'val_{key}'] = []
            self.metrics[f'val_{key}'].append(value)
        
        # Print improvement metrics if we have more than one epoch
        if len(self.epochs) > 1:
            delta_loss = self.train_losses[-1] - self.train_losses[-2]
            delta_map = self.train_maps[-1] - self.train_maps[-2]
            delta_f1 = self.train_f1s[-1] - self.train_f1s[-2]
            
            print(f"\nMetric Changes:")
            print(f"📉 Loss Δ: {delta_loss:.4f} | 📈 mAP Δ: {delta_map:.4f} | 📈 F1 Δ: {delta_f1:.4f}")
    
    def plot_metrics(self, output_dir):
        """Plot training and validation metrics."""
        import matplotlib.pyplot as plt
        from pathlib import Path
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style for better visualization
        plt.style.use('seaborn')
        
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        # Plot Loss
        ax1.plot(self.epochs, self.train_losses, label='Train Loss', marker='o')
        ax1.plot(self.epochs, self.val_losses, label='Val Loss', marker='o')
        ax1.set_title('Loss vs. Epoch')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot mAP
        ax2.plot(self.epochs, self.train_maps, label='Train mAP', marker='o')
        ax2.plot(self.epochs, self.val_maps, label='Val mAP', marker='o')
        ax2.set_title('mAP vs. Epoch')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mAP')
        ax2.legend()
        ax2.grid(True)
        
        # Plot F1 Score
        ax3.plot(self.epochs, self.train_f1s, label='Train F1', marker='o')
        ax3.plot(self.epochs, self.val_f1s, label='Val F1', marker='o')
        ax3.set_title('F1 Score vs. Epoch')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('F1 Score')
        ax3.legend()
        ax3.grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_dir / 'training_metrics.png')
        plt.close()
        
        # Save raw metrics to CSV
        self.save_raw_metrics(output_dir)
    
    def save_raw_metrics(self, output_dir):
        """Save metrics to CSV file."""
        import pandas as pd
        from pathlib import Path
        
        output_dir = Path(output_dir)
        
        # Create DataFrame with all metrics
        df = pd.DataFrame({
            'epoch': self.epochs,
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_mAP': self.train_maps,
            'val_mAP': self.val_maps,
            'train_f1': self.train_f1s,
            'val_f1': self.val_f1s,
        })
        
        # Save to CSV
        df.to_csv(output_dir / 'raw_metrics.csv', index=False)
        
        # Also save as JSON for easier parsing
        metrics_dict = {
            'epochs': self.epochs,
            'train_metrics': {
                'loss': self.train_losses,
                'mAP': self.train_maps,
                'f1': self.train_f1s
            },
            'val_metrics': {
                'loss': self.val_losses,
                'mAP': self.val_maps,
                'f1': self.val_f1s
            }
        }
        
        import json
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=4)

# Register MetricsTracker as a safe global
add_safe_globals([MetricsTracker])

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Object Detector')
    parser.add_argument('--checkpoint', type=str, default='best_model.pth',
                      help='Name of checkpoint file in checkpoints directory')
    parser.add_argument('--data-dir', type=str, default='datasets/SKU-110K',
                      help='Path to dataset directory')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                      help='Directory to save evaluation results')
    parser.add_argument('--num-images', type=int, default=100,
                      help='Number of images to evaluate')
    parser.add_argument('--confidence-threshold', type=float, default=0.1,
                      help='Confidence threshold for detection')
    parser.add_argument('--nms-threshold', type=float, default=0.5,
                      help='NMS IoU threshold')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to run evaluation on')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug output')
    return parser.parse_args()

def calculate_metrics(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """Calculate precision, recall, and F1 score."""
    if len(pred_boxes) == 0:
        if len(gt_boxes) == 0:
            return 1.0, 1.0, 1.0  # Perfect score for correct empty prediction
        return 0.0, 0.0, 0.0  # Zero score for missing all ground truth
    
    if len(gt_boxes) == 0:
        return 0.0, 0.0, 0.0  # Zero score for false positives
        
    # Calculate IoU between all pred and gt boxes
    ious = box_iou(pred_boxes, gt_boxes)
    
    # For each pred box, get the best matching gt box
    max_ious, _ = ious.max(dim=1)
    
    # True positives are predictions that match a gt box with IoU > threshold
    true_positives = (max_ious >= iou_threshold).sum().item()
    
    precision = true_positives / len(pred_boxes)
    recall = true_positives / len(gt_boxes)
    
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0
        
    return precision, recall, f1

def calculate_map(pred_boxes_list, pred_scores_list, gt_boxes_list, iou_thresholds=[0.5]):
    """Calculate mean Average Precision."""
    aps = []
    
    for iou_threshold in iou_thresholds:
        # Collect all predictions and ground truths
        all_pred_scores = []
        all_true_positives = []
        total_gt_boxes = sum(len(gt_boxes) for gt_boxes in gt_boxes_list)
        
        for pred_boxes, pred_scores, gt_boxes in zip(pred_boxes_list, pred_scores_list, gt_boxes_list):
            if len(pred_boxes) == 0:
                continue
                
            # Calculate IoU between predictions and ground truth
            ious = box_iou(pred_boxes, gt_boxes)
            
            # For each prediction, get the best matching gt box
            max_ious, _ = ious.max(dim=1)
            
            # True positives are predictions that match a gt box with IoU > threshold
            true_positives = (max_ious >= iou_threshold).float()
            
            all_pred_scores.extend(pred_scores.tolist())
            all_true_positives.extend(true_positives.tolist())
        
        if not all_pred_scores:
            aps.append(0.0)
            continue
            
        # Sort by confidence
        indices = np.argsort(all_pred_scores)[::-1]
        all_true_positives = np.array(all_true_positives)[indices]
        
        # Compute precision and recall
        cumsum = np.cumsum(all_true_positives)
        precisions = cumsum / (np.arange(len(cumsum)) + 1)
        recalls = cumsum / total_gt_boxes
        
        # Compute average precision
        ap = 0
        for r in np.arange(0, 1.1, 0.1):
            if len(recalls) > 0:
                prec = np.max(precisions[recalls >= r]) if any(recalls >= r) else 0
                ap += prec / 11
        aps.append(ap)
    
    return np.mean(aps)

def visualize_detections(image, pred_boxes, pred_scores, gt_boxes, output_path, resize_size=None):
    """Visualize detection results with memory-efficient handling of large images."""
    try:
        import gc
        import matplotlib.pyplot as plt
        import os
        import torch
        
        # Ensure inputs are valid tensors or convert to empty tensors if None
        if pred_boxes is None or len(pred_boxes) == 0:
            pred_boxes = torch.zeros((0, 4), dtype=torch.float32)
        if pred_scores is None or len(pred_scores) == 0:
            pred_scores = torch.zeros(0, dtype=torch.float32)
        if gt_boxes is None or len(gt_boxes) == 0:
            gt_boxes = torch.zeros((0, 4), dtype=torch.float32)
            
        # Get dimensions from the transformed image
        if isinstance(image, torch.Tensor):
            C, H, W = image.shape
        else:
            print(f"Warning: Image is not a tensor, shape might be incorrect")
            H, W = 1, 1  # Default values
        
        # Convert boxes to tensors if they're not already
        if not isinstance(pred_boxes, torch.Tensor):
            pred_boxes = torch.tensor(pred_boxes, dtype=torch.float32)
        if not isinstance(gt_boxes, torch.Tensor):
            gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)
        if not isinstance(pred_scores, torch.Tensor):
            pred_scores = torch.tensor(pred_scores, dtype=torch.float32)
        
        # Ensure boxes are 2D tensors [N, 4]
        if pred_boxes.ndim == 3:
            pred_boxes = pred_boxes.squeeze(0)
        if gt_boxes.ndim == 3:
            gt_boxes = gt_boxes.squeeze(0)
        if pred_scores.ndim > 1:
            pred_scores = pred_scores.squeeze()
            
        # Verify shapes match
        if len(pred_boxes) != len(pred_scores):
            print(f"Warning: Mismatch between pred_boxes ({len(pred_boxes)}) and pred_scores ({len(pred_scores)})")
            # Take the minimum length to avoid index errors
            min_len = min(len(pred_boxes), len(pred_scores))
            pred_boxes = pred_boxes[:min_len]
            pred_scores = pred_scores[:min_len]
        
        # If boxes are normalized and we have resize dimensions, scale boxes
        if resize_size is not None and gt_boxes.numel() > 0 and gt_boxes.max() <= 1.0:
            resize_h, resize_w = resize_size
            print(f"[Info] Rescaling boxes to match resized image {resize_w}x{resize_h}")
            
            # Scale ground truth boxes
            if gt_boxes.numel() > 0:
                gt_boxes = gt_boxes.clone()
                gt_boxes[:, [0, 2]] *= resize_w
                gt_boxes[:, [1, 3]] *= resize_h
            
            # Scale predicted boxes
            if pred_boxes.numel() > 0:
                pred_boxes = pred_boxes.clone()
                pred_boxes[:, [0, 2]] *= resize_w
                pred_boxes[:, [1, 3]] *= resize_h
        
        # Calculate figure size
        aspect_ratio = W / H
        if aspect_ratio > 1:
            fig_width = min(6, aspect_ratio * 4)
            fig_height = fig_width / aspect_ratio
        else:
            fig_height = min(6, 4 / aspect_ratio)
            fig_width = fig_height * aspect_ratio

        scale_factor = 2.5
        fig_width = fig_width * scale_factor
        fig_height = fig_height * scale_factor
            
        # Create figure
        plt.figure(figsize=(fig_width, fig_height), dpi=120)
        
        # Process image
        try:
            with torch.no_grad():
                if isinstance(image, torch.Tensor):
                    image = torch.clamp(image * 0.5 + 0.5, 0, 1)
                    image = image.cpu().numpy()
                image = image.transpose(1, 2, 0)
                plt.imshow(image)
        except Exception as e:
            print(f"Warning: Error displaying image: {str(e)}")
            # Create blank image as fallback
            plt.imshow(np.zeros((H, W, 3)))
        
        # Plot ground truth boxes (green)
        gt_handle = None
        if gt_boxes.numel() > 0:
            for box in gt_boxes:
                try:
                    x1, y1, x2, y2 = box.tolist()
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='g', 
                                    linewidth=2.5, alpha=0.9)
                    plt.gca().add_patch(rect)
                    if gt_handle is None:
                        gt_handle = rect
                except Exception as e:
                    print(f"Warning: Error plotting GT box {box}: {str(e)}")
                    continue
        
        # Plot predicted boxes (red)
        pred_handle = None
        if pred_boxes.numel() > 0 and pred_scores.numel() > 0:
            for box, score in zip(pred_boxes, pred_scores):
                try:
                    x1, y1, x2, y2 = box.tolist()
                    score_val = score.item() if isinstance(score, torch.Tensor) else float(score)
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='r',
                                    linewidth=2.5, alpha=0.9)
                    plt.gca().add_patch(rect)
                    plt.text(x1, y1-2, f'{score_val:.2f}', color='red', fontsize=12,
                            bbox=dict(facecolor='white', alpha=0.9, pad=0))
                    if pred_handle is None:
                        pred_handle = rect
                except Exception as e:
                    print(f"Warning: Error plotting pred box {box}: {str(e)}")
                    continue
        
        # Add legend
        legend_elements = []
        legend_labels = []
        if gt_handle is not None:
            legend_elements.append(gt_handle)
            legend_labels.append(f'GT ({len(gt_boxes)})')
        if pred_handle is not None:
            legend_elements.append(pred_handle)
            legend_labels.append(f'Pred ({len(pred_boxes)})')
        
        if legend_elements:
            plt.legend(legend_elements, legend_labels, loc='upper right', fontsize=18)
        plt.title(f'Detections: {len(pred_boxes)} pred, {len(gt_boxes)} gt', fontsize=10)
        plt.axis('off')
        
        # Save figure
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, 
                    dpi=72,
                    bbox_inches='tight',
                    pad_inches=0.1,
                    format='png',
                    transparent=False,
                    facecolor='white',
                    edgecolor='none',
                    bbox_extra_artists=None)
            print(f"Successfully saved visualization to: {output_path}")
        except Exception as e:
            print(f"Error saving visualization: {str(e)}")
        finally:
            # Cleanup
            plt.close('all')
            gc.collect()
        
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        plt.close('all')
        gc.collect()
        return

def preprocess_boxes(boxes, name="boxes"):
    """Safely preprocess boxes tensor to ensure [N, 4] shape."""
    if boxes is None:
        print(f"Warning: {name} is None")
        return torch.empty((0, 4), device='cpu')
        
    if not isinstance(boxes, torch.Tensor):
        print(f"Warning: {name} is not a tensor, got {type(boxes)}")
        return torch.empty((0, 4), device='cpu')
    
    # Move to CPU for analysis
    boxes = boxes.cpu()
    
    # Handle different shapes
    if boxes.ndim == 3:
        boxes = boxes.squeeze(0)
    elif boxes.ndim == 1:
        if boxes.numel() == 4:
            boxes = boxes.view(1, 4)
        else:
            print(f"Warning: {name} has invalid shape: {boxes.shape}")
            return torch.empty((0, 4), device='cpu')
    
    # Verify final shape
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        print(f"Warning: {name} has invalid final shape: {boxes.shape}")
        return torch.empty((0, 4), device='cpu')
    
    return boxes

def analyze_boxes(pred_boxes, gt_boxes, prefix=""):
    """Analyze box coordinate ranges and distributions with detailed statistics."""
    print(f"\n{prefix} Box Analysis:")
    
    # Preprocess boxes
    pred_boxes = preprocess_boxes(pred_boxes, "pred_boxes")
    gt_boxes = preprocess_boxes(gt_boxes, "gt_boxes")
    
    def analyze_box_set(boxes, name):
        if boxes is None or boxes.numel() == 0 or boxes.shape[0] == 0:
            print(f"{name}: No boxes found")
            return None
            
        try:
            # Coordinate ranges
            x1, y1 = boxes[:, 0], boxes[:, 1]
            x2, y2 = boxes[:, 2], boxes[:, 3]
            widths = x2 - x1
            heights = y2 - y1
            areas = widths * heights
            aspect_ratios = widths / (heights + 1e-6)  # Avoid division by zero
            
            stats = {
                'count': len(boxes),
                'widths': widths,
                'heights': heights,
                'areas': areas,
                'aspect_ratios': aspect_ratios
            }
            
            print(f"\n{name}:")
            print(f"- Count: {len(boxes)}")
            print(f"- Coordinate ranges:")
            print(f"  x1: [{x1.min():.4f}, {x1.max():.4f}] (mean: {x1.mean():.4f})")
            print(f"  y1: [{y1.min():.4f}, {y1.max():.4f}] (mean: {y1.mean():.4f})")
            print(f"  x2: [{x2.min():.4f}, {x2.max():.4f}] (mean: {x2.mean():.4f})")
            print(f"  y2: [{y2.min():.4f}, {y2.max():.4f}] (mean: {y2.mean():.4f})")
            
            print(f"- Box dimensions:")
            print(f"  widths:  [{widths.min():.4f}, {widths.max():.4f}] (mean: {widths.mean():.4f})")
            print(f"  heights: [{heights.min():.4f}, {heights.max():.4f}] (mean: {heights.mean():.4f})")
            print(f"  areas:   [{areas.min():.4f}, {areas.max():.4f}] (mean: {areas.mean():.4f})")
            print(f"  aspects: [{aspect_ratios.min():.4f}, {aspect_ratios.max():.4f}] (mean: {aspect_ratios.mean():.4f})")
            
            # Validation checks
            invalid_boxes = (x2 <= x1) | (y2 <= y1)
            out_of_range = (x1 < 0) | (x1 > 1) | (y1 < 0) | (y1 > 1) | (x2 < 0) | (x2 > 1) | (y2 < 0) | (y2 > 1)
            
            if invalid_boxes.any():
                print(f"\n⚠️ Warning: Found {invalid_boxes.sum()} invalid boxes (x2 <= x1 or y2 <= y1)")
                invalid_indices = torch.where(invalid_boxes)[0]
                print(f"Invalid box indices: {invalid_indices.tolist()}")
                
            if out_of_range.any():
                print(f"\n⚠️ Warning: Found {out_of_range.sum()} boxes with coordinates outside [0,1] range")
                out_indices = torch.where(out_of_range)[0]
                print(f"Out of range box indices: {out_indices.tolist()}")
                
            return stats
        except Exception as e:
            print(f"Error analyzing {name}: {str(e)}")
            return None
    
    pred_stats = analyze_box_set(pred_boxes, "Predicted Boxes")
    gt_stats = analyze_box_set(gt_boxes, "Ground Truth Boxes")
    
    # Compare predictions with ground truth if both exist
    if pred_stats and gt_stats:
        try:
            print("\nBox Size Comparison:")
            print(f"- Width ratio (pred/gt): {pred_stats['widths'].mean() / (gt_stats['widths'].mean() + 1e-8):.2f}x")
            print(f"- Height ratio (pred/gt): {pred_stats['heights'].mean() / (gt_stats['heights'].mean() + 1e-8):.2f}x")
            print(f"- Area ratio (pred/gt): {pred_stats['areas'].mean() / (gt_stats['areas'].mean() + 1e-8):.2f}x")
            
            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                # Calculate IoUs between predictions and ground truth
                ious = box_iou(pred_boxes, gt_boxes)
                max_ious, _ = ious.max(dim=1)
                print(f"\nIoU Analysis:")
                print(f"- Max IoU: {max_ious.max():.4f}")
                print(f"- Mean IoU: {max_ious.mean():.4f}")
                print(f"- IoU > 0.5: {(max_ious > 0.5).sum()}/{len(pred_boxes)} boxes")
                print(f"- IoU > 0.3: {(max_ious > 0.3).sum()}/{len(pred_boxes)} boxes")
                
                # Analyze IoU distribution
                iou_thresholds = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
                for thresh in iou_thresholds:
                    matches = (max_ious > thresh).sum()
                    print(f"- IoU > {thresh:.1f}: {matches}/{len(pred_boxes)} boxes ({(matches/len(pred_boxes)*100):.1f}%)")
        except Exception as e:
            print(f"Error comparing boxes: {str(e)}")

def load_model(args):
    """Load model from checkpoint."""
    print(f"Loading checkpoint from: {args.checkpoint}")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Create model
    model = ObjectDetector().to(args.device)
    
    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Clean the state dict to match model keys
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        # Remove 'module.' prefix if it exists (from DataParallel)
        if k.startswith('module.'):
            k = k[7:]
        # Only keep model weights, skip optimizer etc.
        if not any(skip in k for skip in ['optimizer', 'scheduler', 'epoch', 'loss']):
            cleaned_state_dict[k] = v
    
    # Load state dict
    try:
        model.load_state_dict(cleaned_state_dict, strict=False)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Warning: Error loading state dict: {str(e)}")
        print("Continuing with partial weights...")
    
    return model

def plot_score_distribution(scores_list, output_path):
    """Plot distribution of confidence scores."""
    plt.figure(figsize=(10, 6))
    all_scores = [score.item() for scores in scores_list for score in scores]
    
    if all_scores:
        plt.hist(all_scores, bins=50, alpha=0.75)
        plt.axvline(x=np.mean(all_scores), color='r', linestyle='--', label=f'Mean: {np.mean(all_scores):.3f}')
        plt.axvline(x=np.median(all_scores), color='g', linestyle='--', label=f'Median: {np.median(all_scores):.3f}')
    
    plt.title('Distribution of Confidence Scores')
    plt.xlabel('Confidence Score')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_path, 'score_distribution.png'))
    plt.close()

def plot_box_size_distribution(pred_boxes_list, gt_boxes_list, output_path):
    """Plot distribution of predicted vs ground truth box sizes."""
    plt.figure(figsize=(12, 6))
    
    # Calculate widths and heights
    pred_widths = [(boxes[:, 2] - boxes[:, 0]).cpu().numpy() for boxes in pred_boxes_list if len(boxes) > 0]
    pred_heights = [(boxes[:, 3] - boxes[:, 1]).cpu().numpy() for boxes in pred_boxes_list if len(boxes) > 0]
    gt_widths = [(boxes[:, 2] - boxes[:, 0]).cpu().numpy() for boxes in gt_boxes_list if len(boxes) > 0]
    gt_heights = [(boxes[:, 3] - boxes[:, 1]).cpu().numpy() for boxes in gt_boxes_list if len(boxes) > 0]
    
    # Flatten lists
    pred_widths = np.concatenate(pred_widths) if pred_widths else np.array([])
    pred_heights = np.concatenate(pred_heights) if pred_heights else np.array([])
    gt_widths = np.concatenate(gt_widths) if gt_widths else np.array([])
    gt_heights = np.concatenate(gt_heights) if gt_heights else np.array([])
    
    plt.subplot(1, 2, 1)
    if len(pred_widths) > 0 and len(gt_widths) > 0:
        plt.hist(pred_widths, bins=30, alpha=0.5, label='Predicted', density=True)
        plt.hist(gt_widths, bins=30, alpha=0.5, label='Ground Truth', density=True)
    plt.title('Width Distribution')
    plt.xlabel('Normalized Width')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    if len(pred_heights) > 0 and len(gt_heights) > 0:
        plt.hist(pred_heights, bins=30, alpha=0.5, label='Predicted', density=True)
        plt.hist(gt_heights, bins=30, alpha=0.5, label='Ground Truth', density=True)
    plt.title('Height Distribution')
    plt.xlabel('Normalized Height')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'box_size_distribution.png'))
    plt.close()

def plot_aspect_ratios(pred_boxes_list, gt_boxes_list, output_path):
    """Plot aspect ratio distribution of predicted vs ground truth boxes."""
    plt.figure(figsize=(10, 6))
    
    # Calculate aspect ratios (width/height)
    pred_aspects = []
    gt_aspects = []
    
    for boxes in pred_boxes_list:
        if len(boxes) > 0:
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            aspects = (widths / heights).cpu().numpy()
            pred_aspects.extend(aspects)
    
    for boxes in gt_boxes_list:
        if len(boxes) > 0:
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            aspects = (widths / heights).cpu().numpy()
            gt_aspects.extend(aspects)
    
    if pred_aspects and gt_aspects:
        plt.hist(pred_aspects, bins=30, alpha=0.5, label='Predicted', density=True)
        plt.hist(gt_aspects, bins=30, alpha=0.5, label='Ground Truth', density=True)
    
    plt.title('Aspect Ratio Distribution')
    plt.xlabel('Aspect Ratio (width/height)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_path, 'aspect_ratio_distribution.png'))
    plt.close()

def plot_metrics_by_confidence(pred_boxes_list, pred_scores_list, gt_boxes_list, output_path):
    """Plot precision-recall curve and metrics vs confidence threshold."""
    confidence_thresholds = np.linspace(0.1, 0.9, 20)
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in confidence_thresholds:
        batch_precisions = []
        batch_recalls = []
        batch_f1s = []
        
        for pred_boxes, pred_scores, gt_boxes in zip(pred_boxes_list, pred_scores_list, gt_boxes_list):
            # Filter predictions by confidence threshold
            mask = pred_scores >= threshold
            filtered_boxes = pred_boxes[mask]
            filtered_scores = pred_scores[mask]
            
            # Calculate metrics
            precision, recall, f1 = calculate_metrics(filtered_boxes, filtered_scores, gt_boxes)
            batch_precisions.append(precision)
            batch_recalls.append(recall)
            batch_f1s.append(f1)
        
        precisions.append(np.mean(batch_precisions))
        recalls.append(np.mean(batch_recalls))
        f1_scores.append(np.mean(batch_f1s))
    
    # Plot metrics vs confidence threshold
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(confidence_thresholds, precisions, label='Precision')
    plt.plot(confidence_thresholds, recalls, label='Recall')
    plt.plot(confidence_thresholds, f1_scores, label='F1 Score')
    plt.title('Metrics vs Confidence Threshold')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(recalls, precisions)
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'metrics_by_confidence.png'))
    plt.close()

def plot_predictions_per_level(level_predictions, output_path):
    """Plot number of predictions and average confidence per FPN level."""
    levels = list(range(len(level_predictions)))
    num_predictions = [len(preds) for preds in level_predictions]
    avg_confidences = [preds['scores'].mean().item() if len(preds['scores']) > 0 else 0 for preds in level_predictions]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(levels, num_predictions)
    plt.title('Number of Predictions per Level')
    plt.xlabel('FPN Level')
    plt.ylabel('Number of Predictions')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(levels, avg_confidences)
    plt.title('Average Confidence per Level')
    plt.xlabel('FPN Level')
    plt.ylabel('Average Confidence')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'level_statistics.png'))
    plt.close()

def analyze_anchors(anchors, gt_boxes, level_idx):
    """Analyze anchor box statistics and their match with ground truth boxes."""
    print(f"\nAnchor Analysis for Level {level_idx}:")
    
    # Compute anchor box sizes
    anchor_widths = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]
    
    # Compute ground truth box sizes
    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
    
    print(f"Anchor size ranges:")
    print(f"- Width: [{anchor_widths.min():.4f}, {anchor_widths.max():.4f}]")
    print(f"- Height: [{anchor_heights.min():.4f}, {anchor_heights.max():.4f}]")
    
    print(f"\nGround truth size ranges:")
    print(f"- Width: [{gt_widths.min():.4f}, {gt_widths.max():.4f}]")
    print(f"- Height: [{gt_heights.min():.4f}, {gt_heights.max():.4f}]")
    
    # Compute IoU between anchors and ground truth boxes
    ious = box_iou(anchors, gt_boxes)
    max_ious, _ = ious.max(dim=1)
    
    print(f"\nIoU Statistics:")
    print(f"- Mean IoU: {max_ious.mean():.4f}")
    print(f"- Max IoU: {max_ious.max():.4f}")
    print(f"- Anchors with IoU > 0.5: {(max_ious > 0.5).sum()}/{len(anchors)}")
    
    return max_ious.mean().item()

def analyze_fpn_predictions(level_predictions, gt_boxes):
    """Analyze predictions from different FPN levels."""
    print("\nFPN Level Analysis:")
    
    for level_idx, preds in enumerate(level_predictions):
        if not preds:
            print(f"\nLevel {level_idx}: No predictions")
            continue
            
        boxes = preds['boxes']
        scores = preds['scores']
        
        if len(boxes) == 0:
            print(f"\nLevel {level_idx}: Empty predictions")
            continue
            
        # Calculate box statistics
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        areas = widths * heights
        
        print(f"\nLevel {level_idx}:")
        print(f"- Number of predictions: {len(boxes)}")
        print(f"- Score range: [{scores.min():.4f}, {scores.max():.4f}] (mean: {scores.mean():.4f})")
        print(f"- Box sizes:")
        print(f"  Width:  [{widths.min():.4f}, {widths.max():.4f}] (mean: {widths.mean():.4f})")
        print(f"  Height: [{heights.min():.4f}, {heights.max():.4f}] (mean: {heights.mean():.4f})")
        print(f"  Area:   [{areas.min():.4f}, {areas.max():.4f}] (mean: {areas.mean():.4f})")
        
        # Calculate IoUs with ground truth if available
        if len(gt_boxes) > 0:
            ious = box_iou(boxes, gt_boxes)
            max_ious, _ = ious.max(dim=1)
            print(f"- IoU stats:")
            print(f"  Max: {max_ious.max():.4f}")
            print(f"  Mean: {max_ious.mean():.4f}")
            print(f"  IoU > 0.5: {(max_ious > 0.5).sum()}/{len(boxes)}")
            
            # Find which predictions match small/medium/large GT boxes
            gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
            small_mask = gt_areas < 0.02  # Adjust thresholds as needed
            medium_mask = (gt_areas >= 0.02) & (gt_areas < 0.05)
            large_mask = gt_areas >= 0.05
            
            print("- GT box matches:")
            print(f"  Small boxes:  {(max_ious > 0.5).sum()}/{small_mask.sum()} ({(max_ious > 0.5).sum()/max(small_mask.sum(), 1)*100:.1f}%)")
            print(f"  Medium boxes: {(max_ious > 0.5).sum()}/{medium_mask.sum()} ({(max_ious > 0.5).sum()/max(medium_mask.sum(), 1)*100:.1f}%)")
            print(f"  Large boxes:  {(max_ious > 0.5).sum()}/{large_mask.sum()} ({(max_ious > 0.5).sum()/max(large_mask.sum(), 1)*100:.1f}%)")

def safe_get_predictions(predictions, device):
    """Safely extract predictions from model output with validation."""
    try:
        # Check if predictions is a dictionary
        if not isinstance(predictions, dict):
            raise ValueError(f"Expected dict, got {type(predictions)}")
            
        # Check if detections key exists
        if 'detections' not in predictions:
            raise ValueError("No 'detections' key in predictions")
            
        # Get first batch of detections
        detection = predictions['detections'][0]
        
        # Validate detection dictionary
        if not isinstance(detection, dict):
            raise ValueError(f"Expected dict for detection, got {type(detection)}")
            
        required_keys = ['boxes', 'scores', 'labels']
        missing_keys = [k for k in required_keys if k not in detection]
        if missing_keys:
            raise ValueError(f"Missing required keys in detection: {missing_keys}")
            
        # Extract predictions
        pred_boxes = detection['boxes']
        pred_scores = detection['scores']
        pred_labels = detection['labels']
        
        # Ensure tensors are on correct device
        pred_boxes = pred_boxes.to(device)
        pred_scores = pred_scores.to(device)
        pred_labels = pred_labels.to(device)
        
        # Validate tensor shapes
        if pred_boxes.ndim != 2 or pred_boxes.shape[-1] != 4:
            raise ValueError(f"Invalid box shape: {pred_boxes.shape}")
        if pred_scores.ndim != 1:
            raise ValueError(f"Invalid scores shape: {pred_scores.shape}")
        if pred_labels.ndim != 1:
            raise ValueError(f"Invalid labels shape: {pred_labels.shape}")
            
        # Ensure same number of boxes, scores, and labels
        if not (len(pred_boxes) == len(pred_scores) == len(pred_labels)):
            raise ValueError(
                f"Mismatched lengths: boxes={len(pred_boxes)}, "
                f"scores={len(pred_scores)}, labels={len(pred_labels)}"
            )
            
        return pred_boxes, pred_scores, pred_labels
        
    except Exception as e:
        print(f"Error extracting predictions: {str(e)}")
        # Return empty tensors as fallback
        return (
            torch.empty((0, 4), device=device),
            torch.empty(0, device=device),
            torch.empty(0, device=device)
        )

def validate_batch_data(batch, batch_idx):
    """Validate batch data structure and contents."""
    try:
        # Check if batch is a dictionary
        if not isinstance(batch, dict):
            raise ValueError(f"Expected dict for batch, got {type(batch)}")
            
        # Check required keys
        required_keys = ['images', 'bboxes', 'labels']
        missing_keys = [k for k in required_keys if k not in batch]
        if missing_keys:
            raise ValueError(f"Missing required keys in batch: {missing_keys}")
            
        # Validate images
        images = batch['images']
        if not isinstance(images, torch.Tensor):
            raise ValueError(f"Expected tensor for images, got {type(images)}")
        if images.ndim != 4:  # [batch_size, channels, height, width]
            raise ValueError(f"Invalid image shape: {images.shape}")
            
        # Validate boxes
        boxes = batch['bboxes']
        if not isinstance(boxes, list):
            raise ValueError(f"Expected list for boxes, got {type(boxes)}")
        if len(boxes) != len(images):
            raise ValueError(f"Mismatched batch sizes: images={len(images)}, boxes={len(boxes)}")
            
        # Validate labels
        labels = batch['labels']
        if not isinstance(labels, list):
            raise ValueError(f"Expected list for labels, got {type(labels)}")
        if len(labels) != len(images):
            raise ValueError(f"Mismatched batch sizes: images={len(images)}, labels={len(labels)}")
            
        # Validate each sample in batch
        for i in range(len(images)):
            # Validate box tensor
            if not isinstance(boxes[i], torch.Tensor):
                raise ValueError(f"Expected tensor for boxes[{i}], got {type(boxes[i])}")
            if boxes[i].ndim != 2 or boxes[i].shape[-1] != 4:
                raise ValueError(f"Invalid box shape at index {i}: {boxes[i].shape}")
                
            # Validate label tensor
            if not isinstance(labels[i], torch.Tensor):
                raise ValueError(f"Expected tensor for labels[{i}], got {type(labels[i])}")
            if labels[i].ndim != 1:
                raise ValueError(f"Invalid label shape at index {i}: {labels[i].shape}")
                
            # Validate matching lengths
            if len(boxes[i]) != len(labels[i]):
                raise ValueError(
                    f"Mismatched lengths at index {i}: "
                    f"boxes={len(boxes[i])}, labels={len(labels[i])}"
                )
                
            # Validate box coordinates
            if boxes[i].numel() > 0:
                if not (0 <= boxes[i].min() and boxes[i].max() <= 1):
                    raise ValueError(
                        f"Box coordinates out of range [0,1] at index {i}: "
                        f"min={boxes[i].min():.3f}, max={boxes[i].max():.3f}"
                    )
                
                # Check for invalid boxes
                widths = boxes[i][:, 2] - boxes[i][:, 0]
                heights = boxes[i][:, 3] - boxes[i][:, 1]
                if (widths <= 0).any() or (heights <= 0).any():
                    raise ValueError(f"Invalid box dimensions at index {i}")
        
        return True
        
    except Exception as e:
        print(f"Error validating batch {batch_idx}: {str(e)}")
        return False

def analyze_predictions(pred_boxes, pred_scores, gt_boxes, output_dir, prefix=""):
    """Comprehensive analysis of model predictions."""
    analysis_dir = Path(output_dir) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 1. Analyze spatial distribution
        analyze_spatial_distribution(pred_boxes, gt_boxes, analysis_dir, prefix)
        
        # 2. Analyze confidence scores
        analyze_confidence_distribution(pred_boxes, pred_scores, gt_boxes, analysis_dir, prefix)
        
        # 3. Analyze size distribution
        analyze_size_distribution(pred_boxes, gt_boxes, analysis_dir, prefix)
        
        # 4. Analyze IoU metrics
        analyze_iou_metrics(pred_boxes, pred_scores, gt_boxes, analysis_dir, prefix)
        
        # Save summary statistics
        stats = compute_summary_statistics(pred_boxes, pred_scores, gt_boxes)
        with open(analysis_dir / f"{prefix}summary_stats.json", 'w') as f:
            json.dump(stats, f, indent=4)
            
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

def analyze_spatial_distribution(pred_boxes, gt_boxes, output_dir, prefix=""):
    """Analyze spatial distribution of predictions vs ground truth."""
    plt.figure(figsize=(15, 5))
    
    # Plot prediction centers
    pred_centers_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_centers_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    
    plt.subplot(1, 3, 1)
    plt.hist2d(pred_centers_x.cpu(), pred_centers_y.cpu(), 
              bins=30, range=[[0, 1], [0, 1]])
    plt.colorbar()
    plt.title("Prediction Centers")
    
    # Plot ground truth centers
    if len(gt_boxes) > 0:
        gt_centers_x = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2
        gt_centers_y = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2
        
        plt.subplot(1, 3, 2)
        plt.hist2d(gt_centers_x.cpu(), gt_centers_y.cpu(), 
                  bins=30, range=[[0, 1], [0, 1]])
        plt.colorbar()
        plt.title("Ground Truth Centers")
        
        # Plot center offset distribution
        plt.subplot(1, 3, 3)
        center_offsets_x = pred_centers_x.unsqueeze(1) - gt_centers_x.unsqueeze(0)
        center_offsets_y = pred_centers_y.unsqueeze(1) - gt_centers_y.unsqueeze(0)
        plt.hist2d(center_offsets_x.cpu().flatten(), center_offsets_y.cpu().flatten(), 
                  bins=30, range=[[-0.5, 0.5], [-0.5, 0.5]])
        plt.colorbar()
        plt.title("Center Offsets")
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}spatial_distribution.png")
    plt.close()

def analyze_confidence_distribution(pred_boxes, pred_scores, gt_boxes, output_dir, prefix=""):
    """Analyze confidence score distribution and correlation with other metrics."""
    plt.figure(figsize=(15, 5))
    
    # Score distribution
    plt.subplot(1, 3, 1)
    plt.hist(pred_scores.cpu(), bins=50, range=(0, 1))
    plt.title("Confidence Distribution")
    plt.xlabel("Score")
    plt.ylabel("Count")
    
    # Score vs box size
    box_sizes = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    plt.subplot(1, 3, 2)
    plt.scatter(pred_scores.cpu(), box_sizes.cpu(), alpha=0.5)
    plt.title("Score vs Box Size")
    plt.xlabel("Confidence")
    plt.ylabel("Box Size")
    
    # Score vs IoU (if ground truth available)
    if len(gt_boxes) > 0:
        ious = box_iou(pred_boxes, gt_boxes)
        max_ious, _ = ious.max(dim=1)
        plt.subplot(1, 3, 3)
        plt.scatter(pred_scores.cpu(), max_ious.cpu(), alpha=0.5)
        plt.title("Score vs IoU")
        plt.xlabel("Confidence")
        plt.ylabel("Max IoU")
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}confidence_analysis.png")
    plt.close()

def analyze_size_distribution(pred_boxes, gt_boxes, output_dir, prefix=""):
    """Analyze size distribution of predictions vs ground truth."""
    plt.figure(figsize=(15, 5))
    
    # Prediction size distribution
    pred_widths = pred_boxes[:, 2] - pred_boxes[:, 0]
    pred_heights = pred_boxes[:, 3] - pred_boxes[:, 1]
    
    plt.subplot(1, 3, 1)
    plt.hist2d(pred_widths.cpu(), pred_heights.cpu(), 
              bins=30, range=[[0, 1], [0, 1]])
    plt.colorbar()
    plt.title("Prediction Sizes")
    plt.xlabel("Width")
    plt.ylabel("Height")
    
    # Ground truth size distribution
    if len(gt_boxes) > 0:
        gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0]
        gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1]
        
        plt.subplot(1, 3, 2)
        plt.hist2d(gt_widths.cpu(), gt_heights.cpu(), 
                  bins=30, range=[[0, 1], [0, 1]])
        plt.colorbar()
        plt.title("Ground Truth Sizes")
        plt.xlabel("Width")
        plt.ylabel("Height")
        
        # Size ratio distribution
        plt.subplot(1, 3, 3)
        pred_areas = pred_widths * pred_heights
        gt_areas = gt_widths * gt_heights
        area_ratios = pred_areas.unsqueeze(1) / (gt_areas.unsqueeze(0) + 1e-6)
        plt.hist(area_ratios.cpu().flatten(), bins=50, range=(0, 4))
        plt.title("Area Ratios (Pred/GT)")
        plt.xlabel("Ratio")
        plt.ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}size_distribution.png")
    plt.close()

def analyze_iou_metrics(pred_boxes, pred_scores, gt_boxes, output_dir, prefix=""):
    """Analyze IoU metrics and matching quality."""
    if len(gt_boxes) == 0:
        return
        
    plt.figure(figsize=(15, 5))
    
    # IoU distribution
    ious = box_iou(pred_boxes, gt_boxes)
    max_ious, _ = ious.max(dim=1)
    
    plt.subplot(1, 3, 1)
    plt.hist(max_ious.cpu(), bins=50, range=(0, 1))
    plt.title("Max IoU Distribution")
    plt.xlabel("IoU")
    plt.ylabel("Count")
    
    # IoU vs confidence correlation
    plt.subplot(1, 3, 2)
    plt.scatter(pred_scores.cpu(), max_ious.cpu(), alpha=0.5)
    plt.title("IoU vs Confidence")
    plt.xlabel("Confidence")
    plt.ylabel("Max IoU")
    
    # IoU vs box size
    box_sizes = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    plt.subplot(1, 3, 3)
    plt.scatter(box_sizes.cpu(), max_ious.cpu(), alpha=0.5)
    plt.title("IoU vs Box Size")
    plt.xlabel("Box Size")
    plt.ylabel("Max IoU")
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{prefix}iou_analysis.png")
    plt.close()

def compute_summary_statistics(pred_boxes, pred_scores, gt_boxes):
    """Compute summary statistics for predictions."""
    stats = {
        'num_predictions': len(pred_boxes),
        'num_gt_boxes': len(gt_boxes),
        'mean_confidence': pred_scores.mean().item(),
        'high_confidence_ratio': (pred_scores > 0.5).float().mean().item(),
        'pred_size_stats': {
            'mean_width': (pred_boxes[:, 2] - pred_boxes[:, 0]).mean().item(),
            'mean_height': (pred_boxes[:, 3] - pred_boxes[:, 1]).mean().item(),
            'mean_area': ((pred_boxes[:, 2] - pred_boxes[:, 0]) * 
                         (pred_boxes[:, 3] - pred_boxes[:, 1])).mean().item()
        }
    }
    
    if len(gt_boxes) > 0:
        ious = box_iou(pred_boxes, gt_boxes)
        max_ious, _ = ious.max(dim=1)
        stats.update({
            'iou_stats': {
                'mean_iou': max_ious.mean().item(),
                'median_iou': max_ious.median().item(),
                'num_good_matches': (max_ious > 0.5).sum().item()
            }
        })
    
    return stats

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and dataset
    try:
        model = load_model(args)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    print("Loading dataset...")
    try:
        from v1.utils.augmentation import DetectionAugmentation
        
        # Create augmentation with test transforms
        augmentation = DetectionAugmentation(height=640, width=640)
        
        dataset = SKU110KDataset(
            data_dir=args.data_dir,
            split='test',
            transform=augmentation
        )
        
        # Limit dataset size if specified
        if args.num_images > 0:
            dataset.image_ids = dataset.image_ids[:args.num_images]
            print(f"Using {args.num_images} images for evaluation")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return
    
    # Initialize lists to store predictions and ground truth
    all_pred_boxes = []
    all_pred_scores = []
    all_gt_boxes = []
    level_predictions = [[] for _ in range(4)]  # For 4 FPN levels
    mean_ious_per_level = []
    
    print("Starting evaluation...")
    
    with torch.no_grad():
        for i in tqdm(range(min(args.num_images, len(dataset)))):
            try:
                # Get image and targets
                data = dataset[i]
                if not isinstance(data, dict):
                    print(f"Warning: Dataset returned non-dict type for index {i}")
                    continue
                    
                # Validate required keys
                required_keys = ['image', 'boxes', 'image_id']
                if not all(k in data for k in required_keys):
                    print(f"Warning: Missing required keys in data: {[k for k in required_keys if k not in data]}")
                    continue
                
                # Ensure image is a tensor and move to device
                if not isinstance(data['image'], torch.Tensor):
                    print(f"Warning: Image is not a tensor for index {i}")
                    continue
                image = data['image'].unsqueeze(0).to(args.device)
                
                # Validate and process boxes
                if not isinstance(data['boxes'], torch.Tensor):
                    print(f"Warning: Boxes is not a tensor for index {i}")
                    continue
                gt_boxes = data['boxes'].to(args.device)
                
                # Skip if no ground truth boxes
                if gt_boxes.numel() == 0:
                    print(f"Warning: No ground truth boxes for image {i}")
                    continue
                
                # Run inference with error handling
                try:
                    predictions = model(image)
                except Exception as e:
                    print(f"Error during model inference for image {i}: {str(e)}")
                    continue
                
                # Validate predictions structure
                if not isinstance(predictions, dict) or 'detections' not in predictions:
                    print(f"Warning: Invalid prediction structure for image {i}")
                    continue
                
                # Extract and validate predictions
                pred_boxes, pred_scores, pred_labels = safe_get_predictions(predictions, device=args.device)
                
                if pred_boxes.shape[0] == 0:
                    print(f"No valid predictions for image {i}")
                    all_pred_boxes.append(pred_boxes)
                    all_pred_scores.append(pred_scores)
                    all_gt_boxes.append(gt_boxes)
                    continue
                
                # Move tensors to CPU for post-processing
                pred_boxes = pred_boxes.cpu()
                pred_scores = pred_scores.cpu()
                pred_labels = pred_labels.cpu()
                gt_boxes = gt_boxes.cpu()
                
                # Apply NMS with debug info
                if len(pred_boxes) > 0:
                    # Apply confidence threshold before NMS
                    conf_mask = pred_scores > args.confidence_threshold
                    if not conf_mask.any():
                        print(f"No predictions above confidence threshold for image {i}")
                        continue
                        
                    pred_boxes = pred_boxes[conf_mask]
                    pred_scores = pred_scores[conf_mask]
                    pred_labels = pred_labels[conf_mask]
                    
                    # Apply NMS
                    try:
                        keep = nms(pred_boxes, pred_scores, args.nms_threshold)
                        pred_boxes = pred_boxes[keep]
                        pred_scores = pred_scores[keep]
                        pred_labels = pred_labels[keep]
                    except Exception as e:
                        print(f"Error during NMS for image {i}: {str(e)}")
                        continue
                
                # Store predictions and ground truth
                all_pred_boxes.append(pred_boxes)
                all_pred_scores.append(pred_scores)
                all_gt_boxes.append(gt_boxes)
                
                # Visualize first 5 images with error handling
                if i < 5:
                    try:
                        output_path = os.path.join(args.output_dir, f'detection_{i}.png')
                        resize_size = data.get('resize_size', None)
                        visualize_detections(
                            image=image[0].cpu(),
                            pred_boxes=pred_boxes,
                            pred_scores=pred_scores,
                            gt_boxes=gt_boxes,
                            output_path=output_path,
                            resize_size=resize_size
                        )
                    except Exception as e:
                        print(f"Error visualizing image {i}: {str(e)}")
                        continue
                
                # Add analysis every N images
                if i % 10 == 0:  # Analyze every 10 images
                    analyze_predictions(
                        pred_boxes,
                        pred_scores,
                        gt_boxes,
                        args.output_dir,
                        prefix=f"image_{i}_"
                    )
                
                # Clear memory
                del image, predictions, pred_boxes, pred_scores, gt_boxes
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing image {i}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    # Calculate mean metrics
    metrics = []
    for pred_boxes, pred_scores, gt_boxes in zip(all_pred_boxes, all_pred_scores, all_gt_boxes):
        if len(pred_boxes) > 0 and len(pred_scores) > 0:
            precision, recall, f1 = calculate_metrics(pred_boxes, pred_scores, gt_boxes, args.confidence_threshold)
            metrics.append((precision, recall, f1))
    
    if metrics:
        mean_precision = np.mean([m[0] for m in metrics])
        mean_recall = np.mean([m[1] for m in metrics])
        mean_f1 = np.mean([m[2] for m in metrics])
    else:
        mean_precision = 0.0
        mean_recall = 0.0
        mean_f1 = 0.0
    
    # Calculate mAP
    mAP = calculate_map(all_pred_boxes, all_pred_scores, all_gt_boxes)
    
    # Print statistics
    print("\nPrediction Statistics:")
    print(f"Total images processed: {len(all_pred_boxes)}")
    print(f"Images with valid predictions: {len([boxes for boxes in all_pred_boxes if len(boxes) > 0])}")
    print(f"Average predictions per image: {np.mean([len(boxes) for boxes in all_pred_boxes]):.2f}")
    
    print("\nEvaluation Results:")
    print(f"mAP: {mAP:.4f}")
    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Recall: {mean_recall:.4f}")
    print(f"Mean F1: {mean_f1:.4f}")
    
    # Save results
    results = {
        'mAP': float(mAP),
        'mean_precision': float(mean_precision),
        'mean_recall': float(mean_recall),
        'mean_f1': float(mean_f1),
        'total_images': len(all_pred_boxes),
        'images_with_predictions': len([boxes for boxes in all_pred_boxes if len(boxes) > 0]),
        'avg_predictions_per_image': float(np.mean([len(boxes) for boxes in all_pred_boxes]))
    }
    
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to {args.output_dir}")

    # After processing all images, generate additional visualizations
    plot_score_distribution(all_pred_scores, args.output_dir)
    plot_box_size_distribution(all_pred_boxes, all_gt_boxes, args.output_dir)
    plot_aspect_ratios(all_pred_boxes, all_gt_boxes, args.output_dir)
    plot_metrics_by_confidence(all_pred_boxes, all_pred_scores, all_gt_boxes, args.output_dir)
    
    # Plot level-specific statistics if we have them
    if any(len(preds) > 0 for preds in level_predictions):
        plot_predictions_per_level(level_predictions, args.output_dir)

if __name__ == '__main__':
    main() 