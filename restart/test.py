# Imports.
import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from torchvision.ops import box_iou
from torch.utils.data import DataLoader

# Local Imports.
from restart.data.dataset import SKU110KDataset
from restart.model.detector import ObjectDetector
from restart.utils.visualize_detections import visualize_detections
from restart.utils.box_ops import normalize_boxes
from restart.utils.plots import plot_precision_recall_curve, plot_map_progress, plot_iou_histogram

# Debug.
debug = False

def save_metrics_summary(metrics, save_dir):
    """Save Metrics Summary as Text File."""
    # Save text summary
    summary_path = save_dir / 'metrics_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Evaluation Results:\n")
        f.write("-" * 50 + "\n")
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                f.write(f"{metric_name:15s}: {value:.4f}\n")
            else:
                f.write(f"{metric_name:15s}: {value}\n")
        f.write("-" * 50 + "\n")

# Object Detection Evaluator Class.
class ObjectDetectionEvaluator:
    def __init__(self, iou_threshold=0.2):
        """Initialize Object Detection Evaluator."""
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        """Reset Metrics."""
        self.total_gt = 0
        self.total_pred = 0
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.total_iou = 0
        self.matched_detections = 0
        self.aps = []  # Store AP for Each Image.
        self.all_precisions = []  # Store All Precision Values.
        self.all_recalls = []     # Store All Recall Values.
        self.cumulative_precisions = []  # Store Cumulative Precision for PR Curve.
        self.cumulative_recalls = []     # Store Cumulative Recall for PR Curve.
        self.iou_list = []  # Store IoUs from matched detections
    
    def update(self, pred_boxes, pred_scores, gt_boxes, orig_size, resize_size):
        """Update Metrics for a Single Image."""
        
        # Ensure Boxes are Tensors.
        if not isinstance(pred_boxes, torch.Tensor):
            pred_boxes = torch.tensor(pred_boxes)
        if not isinstance(gt_boxes, torch.Tensor):
            gt_boxes = torch.tensor(gt_boxes)
            
        # Ensure Boxes are on the Same Device.
        if pred_boxes.device != gt_boxes.device:
            gt_boxes = gt_boxes.to(pred_boxes.device)
        
        # If No Predictions, Add False Negatives.
        if len(pred_boxes) == 0:
            self.false_negatives += len(gt_boxes)
            self.total_gt += len(gt_boxes)
            print(f"No predictions, adding {len(gt_boxes)} false negatives")
            return
        
        # If No Ground Truth, Add False Positives.
        if len(gt_boxes) == 0:
            self.false_positives += len(pred_boxes)
            self.total_pred += len(pred_boxes)
            print(f"No ground truth, adding {len(pred_boxes)} false positives")
            return

        # Scale GT Boxes to Match Prediction Space.
        gt_max = gt_boxes.max()
        if gt_max < 0.5:  # If GT Boxes are in a Much Smaller Range.
            scale_factor = 1.0 / gt_max
            gt_boxes = gt_boxes * scale_factor
            if debug :
                print(f"Scaling GT boxes by factor {scale_factor}")
        
        if debug :
            print(f"Pred boxes range: [{pred_boxes.min():.4f}, {pred_boxes.max():.4f}]")
            print(f"GT boxes range (after scaling): [{gt_boxes.min():.4f}, {gt_boxes.max():.4f}]")
        
        # Calculate IoU between All Pred and GT Boxes.
        ious = box_iou(pred_boxes, gt_boxes)
        if debug :
            print(f"IoU matrix shape: {ious.shape}, Max IoU: {ious.max().item():.4f}, Mean IoU: {ious.mean().item():.4f}")
        
        # Sort Predictions by Confidence.
        sorted_indices = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sorted_indices]
        pred_scores = pred_scores[sorted_indices]
        
        # Initialize Tracking Arrays.
        gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool, device=gt_boxes.device)
        pred_matched = torch.zeros(len(pred_boxes), dtype=torch.bool, device=pred_boxes.device)
        
        # Match Predictions to Ground Truth.
        matches_found = 0
        for pred_idx, pred_box in enumerate(pred_boxes):
            max_iou, max_idx = ious[sorted_indices[pred_idx]].max(dim=0)
            if max_iou >= self.iou_threshold and not gt_matched[max_idx]:
                self.true_positives += 1
                self.total_iou += max_iou.item()
                self.iou_list.append(max_iou.item())  # Track individual IoU values
                self.matched_detections += 1
                gt_matched[max_idx] = True
                pred_matched[pred_idx] = True
                matches_found += 1

                if debug :
                    print(f"Match found! IoU: {max_iou.item():.4f}")
                    print(f"Matched pred box: {pred_box.tolist()}")
                    print(f"Matched GT box: {gt_boxes[max_idx].tolist()}")
        
        if debug :
            print(f"Found {matches_found} matches with IoU >= {self.iou_threshold}")
        
        # Calculate Metrics.
        new_fps = (~pred_matched).sum().item()
        new_fns = (~gt_matched).sum().item()

        # Update Metrics.
        self.false_positives += new_fps
        self.false_negatives += new_fns
        self.total_gt += len(gt_boxes)
        self.total_pred += len(pred_boxes)
        
        if debug :
            print(f"Image results: TP={matches_found}, FP={new_fps}, FN={new_fns}")
        
        # Calculate Precision-Recall Curve Points.
        running_tp = 0
        running_fp = 0
        precisions = []
        recalls = []
        
        # If Matched, Increment True Positives, Otherwise Increment False Positives.
        for matched in pred_matched:
            if matched:
                running_tp += 1
            else:
                running_fp += 1
            
            precision = running_tp / (running_tp + running_fp)
            recall = running_tp / len(gt_boxes)
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Calculate AP using 11-Point Interpolation.
        if precisions:
            ap = 0
            for t in np.arange(0, 1.1, 0.1):
                if np.sum(recalls >= t) == 0:
                    p = 0
                else:
                    p = np.max(np.array(precisions)[np.array(recalls) >= t])
                ap += p / 11
            self.aps.append(ap)
            if debug :
                print(f"Image AP: {ap:.4f}")
        
        # Store Precision-Recall Curve Points.
        self.cumulative_precisions.extend(precisions)
        self.cumulative_recalls.extend(recalls)
    
    def get_metrics(self):
        """Calculate and Return All Metrics."""
        eps = 1e-6  # Small epsilon to avoid division by zero
        
        precision = self.true_positives / (self.true_positives + self.false_positives + eps)
        recall = self.true_positives / (self.true_positives + self.false_negatives + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        avg_iou = self.total_iou / (self.matched_detections + eps)
        mAP = np.mean(self.aps) if self.aps else 0
        
        return {
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Average IoU': avg_iou,
            'mAP': mAP,
            'True Positives': self.true_positives,
            'False Positives': self.false_positives,
            'False Negatives': self.false_negatives,
            'Total GT': self.total_gt,
            'Total Pred': self.total_pred
        }

def evaluate_model(model_path, config_path, num_visualizations=5, max_images=100):
    """
    Evaluate a trained object detection model.
    
    Args:
        model_path: Path to the model checkpoint
        config_path: Path to the config file
        num_visualizations: Number of sample images to visualize
        max_images: Maximum number of images to evaluate (default: 100)
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directories with model name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(model_path).stem  # Gets filename without extension
    output_dir = Path(config['output_dir']) / f'eval_{model_name}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    metrics_dir = output_dir / 'metrics'
    metrics_dir.mkdir(exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test dataset
    test_dataset = SKU110KDataset(
        data_dir=config['dataset_path'],
        split='test',
        resize_dims=tuple(config['resize_dims'])
    )
    
    # Create a subset of the dataset
    if max_images and max_images < len(test_dataset):
        subset_indices = list(range(max_images))
        test_dataset = torch.utils.data.Subset(test_dataset, subset_indices)
        print(f"\nUsing subset of {max_images} images for evaluation")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=SKU110KDataset.collate_fn
    )
    
    # Initialize model
    model = ObjectDetector(
        pretrained_backbone=False,
        num_classes=1,
        num_anchors=9
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Initialize evaluator
    evaluator = ObjectDetectionEvaluator(iou_threshold=0.3)
    
    print("\nStarting evaluation...")
    total_images_processed = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader)):
            try:
                # Move batch to device
                images = batch['images'].to(device)
                gt_boxes = [b.to(device) for b in batch['boxes']]
                gt_labels = [l.to(device) for l in batch['labels']]
                orig_sizes = batch['orig_sizes']
                resize_sizes = batch['resize_sizes']
                
                # Get predictions
                outputs = model(images)
                
                # Update metrics for each image in batch
                for i in range(len(images)):
                    try:
                        pred_boxes = outputs['detections'][i]['boxes']
                        pred_scores = outputs['detections'][i]['scores']
                        
                        # Update metrics
                        evaluator.update(pred_boxes, pred_scores, gt_boxes[i], orig_sizes[i], resize_sizes[i])
                        
                        # Visualize some predictions
                        if batch_idx * config['batch_size'] + i < num_visualizations:
                            save_path = vis_dir / f'test_vis_{batch_idx}_{i}.png'
                            visualize_detections(
                                image_tensor=images[i],
                                detections=outputs['detections'][i],
                                ground_truths={'boxes': gt_boxes[i], 'labels': gt_labels[i]},
                                orig_size=orig_sizes[i],
                                resize_size=resize_sizes[i],
                                save_path=save_path,
                                title=f'Test Image {batch_idx}_{i}'
                            )
                            
                        total_images_processed += 1
                        if max_images and total_images_processed >= max_images:
                            break
                            
                    except Exception as e:
                        print(f"Error processing image {i} in batch {batch_idx}: {str(e)}")
                        continue
                        
                if max_images and total_images_processed >= max_images:
                    break
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {str(e)}")
                continue
    
    print(f"\nProcessed {total_images_processed} images")
    
    # Get metrics and create visualizations
    metrics = evaluator.get_metrics()
    
    # Save metrics summary and create visualizations
    save_metrics_summary(metrics, metrics_dir)
    
    # Plot precision-recall curve
    if evaluator.cumulative_precisions and evaluator.cumulative_recalls:
        plot_precision_recall_curve(
            evaluator.cumulative_precisions,
            evaluator.cumulative_recalls,
            metrics['mAP'],
            metrics_dir
        )
    
    # Plot mAP progress
    if evaluator.aps:
        plot_map_progress(evaluator.aps, metrics_dir)
    
    # Plot IoU histogram
    if evaluator.iou_list:
        plot_iou_histogram(evaluator.iou_list, metrics_dir)
    
    # Save config used for evaluation
    with open(output_dir / 'eval_config.yaml', 'w') as f:
        yaml.dump({
            'model_path': str(model_path),
            'config_path': str(config_path),
            'num_visualizations': num_visualizations,
            'max_images': max_images,
            'evaluation_timestamp': timestamp,
            'iou_threshold': evaluator.iou_threshold
        }, f)
    
    print("\nEvaluation Results:")
    print("-" * 50)
    for metric_name, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{metric_name:15s}: {value:.4f}")
        else:
            print(f"{metric_name:15s}: {value}")
    print("-" * 50)
    print(f"\nResults saved to: {output_dir}")
    
    return metrics

if __name__ == "__main__":
    # Load config first
    config_path = "restart/config/testing_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Evaluate using config parameters
    metrics = evaluate_model(
        model_path=config['checkpoint_path'],
        config_path=config_path,
        num_visualizations=config['num_visualizations'],
        max_images=config.get('subset_size', 100)
    )
