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

from src.model.detector import ObjectDetector
from src.data.dataset import SKU110KDataset
from src.utils.box_ops import box_iou
from torchvision.ops import nms

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
    parser.add_argument('--confidence-threshold', type=float, default=0.5,
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

def visualize_detections(image, pred_boxes, pred_scores, gt_boxes, output_path):
    """Visualize detection results."""
    plt.figure(figsize=(12, 8))
    
    # Denormalize image
    image = torch.clamp(image * 0.5 + 0.5, 0, 1)  # Assuming standard normalization
    plt.imshow(image.permute(1, 2, 0))
    
    H, W = image.shape[1:3]
    
    # Plot ground truth boxes first (green)
    gt_handle = None
    for box in gt_boxes:
        # Get normalized coordinates
        x1, y1, x2, y2 = box.cpu().numpy()
        # Convert to absolute pixel coordinates
        x1, x2 = x1 * W, x2 * W
        y1, y2 = y1 * H, y2 * H
        # Draw box
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='g', linewidth=2, alpha=0.5)
        plt.gca().add_patch(rect)
        if gt_handle is None:
            gt_handle = rect
    
    # Plot predicted boxes (red)
    pred_handle = None
    for box, score in zip(pred_boxes, pred_scores):
        # Get normalized coordinates
        x1, y1, x2, y2 = box.cpu().numpy()
        # Convert to absolute pixel coordinates
        x1, x2 = x1 * W, x2 * W
        y1, y2 = y1 * H, y2 * H
        # Draw box with confidence score
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='r', linewidth=1, alpha=min(score.item(), 0.9))
        plt.gca().add_patch(rect)
        plt.text(x1, y1-2, f'{score.item():.2f}', color='red', fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
        if pred_handle is None:
            pred_handle = rect
    
    # Add legend with counts
    legend_elements = []
    legend_labels = []
    if gt_handle is not None:
        legend_elements.append(gt_handle)
        legend_labels.append(f'Ground Truth ({len(gt_boxes)})')
    if pred_handle is not None:
        legend_elements.append(pred_handle)
        legend_labels.append(f'Predictions ({len(pred_boxes)}, conf>{min(pred_scores):.2f})')
    
    plt.legend(legend_elements, legend_labels, loc='upper right')
    
    plt.title(f'Object Detection Results\nGT: {len(gt_boxes)} boxes, Pred: {len(pred_boxes)} boxes')
    plt.axis('off')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# Add debug function to analyze box coordinates
def analyze_boxes(pred_boxes, gt_boxes, prefix=""):
    """Analyze box coordinate ranges and distributions."""
    print(f"\n{prefix} Box Analysis:")
    if len(pred_boxes) > 0:
        print("Predicted Boxes:")
        print(f"- Coordinate ranges: x1=[{pred_boxes[:,0].min():.3f}, {pred_boxes[:,0].max():.3f}], "
              f"y1=[{pred_boxes[:,1].min():.3f}, {pred_boxes[:,1].max():.3f}]")
        print(f"- Box sizes: width=[{(pred_boxes[:,2]-pred_boxes[:,0]).min():.3f}, "
              f"{(pred_boxes[:,2]-pred_boxes[:,0]).max():.3f}], "
              f"height=[{(pred_boxes[:,3]-pred_boxes[:,1]).min():.3f}, "
              f"{(pred_boxes[:,3]-pred_boxes[:,1]).max():.3f}]")
    
    if len(gt_boxes) > 0:
        print("\nGround Truth Boxes:")
        print(f"- Coordinate ranges: x1=[{gt_boxes[:,0].min():.3f}, {gt_boxes[:,0].max():.3f}], "
              f"y1=[{gt_boxes[:,1].min():.3f}, {gt_boxes[:,1].max():.3f}]")
        print(f"- Box sizes: width=[{(gt_boxes[:,2]-gt_boxes[:,0]).min():.3f}, "
              f"{(gt_boxes[:,2]-gt_boxes[:,0]).max():.3f}], "
              f"height=[{(gt_boxes[:,3]-gt_boxes[:,1]).min():.3f}, "
              f"{(gt_boxes[:,3]-gt_boxes[:,1]).max():.3f}]")

def load_model(args):
    """Load model from checkpoint."""
    # Initialize model
    model = ObjectDetector(
        pretrained_backbone=True,
        num_classes=3,
        debug=args.debug
    )
    
    # Find checkpoint
    checkpoint_dir = Path('checkpoints')
    if not checkpoint_dir.exists():
        raise FileNotFoundError("Checkpoints directory not found!")
    
    checkpoint_path = checkpoint_dir / args.checkpoint
    if not checkpoint_path.exists():
        # Try to find the latest checkpoint if specified checkpoint doesn't exist
        checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        checkpoint_path = max(checkpoints, key=lambda x: int(str(x).split('_')[-1].split('.')[0]))
        print(f"Using latest checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Load weights
    try:
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Warning: Error loading state dict: {str(e)}")
        # Try loading with strict=False
        model.load_state_dict(state_dict, strict=False)
        print("Loaded checkpoint with some missing or unexpected keys")
    
    model = model.to(args.device)
    model.eval()
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

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and dataset
    try:
        model = load_model(args)
        model.eval()  # Ensure model is in eval mode
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    print("Loading dataset...")
    try:
        from src.utils.augmentation import DetectionAugmentation
        
        # Create augmentation with test transforms
        augmentation = DetectionAugmentation(height=640, width=640)  # Reduced image size
        
        dataset = SKU110KDataset(
            data_dir=args.data_dir,
            split='test',
            transform=augmentation.val_transform
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
    
    print("Starting evaluation...")
    
    with torch.no_grad():
        for i in tqdm(range(min(args.num_images, len(dataset)))):
            try:
                # Get image and targets
                data = dataset[i]
                image = data['image'].unsqueeze(0).to(args.device)
                gt_boxes = data['boxes'].to(args.device)
                
                # Normalize ground truth boxes to [0, 1] range
                gt_boxes = gt_boxes.clone()  # Create a copy to avoid modifying original data
                gt_boxes[:, [0, 2]] /= image.shape[3]  # Normalize x coordinates
                gt_boxes[:, [1, 3]] /= image.shape[2]  # Normalize y coordinates
                
                if args.debug:
                    print(f"\nProcessing image {i}")
                    print(f"Input image shape: {image.shape}")
                    print(f"Ground truth boxes: {len(gt_boxes)}")
                
                # Run inference
                try:
                    predictions = model(image)
                    if args.debug:
                        print("Model inference completed successfully")
                except RuntimeError as e:
                    print(f"\nError during inference for image {i}: {str(e)}")
                    continue
                
                # Extract predictions (now they're already filtered and processed)
                pred_boxes = predictions['boxes'][0]  # [N, 4]
                pred_scores = predictions['scores'][0]  # [N]
                pred_labels = predictions['labels'][0]  # [N]
                
                if args.debug:
                    print(f"\nRaw predictions:")
                    print(f"- Boxes shape: {pred_boxes.shape}")
                    print(f"- Scores shape: {pred_scores.shape}")
                    print(f"- Labels shape: {pred_labels.shape}")
                    
                    # Analyze box coordinates before any processing
                    analyze_boxes(pred_boxes, gt_boxes, "Before Processing")
                
                # Move tensors to CPU
                pred_boxes = pred_boxes.cpu()
                pred_scores = pred_scores.cpu()
                pred_labels = pred_labels.cpu()
                gt_boxes = gt_boxes.cpu()
                
                # Apply NMS with debug info
                if len(pred_boxes) > 0:
                    if args.debug:
                        print(f"\nBefore NMS:")
                        print(f"- Number of predictions: {len(pred_boxes)}")
                        print(f"- Score range: [{pred_scores.min():.3f}, {pred_scores.max():.3f}]")
                    
                    # Apply confidence threshold before NMS
                    conf_mask = pred_scores > args.confidence_threshold
                    pred_boxes = pred_boxes[conf_mask]
                    pred_scores = pred_scores[conf_mask]
                    pred_labels = pred_labels[conf_mask]
                    
                    if len(pred_boxes) > 0:
                        keep = nms(pred_boxes, pred_scores, args.nms_threshold)
                        pred_boxes = pred_boxes[keep]
                        pred_scores = pred_scores[keep]
                        pred_labels = pred_labels[keep]
                        
                        if args.debug:
                            print(f"\nAfter NMS:")
                            print(f"- Number of predictions: {len(pred_boxes)}")
                            if len(pred_scores) > 0:
                                print(f"- Score range: [{pred_scores.min():.3f}, {pred_scores.max():.3f}]")
                                print(f"- Unique labels: {torch.unique(pred_labels).tolist()}")
                                
                            # Analyze box coordinates after NMS
                            analyze_boxes(pred_boxes, gt_boxes, "After NMS")
                
                # Track prediction statistics
                total_predictions = len(pred_boxes)
                total_valid_predictions = len(pred_boxes)
                
                # Calculate metrics
                precision, recall, f1 = calculate_metrics(pred_boxes, pred_scores, gt_boxes, args.confidence_threshold)
                if args.debug and i < 2:
                    print(f"\nMetrics for image {i}:")
                    print(f"- Precision: {precision:.4f}")
                    print(f"- Recall: {recall:.4f}")
                    print(f"- F1 Score: {f1:.4f}")
                
                # Store predictions and ground truth for visualization
                all_pred_boxes.append(pred_boxes)
                all_pred_scores.append(pred_scores)
                all_gt_boxes.append(gt_boxes)
                
                # Collect level-specific predictions if available
                if 'level_outputs' in predictions:
                    for level_idx, level_output in enumerate(predictions['level_outputs']):
                        level_predictions[level_idx].append({
                            'boxes': level_output['boxes'][0],
                            'scores': level_output['scores'][0],
                            'labels': level_output['labels'][0]
                        })
                
                # Visualize first 5 images
                if i < 5:
                    output_path = os.path.join(args.output_dir, f'detection_{i}.png')
                    visualize_detections(image[0].cpu(), pred_boxes, pred_scores, gt_boxes, output_path)
                    if args.debug:
                        print(f"\nVisualization saved to: {output_path}")
                
                # Clear memory
                del image, predictions, pred_boxes, pred_scores, gt_boxes
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"\nError processing image {i}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    # Calculate mean metrics
    metrics = []
    for pred_boxes, pred_scores, gt_boxes in zip(all_pred_boxes, all_pred_scores, all_gt_boxes):
        if len(pred_boxes) > 0 and len(pred_scores) > 0:
            precision, recall, f1 = calculate_metrics(pred_boxes, pred_scores, gt_boxes)
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