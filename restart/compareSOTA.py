import os
import torch
import yaml
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from ultralytics import YOLO

# Import evaluation utilities from your existing code
from restart.utils.visualize_detections import visualize_detections
from restart.utils.plots import plot_precision_recall_curve, plot_map_progress
from restart.data.dataset import SKU110KDataset
from test import ObjectDetectionEvaluator, save_metrics_summary

def evaluate_yolov5(config_path, num_visualizations=5, max_images=100):
    """
    Evaluate YOLOv5 model on SKU110K dataset using the same metrics as SecureShelfNet.
    
    Args:
        config_path: Path to the testing config file
        num_visualizations: Number of sample images to visualize
        max_images: Maximum number of images to evaluate
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directories with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(config['output_dir']) / f'yolov5_eval_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    metrics_dir = output_dir / 'metrics'
    metrics_dir.mkdir(exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load validation dataset instead of test to ensure we have labels
    print("\nLoading validation dataset...")
    val_dataset = SKU110KDataset(
        data_dir=config['dataset_path'],
        split='val',  # Use validation split instead of test
        resize_dims=tuple(config['resize_dims'])
    )
    
    # Create a subset if specified
    if max_images and max_images < len(val_dataset):
        subset_indices = list(range(max_images))
        val_dataset = torch.utils.data.Subset(val_dataset, subset_indices)
        print(f"\nUsing subset of {max_images} images for evaluation")
    
    # Initialize YOLO model with lower confidence threshold
    print("\nInitializing YOLOv5 model...")
    model = YOLO('yolov5x.pt')  # Using YOLOv5x for best accuracy
    model.conf = 0.1  # Lower confidence threshold for detection
    model.iou = config['iou_threshold']  # Match IoU threshold from config
    
    # Initialize evaluator with same IoU threshold
    evaluator = ObjectDetectionEvaluator(iou_threshold=config['iou_threshold'])
    
    print("\nStarting evaluation...")
    total_images_processed = 0
    
    # Evaluate images
    for idx in tqdm(range(len(val_dataset))):
        try:
            # Get image and ground truth
            if isinstance(val_dataset, torch.utils.data.Subset):
                data = val_dataset.dataset[val_dataset.indices[idx]]
            else:
                data = val_dataset[idx]
            
            # Ensure we have all required data
            if 'image' not in data or 'boxes' not in data:
                print(f"Skipping image {idx}: Missing required data")
                continue
                
            image = data['image'].unsqueeze(0).to(device)  # Add batch dimension
            gt_boxes = data['boxes'].to(device)
            gt_labels = torch.ones(len(gt_boxes), dtype=torch.long, device=device)  # All objects are class 1
            orig_size = data['orig_size']
            resize_size = data['resize_size']
            
            # Run YOLO inference
            results = model(image, verbose=False)[0]  # Get first (only) image result
            
            # Extract predictions
            pred_boxes = results.boxes.xyxy  # Get boxes in x1,y1,x2,y2 format
            pred_scores = results.boxes.conf
            
            if len(pred_boxes) > 0:
                # Filter out low confidence predictions
                mask = pred_scores > config.get('confidence_threshold', 0.1)
                pred_boxes = pred_boxes[mask]
                pred_scores = pred_scores[mask]
            
            # Update metrics
            evaluator.update(pred_boxes, pred_scores, gt_boxes, orig_size, resize_size)
            
            # Visualize predictions
            if idx < num_visualizations:
                save_path = vis_dir / f'val_vis_{idx}.png'
                visualize_detections(
                    image_tensor=image[0],  # Remove batch dimension
                    detections={'boxes': pred_boxes, 'scores': pred_scores},
                    ground_truths={'boxes': gt_boxes, 'labels': gt_labels},
                    orig_size=orig_size,
                    resize_size=resize_size,
                    save_path=save_path,
                    title=f'Validation Image {idx}'
                )
            
            total_images_processed += 1
            if max_images and total_images_processed >= max_images:
                break
                
        except Exception as e:
            print(f"Error processing image {idx}: {str(e)}")
            continue
    
    print(f"\nProcessed {total_images_processed} images")
    
    # Get metrics and create visualizations
    metrics = evaluator.get_metrics()
    
    # Save metrics summary
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
    
    # Save evaluation config
    with open(output_dir / 'eval_config.yaml', 'w') as f:
        yaml.dump({
            'model': 'yolov5x',
            'config_path': str(config_path),
            'num_visualizations': num_visualizations,
            'max_images': max_images,
            'evaluation_timestamp': timestamp,
            'iou_threshold': evaluator.iou_threshold,
            'confidence_threshold': config.get('confidence_threshold', 0.1)
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
    # Use the same config as SecureShelfNet evaluation
    config_path = "restart/config/testing_config.yaml"
    
    # Run evaluation
    metrics = evaluate_yolov5(
        config_path=config_path,
        num_visualizations=10,
        max_images=100  # Match subset_size from config
    )
