import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from .utils import load_config, compute_iou
from .dataset_loader import SKU110KDataset
from .model import CNNViTHybrid

def evaluate(model, data_loader, device, config):
    """
    Evaluate model performance on validation/test set
    Returns metrics including precision, recall, and F1-score
    """
    model.eval()  # Set model to evaluation mode
    all_predictions = []
    all_targets = []
    
    # Disable gradient computation for evaluation
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Move input data to device (GPU/CPU)
            images = batch['image'].to(device)
            target_boxes = batch['boxes']
            target_labels = batch['labels']
            
            # Get model predictions
            predictions = model(images)
            
            # Process predictions: filter by confidence threshold
            pred_boxes = []
            for pred in predictions:
                # Only keep predictions above confidence threshold
                conf_mask = pred[..., 4] > config['evaluation']['conf_threshold']
                boxes = pred[conf_mask][..., :4]  # Get bounding box coordinates
                pred_boxes.append(boxes.cpu().numpy())
            
            # Collect all predictions and targets for metric calculation
            all_predictions.extend(pred_boxes)
            all_targets.extend([boxes.numpy() for boxes in target_boxes])
    
    # Calculate evaluation metrics
    metrics = calculate_metrics(all_predictions, all_targets, config)
    return metrics

def calculate_metrics(predictions, targets, config):
    """
    Calculate object detection metrics:
    - Precision: TP / (TP + FP)
    - Recall: TP / (TP + FN)
    - F1-score: 2 * (Precision * Recall) / (Precision + Recall)
    """
    total_tp = 0  # True Positives
    total_fp = 0  # False Positives
    total_fn = 0  # False Negatives
    
    # Process each image's predictions
    for pred_boxes, target_boxes in zip(predictions, targets):
        # Handle edge cases
        if len(pred_boxes) == 0:
            total_fn += len(target_boxes)  # All targets missed
            continue
            
        if len(target_boxes) == 0:
            total_fp += len(pred_boxes)  # All predictions are false positives
            continue
        
        # Calculate IoU between all predictions and targets
        ious = compute_iou(pred_boxes, target_boxes)
        
        # Consider predictions as matches if IoU > threshold
        iou_threshold = config['evaluation']['iou_threshold']
        matches = ious > iou_threshold
        
        # Update metrics
        total_tp += matches.any(axis=1).sum()  # Predictions that match any target
        total_fp += len(pred_boxes) - matches.any(axis=1).sum()  # Predictions that match no targets
        total_fn += len(target_boxes) - matches.any(axis=0).sum()  # Targets that match no predictions
    
    # Calculate final metrics
    precision = total_tp / (total_tp + total_fp + 1e-6)  # Add epsilon to avoid division by zero
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def calculate_map(predictions, targets, iou_threshold=0.5):
    """Calculate mean Average Precision"""
    # Implementation of mAP calculation
    pass

def calculate_recall(predictions, targets, iou_threshold=0.5):
    """Calculate Recall at specific IoU threshold"""
    # Implementation of recall calculation
    pass

def main():
    """
    Main evaluation script to test model performance on test set
    """
    # Load configuration and setup
    config = load_config('config/config.yaml')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create test dataset and dataloader
    test_dataset = SKU110KDataset(config, split='test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['preprocessing']['batch_size'],
        shuffle=False,
        num_workers=config['preprocessing']['num_workers']
    )
    
    # Initialize and load trained model
    model = CNNViTHybrid(config).to(device)
    checkpoint_path = f"{config['training']['save_dir']}/best_model.pt"
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])
    
    # Run evaluation
    metrics = evaluate(model, test_loader, device, config)
    
    # Print results
    print("\nEvaluation Results:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

if __name__ == '__main__':
    main()
