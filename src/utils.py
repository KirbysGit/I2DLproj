import yaml
import torch
import numpy as np
from pathlib import Path

def load_config(config_path):
    """
    Load and parse YAML configuration file
    Args:
        config_path: Path to the YAML configuration file
    Returns:
        dict: Configuration dictionary with all parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def compute_iou(boxes1, boxes2):
    """
    Compute Intersection over Union (IoU) between two sets of bounding boxes
    
    Args:
        boxes1: (N, 4) array of boxes in format [x1, y1, x2, y2]
        boxes2: (M, 4) array of boxes in format [x1, y1, x2, y2]
    
    Returns:
        iou_matrix: (N, M) array containing IoU values for each box pair
    
    Note:
        IoU = Area of Intersection / Area of Union
        Used for matching predicted boxes with ground truth boxes
    """
    # Split coordinates for easier computation
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)
    
    # Calculate intersection coordinates
    xa = np.maximum(x11, np.transpose(x21))  # Maximum of left coordinates
    ya = np.maximum(y11, np.transpose(y21))  # Maximum of top coordinates
    xb = np.minimum(x12, np.transpose(x22))  # Minimum of right coordinates
    yb = np.minimum(y12, np.transpose(y22))  # Minimum of bottom coordinates
    
    # Calculate intersection area
    inter_area = np.maximum(0, xb - xa) * np.maximum(0, yb - ya)
    
    # Calculate box areas
    box1_area = (x12 - x11) * (y12 - y11)
    box2_area = (x22 - x21) * (y22 - y21)
    
    # Calculate IoU
    # IoU = intersection / (area1 + area2 - intersection)
    iou = inter_area / (box1_area + np.transpose(box2_area) - inter_area + 1e-6)
    return iou

def save_checkpoint(model, optimizer, epoch, path):
    """
    Save model and training state to checkpoint file
    
    Args:
        model: PyTorch model to save
        optimizer: Optimizer state to save
        epoch: Current epoch number
        path: Path where to save the checkpoint
    
    Note:
        Saves both model and optimizer state for training resumption
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    # Create directory if it doesn't exist
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer, path):
    """
    Load model and training state from checkpoint file
    
    Args:
        model: PyTorch model to load weights into
        optimizer: Optimizer to load state into
        path: Path to the checkpoint file
    
    Returns:
        int: The epoch number when checkpoint was saved
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

def custom_collate_fn(batch):
    """Custom collate function with data verification"""
    # Verify data shapes
    for i, sample in enumerate(batch):
        if sample['image'].ndim != 3:
            raise ValueError(f"Image {i} has wrong dimensions: {sample['image'].shape}")
        if sample['boxes'].ndim != 2:
            raise ValueError(f"Boxes {i} has wrong dimensions: {sample['boxes'].shape}")
    
    # Get max number of boxes in this batch
    max_boxes = max(len(sample['boxes']) for sample in batch)
    
    # Initialize lists for batch items
    images = []
    boxes = []
    labels = []
    obj_targets = []
    box_targets = []
    image_names = []
    
    for sample in batch:
        images.append(torch.from_numpy(sample['image']))
        
        # Pad boxes and labels if necessary
        num_boxes = len(sample['boxes'])
        if num_boxes < max_boxes:
            # Pad boxes with zeros
            padded_boxes = np.zeros((max_boxes, 4), dtype=np.float32)
            padded_boxes[:num_boxes] = sample['boxes']
            padded_labels = np.zeros(max_boxes, dtype=np.int64)
            padded_labels[:num_boxes] = sample['labels']
        else:
            padded_boxes = sample['boxes']
            padded_labels = sample['labels']
        
        boxes.append(torch.from_numpy(padded_boxes))
        labels.append(torch.from_numpy(padded_labels))
        obj_targets.append(sample['obj_targets'])
        box_targets.append(sample['box_targets'])
        image_names.append(sample['image_name'])
    
    # Stack all tensors
    images = torch.stack(images)
    boxes = torch.stack(boxes)
    labels = torch.stack(labels)
    obj_targets = torch.stack(obj_targets)
    box_targets = torch.stack(box_targets)
    
    return {
        'image': images,
        'boxes': boxes,
        'labels': labels,
        'obj_targets': obj_targets,
        'box_targets': box_targets,
        'image_names': image_names,
        'num_boxes': torch.tensor([len(sample['boxes']) for sample in batch])  # Store original box counts
    }
