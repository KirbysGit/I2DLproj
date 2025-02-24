import yaml
import torch
import numpy as np
from pathlib import Path
from colorama import init, Fore, Style
import logging
from datetime import datetime

# Initialize colorama for Windows support
init()

class ColorLogger:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, log_dir='results/logs'):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.setup_logger(log_dir)
    
    def setup_logger(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create file handler
        log_file = self.log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Set file logging to DEBUG level
        
        # Create console handler with colors
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Set console to INFO level
        
        # Create formatters and add them to the handlers
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        
        # Setup logger
        self.logger = logging.getLogger('TrainingLogger')
        self.logger.setLevel(logging.DEBUG)  # Set root logger to DEBUG
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def debug(self, msg):
        """Log debug message (only to file)"""
        self.logger.debug(msg)

    def info(self, msg, color=None):
        """Log info message with optional color"""
        colored_msg = f"{color}{msg}{Style.RESET_ALL}" if color else msg
        self.logger.info(msg)  # Log original message to file
        if color:  # Print colored message to console
            print(colored_msg)
        else:
            print(msg)

    def warning(self, msg):
        """Log warning message in yellow"""
        colored_msg = f"{Fore.YELLOW}WARNING: {msg}{Style.RESET_ALL}"
        self.logger.warning(msg)
        print(colored_msg)

    def error(self, msg):
        """Log error message in red"""
        colored_msg = f"{Fore.RED}ERROR: {msg}{Style.RESET_ALL}"
        self.logger.error(msg)
        print(colored_msg)

    def success(self, msg):
        """Log success message in green"""
        colored_msg = f"{Fore.GREEN}{msg}{Style.RESET_ALL}"
        self.logger.info(msg)
        print(colored_msg)

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
    """Custom collate function for DataLoader"""
    images = []
    boxes = []
    labels = []  # This expects 'labels' but dataset provides 'class_labels'
    obj_targets = []
    box_targets = []
    
    for sample in batch:
        images.append(sample['image'])
        boxes.append(sample['boxes'])
        labels.append(sample['class_labels'])  # Changed from 'labels' to 'class_labels'
        obj_targets.append(sample['obj_targets'])
        box_targets.append(sample['box_targets'])
    
    # Stack tensors
    images = torch.stack(images)
    obj_targets = torch.stack(obj_targets)
    box_targets = torch.stack(box_targets)
    
    return {
        'image': images,
        'boxes': boxes,
        'labels': torch.cat(labels),  # Keep as 'labels' for backward compatibility
        'obj_targets': obj_targets,
        'box_targets': box_targets
    }

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.01, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, value):
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'max':
            if value > self.best_value + self.min_delta:
                self.best_value = value
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'min'
            if value < self.best_value - self.min_delta:
                self.best_value = value
                self.counter = 0
            else:
                self.counter += 1
        
        self.should_stop = self.counter >= self.patience
        return self.should_stop
