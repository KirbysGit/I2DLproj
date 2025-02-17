import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
import torch

class SKU110KDataset(Dataset):
    """
    PyTorch Dataset class for SKU-110K retail product detection dataset.
    Handles loading of images and annotations, and applies transformations.
    """
    
    def __init__(self, config, split='train', transform=None):
        """
        Initialize dataset with configuration and split
        Args:
            config (dict): Configuration dictionary with dataset paths and parameters
            split (str): Dataset split - 'train', 'val', or 'test'
            transform: Optional custom transform pipeline
        """
        self.config = config
        self.split = split
        self.transform = transform or self._get_default_transforms()
        
        print(f"\nInitializing {split} dataset:")
        
        # Setup paths
        self.dataset_path = config['dataset']['path']
        self.images_path = os.path.join(self.dataset_path, f"images/{split}")
        print(f"- Image path: {self.images_path}")
        
        # For testing, create dummy data first
        if config.get('dataset', {}).get('test_mode', False):
            print("- Creating test data...")
            self._create_test_data()
            self._create_test_annotations()
            print(f"- Created {config['dataset'].get('test_samples', 5)} test samples")
        else:
            # Load annotations
            annotations_file = os.path.join(
                self.dataset_path,
                'annotations',
                f'annotations_{split}.csv'
            )
            if os.path.exists(annotations_file):
                self.annotations = pd.read_csv(annotations_file)
                print(f"- Loaded {len(self.annotations)} annotations")
            else:
                print("- No annotations found, creating test data")
                self._create_test_annotations()
        
        # Group annotations
        self.image_groups = self.annotations.groupby('image_name')
        self.image_names = list(self.image_groups.groups.keys())
        print(f"- Total images: {len(self.image_names)}")

        # Add caching for transformed images
        self.cache = {}
        self.cache_size = config.get('dataset', {}).get('cache_size', 100)

    def _get_default_transforms(self):
        """Optimized transform pipeline"""
        return A.Compose([
            A.Resize(height=self.config['preprocessing']['image_size'][0],
                    width=self.config['preprocessing']['image_size'][1]),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

    def _create_test_data(self):
        """Create dummy data for testing"""
        os.makedirs(self.images_path, exist_ok=True)
        
        # Create a few dummy images
        num_samples = self.config['dataset'].get('test_samples', 5)
        for i in range(num_samples):
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            img_path = os.path.join(self.images_path, f'test_image_{i}.jpg')
            cv2.imwrite(img_path, img)

    def _create_test_annotations(self):
        """Create dummy annotations for testing"""
        annotations = []
        num_samples = self.config['dataset'].get('test_samples', 5)
        
        for i in range(num_samples):
            # Create random boxes for each image
            num_boxes = np.random.randint(1, 5)
            for _ in range(num_boxes):
                x1 = np.random.randint(0, 500)
                y1 = np.random.randint(0, 500)
                x2 = x1 + np.random.randint(50, 100)
                y2 = y1 + np.random.randint(50, 100)
                
                annotations.append({
                    'image_name': f'test_image_{i}.jpg',
                    'x_min': x1,
                    'y_min': y1,
                    'x_max': x2,
                    'y_max': y2,
                    'class': 0
                })
        
        self.annotations = pd.DataFrame(annotations)

    def __len__(self):
        """Return the total number of images in the dataset"""
        return len(self.image_names)

    def __getitem__(self, idx):
        """Get a single sample from the dataset with caching"""
        if idx in self.cache:
            return self.cache[idx]
        
        # Get image name and its annotations
        image_name = self.image_names[idx]
        image_anns = self.image_groups.get_group(image_name)
        
        # Load and preprocess image
        image_path = os.path.join(self.images_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        # Extract bounding box coordinates
        boxes = image_anns[['x_min', 'y_min', 'x_max', 'y_max']].values
        labels = np.zeros(len(boxes))
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                class_labels=labels
            )
            image = transformed['image']  # This is now in HWC format
            boxes = transformed['bboxes']
            labels = transformed['class_labels']
        
        # Convert to CHW format (what PyTorch expects)
        image = np.transpose(image, (2, 0, 1))
        
        # Process boxes into grid format
        obj_targets, box_targets = self.process_ground_truth(
            boxes, 
            image_size=self.config['preprocessing']['image_size']
        )
        
        result = {
            'image': image,  # Now in CHW format
            'boxes': np.array(boxes, dtype=np.float32),
            'labels': np.array(labels, dtype=np.int64),
            'image_name': image_name,
            'obj_targets': obj_targets,
            'box_targets': box_targets
        }
        
        # Cache the result
        if len(self.cache) < self.cache_size:
            self.cache[idx] = result
        
        return result

    def process_ground_truth(self, boxes, image_size=(640, 640)):
        """Convert ground truth boxes to grid format"""
        grid_size = (20, 20)  # Based on our model's output
        grid_h, grid_w = grid_size
        
        # Initialize target tensors
        obj_targets = torch.zeros(grid_h, grid_w)
        box_targets = torch.zeros(grid_h, grid_w, 4)
        
        # Convert boxes to grid cells
        for box in boxes:
            x1, y1, x2, y2 = box
            # Convert to grid coordinates
            grid_x = int((x1 + x2) / 2 * grid_w / image_size[1])
            grid_y = int((y1 + y2) / 2 * grid_h / image_size[0])
            
            if 0 <= grid_x < grid_w and 0 <= grid_y < grid_h:
                obj_targets[grid_y, grid_x] = 1
                box_targets[grid_y, grid_x] = torch.tensor([x1, y1, x2, y2])
        
        return obj_targets, box_targets
