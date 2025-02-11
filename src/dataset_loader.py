import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A

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
        # Use custom transforms if provided, else use default transforms
        self.transform = transform or self._get_default_transforms()
        
        # Setup dataset paths
        self.dataset_path = config['dataset']['path']
        self.images_path = os.path.join(self.dataset_path, f"images/{split}")
        
        # For testing, create dummy data
        if config.get('dataset', {}).get('test_mode', False):
            self._create_test_data()
            
        # Load annotations
        annotations_file = os.path.join(
            self.dataset_path,
            'annotations',
            f'annotations_{split}.csv'
        )
        
        if os.path.exists(annotations_file):
            self.annotations = pd.read_csv(annotations_file)
        else:
            # Create dummy annotations for testing
            self._create_test_annotations()
            
        # Group annotations by image for efficient retrieval
        self.image_groups = self.annotations.groupby('image_name')
        self.image_names = list(self.image_groups.groups.keys())

    def _get_default_transforms(self):
        """
        Default augmentation pipeline using Albumentations library
        Includes:
        - Resize to standard input size
        - Normalize using ImageNet statistics
        """
        return A.Compose([
            # Resize images to model's expected input size
            A.Resize(height=self.config['preprocessing']['image_size'][0],
                    width=self.config['preprocessing']['image_size'][1]),
            # Normalize using ImageNet mean and std
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
                    'y_max': y2
                })
        
        self.annotations = pd.DataFrame(annotations)

    def __len__(self):
        """Return the total number of images in the dataset"""
        return len(self.image_names)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset
        Returns:
            dict containing:
            - image: preprocessed image tensor
            - boxes: bounding box coordinates
            - labels: object class labels (all 0 for SKU-110K)
            - image_name: original image filename
        """
        # Get image name and its annotations
        image_name = self.image_names[idx]
        image_anns = self.image_groups.get_group(image_name)
        
        # Load and preprocess image
        image_path = os.path.join(self.images_path, image_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        # Extract bounding box coordinates
        boxes = image_anns[['x_min', 'y_min', 'x_max', 'y_max']].values
        # All objects are same class (retail products) in SKU-110K
        labels = np.zeros(len(boxes))
        
        # Apply transforms to both image and bounding boxes
        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                class_labels=labels
            )
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']
        
        return {
            'image': image,
            'boxes': np.array(boxes, dtype=np.float32),
            'labels': np.array(labels, dtype=np.int64),
            'image_name': image_name
        }
