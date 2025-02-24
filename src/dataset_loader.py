import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
import torch
from pathlib import Path
from .utils import ColorLogger

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
        self.transform = transform or self._get_transforms()
        
        # Initialize logger
        self.logger = ColorLogger()
        
        # Setup paths
        dataset_path = Path(config['dataset']['path'])
        self.image_dir = dataset_path / config['dataset'][f'{split}_path']
        annotations_path = dataset_path / config['dataset']['annotations_path']
        
        # Load annotations
        print(f"\nInitializing {split} dataset:")
        print(f"- Image path: {self.image_dir}")
        
        # Add class mapping
        self.class_to_idx = {'object': 0}  # Map class names to indices
        
        # Load annotations file with correct column names
        ann_file = annotations_path / f'annotations_{split}.csv'
        try:
            # Define column names based on the CSV structure
            column_names = [
                'image_name',  # First column is the image name
                'x1', 'y1', 'x2', 'y2',  # Bounding box coordinates
                'class',  # Object class (e.g., 'object')
                'width', 'height'  # Image dimensions
            ]
            
            # Load CSV with specified column names
            self.annotations = pd.read_csv(
                ann_file, 
                names=column_names,  # Use our defined column names
                header=None  # CSV has no header row
            )
            
            print("- Loaded annotations columns:", self.annotations.columns.tolist())
            print("- First few rows:")
            print(self.annotations.head())
            print(f"- Loaded {len(self.annotations)} annotations")
            
            # Convert class strings to indices
            self.annotations['class'] = self.annotations['class'].map(self.class_to_idx)
            
            print("- Classes mapped to indices:", self.class_to_idx)
            
            # Group by image
            self.image_groups = self.annotations.groupby('image_name')
            self.image_names = list(self.image_groups.groups.keys())
            
            if config['dataset'].get('test_mode', False):
                print("- Test mode enabled")
                self.image_names = self.image_names[:config['dataset']['test_samples']]
                print(f"- Created {len(self.image_names)} test samples")
            
            print(f"- Total images: {len(self.image_names)}")
            
        except Exception as e:
            print(f"Error loading annotations: {str(e)}")
            print(f"Attempted to load file: {ann_file}")
            print(f"File exists: {ann_file.exists()}")
            if ann_file.exists():
                print("First few lines of file:")
                with open(ann_file, 'r') as f:
                    print(f.read(500))
            raise
        
        # Verify images exist
        print("\nVerifying image files...")
        valid_images = []
        missing_images = []
        for img_name in self.image_names:
            img_path = self.image_dir / img_name
            if img_path.exists():
                valid_images.append(img_name)
            else:
                missing_images.append(img_name)
        
        if missing_images:
            print(f"Warning: {len(missing_images)} images not found:")
            print(f"First few missing: {missing_images[:5]}")
        
        self.image_names = valid_images
        print(f"Using {len(valid_images)} valid images")
        
        # Filter annotations to only include valid images
        self.annotations = self.annotations[
            self.annotations['image_name'].isin(valid_images)
        ]
        
        # Add caching for transformed images
        self.cache = {}
        self.cache_size = config.get('dataset', {}).get('cache_size', 100)
        
        # Check directory structure
        self.image_dir = dataset_path / config['dataset'][f'{split}_path']
        
        # Verify directories exist
        print("\nChecking directory structure:")
        print(f"Dataset root exists: {dataset_path.exists()}")
        print(f"Images directory exists: {self.image_dir.exists()}")
        print(f"Directory contents:")
        if self.image_dir.exists():
            print([x.name for x in self.image_dir.iterdir()][:5])
        else:
            print("Image directory not found!")
            print(f"Expected path: {self.image_dir}")
        
        # Apply sample limits based on mode
        if config['dataset'].get('custom_mode', False):
            num_samples = config['dataset']['custom_samples'][split]
            self.image_names = self.image_names[:num_samples]
            print(f"Using {num_samples} images for {split} in custom mode")

    def _get_transforms(self):
        """Get data augmentation transforms"""
        return A.Compose([
            # Use Resize instead of RandomResizedCrop
            A.Resize(
                height=640,
                width=640,
                always_apply=True
            ),
            # Add basic augmentations
            A.OneOf([
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
                A.HorizontalFlip(p=1)
            ], p=0.5),
            # Always normalize
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                always_apply=True
            ),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=1024,  # Minimum box area
            min_visibility=0.3,  # Minimum box visibility
            label_fields=['class_labels']
        ))

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
        
        try:
            # Get image name and its annotations
            image_name = self.image_names[idx]
            image_anns = self.image_groups.get_group(image_name)
            
            # Load and preprocess image
            image_path = self.image_dir / image_name
            image = cv2.imread(str(image_path))
            
            if image is None or image.size == 0:
                print(f"Skipping corrupted image: {image_path}")
                return self.__getitem__((idx + 1) % len(self))
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Extract bounding box coordinates and class indices
            boxes = image_anns[['x1', 'y1', 'x2', 'y2']].values
            
            # Clip coordinates to image boundaries
            boxes = np.clip(boxes, 0, [image.shape[1], image.shape[0], image.shape[1], image.shape[0]])
            
            # Get class indices (already converted to integers in __init__)
            class_labels = image_anns['class'].values  # Now these are integers
            
            # Apply transforms
            if self.transform:
                transformed = self.transform(
                    image=image,
                    bboxes=boxes,
                    class_labels=class_labels
                )
                image = transformed['image']
                boxes = np.clip(transformed['bboxes'], 0, 1)  # Clip to [0,1] range
                class_labels = transformed['class_labels']
            
            # Convert boxes to grid format
            obj_targets, box_targets = self.process_ground_truth(
                boxes, 
                image_size=self.config['preprocessing']['image_size']
            )
            
            # Convert to tensor format
            image = torch.from_numpy(np.transpose(image, (2, 0, 1)))
            boxes = torch.from_numpy(np.array(boxes, dtype=np.float32))
            class_labels = torch.from_numpy(np.array(class_labels, dtype=np.int64))
            
            result = {
                'image': image,
                'boxes': boxes,
                'class_labels': class_labels,
                'obj_targets': obj_targets,
                'box_targets': box_targets,
                'image_name': image_name
            }
            
            # Cache the result
            if len(self.cache) < self.cache_size:
                self.cache[idx] = result
            
            return result
        
        except Exception as e:
            print(f"Error processing image {image_name}: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))

    def process_ground_truth(self, boxes, image_size=(640, 640)):
        """Convert ground truth boxes to grid format"""
        grid_size = (20, 20)  # Based on model's output size
        grid_h, grid_w = grid_size
        
        # Initialize target tensors
        obj_targets = torch.zeros(grid_h, grid_w)
        box_targets = torch.zeros(grid_h, grid_w, 4)
        
        # Convert boxes to grid cells
        for box in boxes:
            x1, y1, x2, y2 = box
            
            # Calculate box center
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            # Convert to grid coordinates
            grid_x = int(cx * grid_w)
            grid_y = int(cy * grid_h)
            
            # Ensure grid coordinates are within bounds
            grid_x = min(max(grid_x, 0), grid_w - 1)
            grid_y = min(max(grid_y, 0), grid_h - 1)
            
            # Set targets
            obj_targets[grid_y, grid_x] = 1
            box_targets[grid_y, grid_x] = torch.tensor([x1, y1, x2, y2])
        
        return obj_targets, box_targets
