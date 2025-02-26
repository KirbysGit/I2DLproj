import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
import torch
from pathlib import Path


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
    
        
        # Setup paths
        dataset_path = Path(config['dataset']['path'])
        self.image_dir = dataset_path / config['dataset'][f'{split}_path']
        annotations_path = dataset_path / config['dataset']['annotations_path']
        
        # Add data verification step
        self._verify_dataset_structure(dataset_path, annotations_path)
        
        # Load annotations with validation
        ann_file = annotations_path / f'annotations_{split}.csv'
        self.annotations = self._load_and_validate_annotations(ann_file)
        
        # Add class mapping
        self.class_to_idx = {'object': 0}  # Map class names to indices
        
        # Validate class labels
        unique_classes = self.annotations['class'].unique()
        unknown_classes = [cls for cls in unique_classes if cls not in self.class_to_idx]
        if unknown_classes:
            raise ValueError(f"Found unknown classes in annotations: {unknown_classes}")
        
        print(f"Found classes: {unique_classes}")
        print(f"Class mapping: {self.class_to_idx}")
        
        # Group by image
        self.image_groups = self.annotations.groupby('image_name')
        self.image_names = list(self.image_groups.groups.keys())
        
        if config['dataset'].get('test_mode', False):
            print("- Test mode enabled")
            self.image_names = self.image_names[:config['dataset']['test_samples']]
            print(f"- Created {len(self.image_names)} test samples")
        
        print(f"- Total images: {len(self.image_names)}")
        
        # Filter annotations to only include valid images
        self.annotations = self.annotations[
            self.annotations['image_name'].isin(self.image_names)
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

    def _verify_dataset_structure(self, dataset_path, annotations_path):
        """Verify the dataset structure and files exist"""
        if not dataset_path.exists():
            raise RuntimeError(f"Dataset path does not exist: {dataset_path}")
            
        if not self.image_dir.exists():
            raise RuntimeError(f"Image directory does not exist: {self.image_dir}")
            
        if not annotations_path.exists():
            raise RuntimeError(f"Annotations path does not exist: {annotations_path}")
            
        # Print dataset statistics
        print(f"\nDataset Structure Verification:")
        print(f"- Dataset root: {dataset_path}")
        print(f"- Images path: {self.image_dir}")
        print(f"- Annotations path: {annotations_path}")

    def _load_and_validate_annotations(self, ann_file):
        """Load and validate annotations file"""
        if not ann_file.exists():
            raise RuntimeError(f"Annotations file not found: {ann_file}")
            
        # Load annotations
        column_names = ['image_name', 'x1', 'y1', 'x2', 'y2', 'class', 'width', 'height']
        annotations = pd.read_csv(ann_file, names=column_names, header=None)
        
        # Validate annotations
        self._validate_annotations(annotations)
        
        return annotations
        
    def _validate_annotations(self, annotations):
        """Validate annotation format and content"""
        required_columns = ['image_name', 'x1', 'y1', 'x2', 'y2', 'class']
        missing_columns = [col for col in required_columns if col not in annotations.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Validate coordinate values
        invalid_coords = (
            (annotations['x1'] >= annotations['x2']) |
            (annotations['y1'] >= annotations['y2']) |
            (annotations[['x1', 'x2', 'y1', 'y2']] < 0).any(axis=1)
        )
        
        if invalid_coords.any():
            print(f"Found {invalid_coords.sum()} invalid bounding boxes")
            print("First few invalid annotations:")
            print(annotations[invalid_coords].head())
            
        # Print annotation statistics
        print(f"\nAnnotation Statistics:")
        print(f"- Total annotations: {len(annotations)}")
        print(f"- Unique images: {annotations['image_name'].nunique()}")
        print(f"- Average boxes per image: {len(annotations) / annotations['image_name'].nunique():.2f}")

    def _get_transforms(self):
        """Get data augmentation transforms"""
        return A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                always_apply=True
            ),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0.0,
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
        """Get a single sample from the dataset with validation"""
        if idx in self.cache:
            return self.cache[idx]
        
        try:
            image_name = self.image_names[idx]
            image_anns = self.image_groups.get_group(image_name)
            
            # Load and validate image
            image_path = self.image_dir / image_name
            image = cv2.imread(str(image_path))
            
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            if image.size == 0:
                raise ValueError(f"Empty image: {image_path}")
            
            # Verify image dimensions are reasonable
            if image.shape[0] > 4000 or image.shape[1] > 4000:
                print(f"Warning: Large image detected: {image.shape}")
                # Resize large images to reasonable dimensions while maintaining aspect ratio
                scale = 4000 / max(image.shape[0], image.shape[1])
                new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
                image = cv2.resize(image, new_size)
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get original dimensions and target size
            orig_height, orig_width = image.shape[:2]
            target_size = self.config['preprocessing']['image_size']
            
            # Calculate resize scale while preserving aspect ratio
            scale = min(target_size[0] / orig_height, target_size[1] / orig_width)
            
            # Resize image
            new_height = int(orig_height * scale)
            new_width = int(orig_width * scale)
            resized_image = cv2.resize(image, (new_width, new_height))
            
            # Create padded image
            padded_image = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
            pad_y = (target_size[0] - new_height) // 2
            pad_x = (target_size[1] - new_width) // 2
            padded_image[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = resized_image
            
            # Extract and transform box coordinates
            boxes = image_anns[['x1', 'y1', 'x2', 'y2']].values.astype(np.float32)
            
            # Scale coordinates
            boxes = boxes * scale
            
            # Add padding offset
            boxes[:, [0, 2]] += pad_x
            boxes[:, [1, 3]] += pad_y
            
            # Normalize to [0, 1] range
            boxes[:, [0, 2]] /= target_size[1]
            boxes[:, [1, 3]] /= target_size[0]
            
            # Clip boxes to valid range
            boxes = np.clip(boxes, 0, 1)
            
            # Convert class labels
            class_labels = np.array([self.class_to_idx[label] for label in image_anns['class'].values])
            
            # Apply normalization transform
            if self.transform:
                transformed = self.transform(
                    image=padded_image,
                    bboxes=boxes,
                    class_labels=class_labels
                )
                padded_image = transformed['image']
                boxes = np.array(transformed['bboxes'])
                class_labels = np.array(transformed['class_labels'])
            
            # Convert to tensor format
            image = torch.from_numpy(np.transpose(padded_image, (2, 0, 1)))
            boxes = torch.from_numpy(boxes.astype(np.float32))
            class_labels = torch.from_numpy(class_labels)
            
            result = {
                'image': image,
                'boxes': boxes,
                'class_labels': class_labels,
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
        grid_size = (20, 20)
        grid_h, grid_w = grid_size
        
        # Initialize target tensors
        obj_targets = torch.zeros(grid_h, grid_w)
        box_targets = torch.zeros(grid_h, grid_w, 4)
        
        # Normalize boxes to [0,1] range if not already
        if boxes.max() > 1:
            boxes = boxes / np.array([image_size[1], image_size[0], image_size[1], image_size[0]])
        
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
