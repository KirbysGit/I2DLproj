# data / dataset.py

# -----

# Defines dataset for SKU-110K.
# Handles Data Loading, Annotation Parsing, Image Transformations, and Batching.
# Serves As Bridge Between Data and Model.

# -----

# Imports.
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from pathlib import Path
import cv2
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

# Dataset Class.
class SKU110KDataset(Dataset):
    """Dataset class for SKU-110K object detection."""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 transform = None):
        """
        Args:
            data_dir:     Path to SKU-110K Dataset Directory.
            split:        'train', 'val', or 'test'.
            transform:    Data Augmentation Transforms.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Load Annotations from CSV.
        ann_file = self.data_dir / 'annotations' / f'annotations_{split}.csv'
        self.annotations_df = pd.read_csv(ann_file, names=[
            'image_name', 'x1', 'y1', 'x2', 'y2', 'class', 'image_width', 'image_height'
        ])
        
        # Print Column Names to Debug.
        print(f"CSV Columns: {self.annotations_df.columns.tolist()}")
        
        # Get Unique Image IDs - Using First Column Which Should be Image Names.
        image_col = self.annotations_df.columns[0]  # Get First Column Name.
        self.image_ids = self.annotations_df[image_col].unique()
        
        # Create Image Path Mapping.
        self.image_paths = {
            img_id: self.data_dir / 'images' / split / f'{img_id}'  # Remove .jpg since it might be in the name
            for img_id in self.image_ids
        }
    
    # Length of Dataset.
    def __len__(self) -> int:
        return len(self.image_ids)
    
    # Get Image Annotations.
    def get_image_annotations(self, img_id: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get boxes and labels for an image."""

        # Get Annotations for This Image.
        img_anns = self.annotations_df[self.annotations_df['image_name'] == img_id]
        
        # Extract Boxes and Labels.
        boxes = []
        labels = []
        
        # Iterate Over Annotations.
        for _, ann in img_anns.iterrows():
            x1, y1, x2, y2 = ann['x1'], ann['y1'], ann['x2'], ann['y2']
            boxes.append([x1, y1, x2, y2])
            labels.append(1)  # All Objects are Class 1, Background is 0.
        
        return (
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long)  # Make sure labels are long type
        )
    
    # Resize Image with Aspect Ratio.
    def resize_with_aspect_ratio(self, image, boxes, target_size=(800, 800)):
        """Resize image maintaining aspect ratio and pad if necessary."""
        orig_h, orig_w = image.shape[:2]
        
        # Calculate Scale to Maintain Aspect Ratio.
        scale = min(target_size[0] / orig_w, target_size[1] / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        # Resize Image.
        resized = cv2.resize(image, (new_w, new_h))
        
        # Create Black Padding.
        padded = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
        
        # Center the Resized Image.
        y_offset = (target_size[1] - new_h) // 2
        x_offset = (target_size[0] - new_w) // 2
        padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Scale Boxes.
        if len(boxes) > 0:
            # Scale Coordinates.
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + x_offset
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + y_offset
        
        return padded, boxes
    
    # Get Item.
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        img_id = self.image_ids[idx]
        image_path = self.image_paths[img_id]
        
        # Load Image.
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get original image dimensions
        orig_h, orig_w = image.shape[:2]
        
        # Get boxes and labels for this image
        boxes, labels = self.get_image_annotations(img_id)
        
        # Convert boxes to absolute coordinates if they're normalized
        if boxes.max() <= 1.0:
            boxes = boxes * torch.tensor([orig_w, orig_h, orig_w, orig_h], dtype=torch.float32)
        
        # Ensure boxes are valid
        boxes = torch.clamp(boxes, min=0)
        boxes[..., 0].clamp_(max=orig_w)  # x1
        boxes[..., 1].clamp_(max=orig_h)  # y1
        boxes[..., 2].clamp_(max=orig_w)  # x2
        boxes[..., 3].clamp_(max=orig_h)  # y2
        
        # Filter out invalid boxes
        valid_boxes = []
        valid_labels = []
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            if x2 > x1 and y2 > y1:
                valid_boxes.append([x1, y1, x2, y2])
                valid_labels.append(label)
        
        if len(valid_boxes) == 0:
            # Handle case with no valid boxes
            valid_boxes = torch.zeros((0, 4), dtype=torch.float32)
            valid_labels = torch.zeros(0, dtype=torch.long)
        else:
            valid_boxes = torch.tensor(valid_boxes, dtype=torch.float32)
            valid_labels = torch.tensor(valid_labels, dtype=torch.long)
        
        # Apply transforms if any
        if self.transform:
            # Convert boxes to pascal_voc format (already in this format)
            transformed = self.transform(
                image=image,
                bboxes=valid_boxes.numpy(),
                labels=valid_labels.numpy()
            )
            image = transformed['image']  # Will be a tensor
            boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
            labels = torch.tensor(transformed['labels'], dtype=torch.long)
        else:
            # If no transform, convert image to tensor
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            boxes = valid_boxes
            labels = valid_labels
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'image_id': img_id
        }
    
    # Collate Function.
    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching."""
        images = torch.stack([item['image'] for item in batch])
        image_ids = [item['image_id'] for item in batch]
        
        # Pad Boxes and Labels to Same Length.
        max_boxes = max(len(item['boxes']) for item in batch)
        
        batch_boxes = []
        batch_labels = []
        
        for item in batch:
            num_boxes = len(item['boxes'])
            if num_boxes == 0:
                # Handle Images with No Boxes.
                boxes = torch.zeros((max_boxes, 4), dtype=torch.float32)
                labels = torch.zeros(max_boxes, dtype=torch.long)
            else:
                # Pad Using Torch's Native Padding.
                boxes = torch.cat([
                    item['boxes'],
                    torch.zeros((max_boxes - num_boxes, 4), dtype=torch.float32)
                ], dim=0)
                
                labels = torch.cat([
                    item['labels'],
                    torch.zeros(max_boxes - num_boxes, dtype=torch.long)
                ], dim=0)
            
            # Append to Batch.
            batch_boxes.append(boxes)
            batch_labels.append(labels)
        
        # Stack Boxes and Labels.
        boxes = torch.stack(batch_boxes)
        labels = torch.stack(batch_labels)
        
        return {
            'images': images,
            'boxes': boxes,
            'labels': labels,
            'image_ids': image_ids
        } 