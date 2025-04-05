# data / dataset.py

# -----

# Defines dataset for SKU-110K.
# Handles Data Loading, Annotation Parsing, Image Transformations, and Batching.
# Serves As Bridge Between Data and Model.

# -----

# Imports.
import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

# Dataset Class.
class SKU110KDataset(Dataset):
    """Dataset class for SKU-110K object detection."""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 transform = None, resize_dims = None):
        """
        Args:
            data_dir:     Path to SKU-110K Dataset Directory.
            split:        'train', 'val', or 'test'.
            transform:    Data Augmentation Transforms.
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.resize_dims = resize_dims
        
        # Load Annotations from CSV.
        ann_file = self.data_dir / 'annotations' / f'annotations_{split}.csv'
        self.annotations_df = pd.read_csv(ann_file, names=[
            'image_name', 'x1', 'y1', 'x2', 'y2', 'class', 'image_width', 'image_height'
        ])
        
        # Print Column Names to Debug.
        # print(f"CSV Columns: {self.annotations_df.columns.tolist()}")
        
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
        
        # Debug Print.
        #print(f"\nDebug - Annotations for {img_id}:")
        # print(f"Found {len(img_anns)} annotations")
        
        # Extract Boxes and Labels.
        boxes = []
        labels = []
        
        # Iterate Over Annotations.
        for _, ann in img_anns.iterrows():
            try:
                # Extract Box Coordinates.
                x1, y1, x2, y2 = float(ann['x1']), float(ann['y1']), float(ann['x2']), float(ann['y2'])
                
                # Ensure Box Coordinates are Valid.
                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(1)  # All Objects are Class 1
                else:
                    print(f"Warning: Skipping invalid box: {[x1, y1, x2, y2]}")
            except Exception as e:
                print(f"Error processing annotation: {ann}")
                print(f"Error: {str(e)}")
        
        # If No Valid Boxes, Return Zero Tensor.
        if not boxes:
            print(f"Warning: No valid boxes found for image {img_id}")
            return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(0, dtype=torch.long)
        
        # Return Boxes and Labels as Tensors.
        return (
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long)
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
        
        # Return Padded Image and Boxes.
        return padded, boxes
    
    # Get Item.
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""

        # Get Image ID.
        img_id = self.image_ids[idx]

        # Get Image Path.
        image_path = self.image_paths[img_id]
        
        # Debug Print.
        # print(f"\nProcessing image: {img_id}")
        
        # Load Image.
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get Original Image Dimensions.
        orig_h, orig_w = image.shape[:2]
        # print(f"Original image dimensions: {orig_w}x{orig_h}")
        
        # Get Boxes and Labels for This Image.
        boxes, labels = self.get_image_annotations(img_id)
        # print(f"Loaded {len(boxes)} boxes")
        
        if len(boxes) > 0:
            boxes = boxes.float()
            # print(f"Box ranges:")
            # print(f"Original: x=[{boxes[:, 0].min():.1f}-{boxes[:, 2].max():.1f}], y=[{boxes[:, 1].min():.1f}-{boxes[:, 3].max():.1f}]")

        # print(f"[Transform] Input boxes: {boxes.shape}, values: {boxes[:5]}")
        # print(f"[Transform] Input image size: {image.shape[:2]}")

        # Apply Transforms if Any.
        resize_size = None
        if self.transform:
            transformed = self.transform(
                image=image,
                boxes=boxes,
                labels=labels,
                is_train=(self.split == 'train')
            )

            # Get Transformed Image, Boxes, and Labels.
            image = transformed['image']
            boxes = torch.tensor(transformed.get('bboxes', []), dtype=torch.float32)
            labels = torch.tensor(transformed['labels'], dtype=torch.long)

            # Get resize dimensions from transform
            resize_size = self.resize_dims

            # print(f"[Transform] Output boxes: {transformed.get('bboxes', [])[:5]}")
            # print(f"[Transform] Output image size: {transformed['image'].shape}")


            # print(f"After transform: {len(boxes)} boxes")
            # if len(boxes) > 0:
            #     print(f"Box ranges after transform: x=[{boxes[:, 0].min():.3f}-{boxes[:, 2].max():.3f}], y=[{boxes[:, 1].min():.3f}-{boxes[:, 3].max():.3f}]")
        else:
            # If No Transform, Convert Image to Tensor and Normalize to [0, 1].
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        # Return Image, Boxes, Labels, and Image ID.
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'image_id': img_id,
            'orig_size': (orig_h, orig_w),  # Original size for proper scaling
            'resize_size': resize_size  # Add resize dimensions
        }
    
    # Collate Function.
    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Custom Collate Function for Batching."""
        images = torch.stack([item['image'] for item in batch])
        image_ids = [item['image_id'] for item in batch]
        
        # Pad Boxes and Labels to Same Length.
        max_boxes = max(len(item['boxes']) for item in batch)
        
        # Initialize Batch Boxes and Labels.
        batch_boxes = []
        batch_labels = []
        
        # Iterate Over Batch.
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
        
        # Return Image, Boxes, Labels, and Image IDs.
        return {
            'images': images,
            'boxes': boxes,
            'labels': labels,
            'image_ids': image_ids
        } 