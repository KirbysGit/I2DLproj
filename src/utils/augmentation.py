import albumentations as A
from albumentations.pytorch import ToTensorV2

class DetectionAugmentation:
    """Augmentation pipeline for object detection."""
    
    def __init__(self, height=800, width=800):
        """Initialize augmentation pipelines."""
        
        # Common transformations for both train and val
        self.common_transform = A.Compose([
            A.Resize(height=height, width=width),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=1.0,  # Minimum area in pixels
            min_visibility=0.1,
            label_fields=['labels']
        ))
        
        # Training specific transformations - keep it simple initially
        self.train_transform = A.Compose([
            A.Resize(height=height, width=width),
            A.HorizontalFlip(p=0.5),  # Just horizontal flip for now
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=1.0,  # Minimum area in pixels
            min_visibility=0.1,
            label_fields=['labels']
        ))
        
        # Validation transform (just resize and normalize)
        self.val_transform = self.common_transform
        
        # Normalization values for reference (to be applied after visualization if needed)
        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225] 