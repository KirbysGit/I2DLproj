# data / augmentation.py

# -----

# Performs Data Augmentation for Obj Detection.
# Modifies Input Images & Bounding Boxes to Improve Model Robustness.
# Applies, Rezing, Horizontal FLipping, Color Jittering, Normalization.

# -----

# Imports.
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
import numpy as np
from typing import Dict, Tuple, List

# Data Augmentation Class.
class DataAugmentation:
    """Basic data augmentation for object detection."""
    
    def __init__(self,
                 min_size: int = 800,   
                 max_size: int = 1333,
                 flip_prob: float = 0.5,
                 brightness: float = 0.2,
                 contrast: float = 0.2,
                 saturation: float = 0.2,
                 hue: float = 0.1):
        """
        Args:
            min_size:     Minimum Size After Resize.
            max_size:     Maximum Size After Resize.
            flip_prob:    Probability of Horizontal Flip.
            brightness:   Brightness Jittering Range.
            contrast:     Contrast Jittering Range.
            saturation:   Saturation Jittering Range.
            hue:          Hue Jittering Range.
        """

        # Initialize Parameters.
        self.min_size = min_size 
        self.max_size = max_size
        self.flip_prob = flip_prob
        
        # Color Jittering.
        self.color_jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )
        
        # Normalization Parameters.
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    # Resize Function.
    def resize(self, image: torch.Tensor, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resize image and boxes maintaining aspect ratio."""
        
        # Get Original Image Size.
        orig_h, orig_w = image.shape[-2:]
        
        # Calculate New Size.
        min_orig = min(orig_h, orig_w)
        max_orig = max(orig_h, orig_w)
        scale = min(self.min_size / min_orig, self.max_size / max_orig)
        
        # Calculate New Height and Width.
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)
        
        # Resize Image.
        image = F.resize(image, [new_h, new_w])
        
        # Scale Boxes.
        boxes = boxes * torch.tensor([scale, scale, scale, scale], device=boxes.device)
        
        return image, boxes
    
    # Horizontal Flip Function.
    def horizontal_flip(self, image: torch.Tensor, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flip Image and Boxes Horizontally."""

        # Flip Image.
        image = F.hflip(image)
        
        # Flip Boxes: x1' = W - x2, x2' = W - x1.
        W = image.shape[-1]
        boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
        
        return image, boxes

    # Augmentation Function.
    def __call__(self, image: torch.Tensor, boxes: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply Augmentations to Image and Boxes.
        
        Args:
            image: [C, H, W] Image Tensor.
            boxes: [N, 4] Box Coordinates.
            labels: [N] Class Labels.
            
        Returns:
            dict: Augmented Image and Annotations.
        """

        # Resize.
        image, boxes = self.resize(image, boxes)
        
        # Random Horizontal Flip.
        if random.random() < self.flip_prob:
            image, boxes = self.horizontal_flip(image, boxes)
        
        # Color Jittering.
        image = self.color_jitter(image)
        
        # Normalize.
        image = self.normalize(image)
        
        # Return Augmented Image and Annotations.
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels
        } 