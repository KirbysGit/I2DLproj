# src / utils / augmentation.py

# -----

# Defines Augmentation Pipeline for Object Detection.
# Applies Common Transformations for Training & Validation.
# Ensures Boxes Remain Normalized After Augmentation.

# -----

# Imports.
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ----------------------------
# Detection Augmentation Class.
# ----------------------------
class DetectionAugmentation:
    """Augmentation pipeline for object detection."""
    
    def __init__(self, height=800, width=800):
        """Initialize augmentation pipelines."""
        
        # Common Transformations.
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
            min_area=0.0,
            min_visibility=0.0,
            label_fields=['labels']
        ))
        
        # Training Specific Transformations.
        self.train_transform = A.Compose([
            A.Resize(height=height, width=width),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.0, rotate_limit=5, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0.0,
            min_visibility=0.0,
            label_fields=['labels']
        ))
        
        # Validation Transform (Just Resize & Normalize).
        self.val_transform = self.common_transform
        
        # Save Dimensions for Reference.
        self.height = height
        self.width = width
    
    def __call__(self, image, boxes=None, labels=None, is_train=True):
        """Apply transformation and ensure boxes stay normalized."""
        # Input Validation.

        # print(f"🚨 Incoming to __call__: boxes={type(boxes)}, len={len(boxes) if boxes is not None else 'None'}, labels={len(labels) if labels is not None else 'None'}")

        # Apply Transformations.
        transform = self.train_transform if is_train else self.val_transform

        if boxes is None or len(boxes) == 0:
            transform = self.train_transform if is_train else self.val_transform
            result = transform(image=image, bboxes=[], labels=[])
            result['bboxes'] = np.array(result.get('bboxes', []))  # ← Fallback!
            result['labels'] = result.get('labels', [])
            return result

        # ✅ Fix: Check type before copying
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.numpy()
        else:
            boxes = boxes.copy()

        # Print Debug Info.
        # print("\nDebug - Box Transformation:")
        # print(f"Input box ranges: x=[{boxes[:, 0].min():.3f}-{boxes[:, 2].max():.3f}], "
        #       f"y=[{boxes[:, 1].min():.3f}-{boxes[:, 3].max():.3f}]")
        
        # Convert Normalized Boxes to Pixel Coordinates for Albumentations
        h, w = image.shape[:2]

        # Detect whether boxes are normalized or pixel already
        is_normalized = np.all(boxes[:, :4] <= 1.0)
        if is_normalized:
            boxes_pixel = boxes.copy()
            boxes_pixel[:, [0, 2]] *= w
            boxes_pixel[:, [1, 3]] *= h
            # print("🔎 Boxes were normalized. Scaled to pixel coords.")
        else:
            boxes_pixel = boxes.copy()
            # print("🔎 Boxes are already in pixel coords.")

        boxes_pixel[:, [0, 2]] = np.clip(boxes_pixel[:, [0, 2]], 0, w - 1e-3)
        boxes_pixel[:, [1, 3]] = np.clip(boxes_pixel[:, [1, 3]], 0, h - 1e-3)
        
        # Print Pixel Box Ranges.
        # print(f"Pixel Box Ranges: x=[{boxes_pixel[:, 0].min():.1f}-{boxes_pixel[:, 2].max():.1f}], "
        #       f"y=[{boxes_pixel[:, 1].min():.1f}-{boxes_pixel[:, 3].max():.1f}]")

        # ✅ Remove degenerate boxes (x2 <= x1 or y2 <= y1)
        box_widths = boxes_pixel[:, 2] - boxes_pixel[:, 0]
        box_heights = boxes_pixel[:, 3] - boxes_pixel[:, 1]
        valid_mask = (box_widths > 1.0) & (box_heights > 1.0)
        
        if not np.any(valid_mask):
            # print("⚠️ All boxes degenerate after clipping. Returning empty transform.")
            result = transform(image=image, bboxes=[], labels=[])
            result['bboxes'] = np.array(result.get('bboxes', []))
            result['labels'] = result.get('labels', [])
            return result

        boxes_pixel = boxes_pixel[valid_mask]
        labels = np.array(labels)[valid_mask].tolist()
        labels = [int(label) for label in labels]  # ensure ints

        #print(f"✅ Boxes remaining after filtering: {len(boxes_pixel)}")

        try:
            # ✅ Ensure labels are list of ints BEFORE transform call
            if isinstance(labels, torch.Tensor):
                labels = labels.tolist()
            labels = [int(label) for label in labels]
            
            # print(f"Number of boxes BEFORE transform: {len(boxes_pixel)}")

            result = transform(image=image, bboxes=boxes_pixel, labels=labels)

            #print(f"Number of boxes AFTER transform: {len(result['bboxes'])}")
            if len(result['bboxes']) == 0:
                print("⚠️ All boxes removed by albumentations! Possibly due to invalid coordinates after transform.")

            
            # Convert Boxes Back to Normalized Coordinates.
            if len(result['bboxes']) > 0:
                result['bboxes'] = np.array(result['bboxes'])
                
                # Print Transformed Pixel Coordinates.
                #print(f"Transformed Pixel Ranges: x=[{result['bboxes'][:, 0].min():.1f}-{result['bboxes'][:, 2].max():.1f}], "
                #      f"y=[{result['bboxes'][:, 1].min():.1f}-{result['bboxes'][:, 3].max():.1f}]")
                
                # Normalize Coordinates.
                result['bboxes'][:, [0, 2]] /= float(self.width)
                result['bboxes'][:, [1, 3]] /= float(self.height)

                
                # Ensure Boxes are Within [0, 1] Range.
                result['bboxes'] = np.clip(result['bboxes'], 0, 1)
                
                # Print Final Normalized Coordinates.
                # print(f"Final Normalized Ranges: x=[{result['bboxes'][:, 0].min():.3f}-{result['bboxes'][:, 2].max():.3f}], "
                #       f"y=[{result['bboxes'][:, 1].min():.3f}-{result['bboxes'][:, 3].max():.3f}]")
                
                # Validate Box Coordinates.
                assert np.all(result['bboxes'] >= 0) and np.all(result['bboxes'] <= 1), \
                    "Box Coordinates Outside [0,1] Range After Normalization"
                assert np.all(result['bboxes'][:, 2] > result['bboxes'][:, 0]), \
                    "Invalid Box Width (x2 <= x1)"
                assert np.all(result['bboxes'][:, 3] > result['bboxes'][:, 1]), \
                    "Invalid Box Height (y2 <= y1)"
            
            return result
            
        except Exception as e:
            print(f"Error During Transformation: {str(e)}")
            print(f"Image Shape: {image.shape}")
            print(f"Boxes Shape: {boxes_pixel.shape}")
            print(f"Box ranges before transform: x=[{boxes_pixel[:, 0].min():.1f}-{boxes_pixel[:, 2].max():.1f}], "
                  f"y=[{boxes_pixel[:, 1].min():.1f}-{boxes_pixel[:, 3].max():.1f}]")
            raise 