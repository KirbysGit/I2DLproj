import albumentations as A
from albumentations.pytorch import ToTensorV2

class DetectionAugmentation:
    def __init__(self, height=800, width=800):
        self.train_transform = A.Compose([
            A.RandomResizedCrop(height=height, width=width, scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        
        self.val_transform = A.Compose([
            A.Resize(height=height, width=width),
            A.Normalize(),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])) 