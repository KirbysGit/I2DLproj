# Model Architecture

## Overview

Our retail product detection model is designed to handle the unique challenges of detecting products in retail shelf images, particularly focusing on dense object scenarios common in the SKU-110K dataset.

## Architecture Components

### 1. Backbone Network
- ResNet-based Feature Extraction
- Feature Pyramid Network (FPN) for detecting Objs of different sizes.
- Output Features at 4 scales (P2-P5)

### 2. Detection Head
```
Input Features → Shared Convs → Classification Branch
                            └→ Box Regression Branch
```

#### Key Components:
- **Shared Convolutions**: 4 Conv Layers w/ Consistent Channels
- **Classification Branch**: Predicts if the detected area contains a product or is just the background.
- **Box Regression Branch**: Predicts the coords of the detected product.

### 3. Anchor System
- Multi-scale Anchors Aligned w/ FPN levels
- Base sizes: [32, 64, 128, 256] Pixels
- Aspect ratios: [0.5, 1.0, 2.0]
- Single scale factor for efficiency

## Training Pipeline

### Data Processing
1. Image loading and resizing (800x800)
2. Augmentation pipeline:
   - Resize
   - Random crop
   - Normalization

### Target Assignment
1. Anchor generation per FPN level
2. IoU-based matching (threshold: 0.4)
3. Up to 20 matches per ground truth box

### Loss Computation
1. Classification Loss: Binary Cross Entropy
2. Box Regression Loss: Smooth L1 on normalized coordinates

## Current Performance *On Minimal Training*

- Training loss: ~0.32 (cls: 0.01, box: 0.31)
- Validation loss: ~0.34 (cls: 0.01, box: 0.32)
- Average IoU with ground truth: ~0.47-0.49

## Future Improvements

1. Feature Pyramid Attention
2. Additional anchor scales
3. Enhanced box regression
