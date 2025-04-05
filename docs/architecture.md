# Model Architecture

## Overview

Our retail product detection model is designed to handle the unique challenges of detecting products in retail shelf images, particularly focusing on dense object scenarios common in the SKU-110K dataset.

## Architecture Components

### 1. Backbone Network (ResNetBackbone)
- ResNet50-based feature extraction with ImageNet pre-training
- Feature Pyramid Network (FPN) integration
- Outputs features at multiple scales:
  - Layer1 (P2): 256 channels, 1/4 scale
  - Layer2 (P3): 512 channels, 1/8 scale
  - Layer3 (P4): 1024 channels, 1/16 scale
  - Layer4 (P5): 2048 channels, 1/32 scale

### 2. Feature Pyramid Network (FPN)
```
Input Features → Lateral Connections → Top-Down Pathway → Output Features
```

#### Key Components:
- **Lateral Connections**: 1x1 convolutions to unify channel dimensions
- **Top-Down Pathway**: Upsampling and feature fusion
- **Output Convolutions**: 3x3 convs for feature refinement
- **Output Features**: P2-P5 levels with unified 256 channels

### 3. Detection Head
```
Input Features → Shared Convs → Classification Branch
                            └→ Box Regression Branch
```

#### Components:
- **Shared Convolutions**: 4 Conv layers (3x3) with BatchNorm and ReLU
- **Classification Branch**: Binary prediction (object vs. background)
- **Box Regression Branch**: 4D box coordinate refinements
- **Feature Processing**: Maintains spatial resolution with padding

### 4. Anchor System
- **Base Sizes**: [32, 64, 128, 256, 512] pixels
- **Aspect Ratios**: [0.3, 0.5, 1.0, 2.0, 3.0]
- **Scales**: [0.5, 0.75, 1.0, 1.25]
- **Dynamic Generation**: Per-level anchors based on feature map size
- **Center Sampling**: Improved matching for dense scenarios

## Training Pipeline

### Data Processing
1. **Dataset Loading**:
   - SKU110K dataset with train/val/test splits
   - Dynamic resizing with aspect ratio preservation
   - Box coordinate normalization

2. **Augmentation Pipeline**:
   - Resize with maintained aspect ratio
   - Box coordinate adjustment
   - Normalization to [0,1] range

### Target Assignment
1. **Anchor Matching**:
   - IoU-based matching (threshold: 0.3)
   - Center-based sampling for dense objects
   - Positive/negative sample balancing

### Loss Computation
1. **Classification Loss**: 
   - Binary Cross Entropy
   - Temperature scaling for better calibration
   - Confidence thresholding: 0.2

2. **Box Regression Loss**:
   - Smooth L1 Loss
   - Delta clipping for stability
   - Box coordinate normalization

### Optimization
1. **Training Strategy**:
   - Learning rate: 0.001 with scheduling
   - Batch size: 16
   - Warmup epochs: 5
   - LR decay epochs: [30, 40]

2. **Model Checkpointing**:
   - Regular saving of model states
   - Visualization of training progress
   - Metric tracking and logging

## Inference Pipeline

### Detection Process
1. **Feature Extraction**:
   - Multi-scale feature computation
   - FPN feature fusion

2. **Box Prediction**:
   - Anchor-based detection
   - NMS threshold: 0.5
   - Confidence threshold: 0.2
   - Max predictions per image: 300

3. **Post-processing**:
   - Box coordinate denormalization
   - Size-based filtering
   - Overlap removal (IoU > 0.6)

## Performance Metrics

### Evaluation Metrics
- Mean Average Precision (mAP)
- Precision-Recall curves
- IoU distribution analysis
- F1 Score computation

### Visualization Tools
- Detection visualization with confidence scores
- IoU histogram analysis
- Training loss curves
- Per-epoch metric tracking

## Current Performance

- Training loss convergence: ~0.3-0.4
- Classification accuracy: >90%
- Average IoU with ground truth: ~0.5
- Mean Average Precision: Varies by configuration

## Future Improvements

1. **Architecture Enhancements**:
   - Feature pyramid attention mechanisms
   - Multi-scale training improvements
   - Anchor optimization for dense scenarios

2. **Training Optimizations**:
   - Advanced augmentation strategies
   - Curriculum learning implementation
   - Loss function refinements

3. **Memory Efficiency**:
   - Anchor generation optimization
   - Feature map memory reduction
   - Batch processing improvements
