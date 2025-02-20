# Retail Product Detection Project

## Overview
This project implements a hybrid CNN-ViT (Vision Transformer) model for detecting retail products in densely packed shelf images using the SKU-110K dataset. The model combines the strengths of CNNs for feature extraction with Vision Transformers for understanding spatial relationships.

## Quick Links
- [Detailed Technical Approach](approach.md)
- [Command Reference](commands.md)

## Project Structure
```
retail_product_detection/
├── datasets/                  # Dataset storage
│   └── SKU-110K/             
│       ├── images/           # Image directories
│       │   ├── train/       
│       │   ├── val/         
│       │   └── test/        
│       └── annotations/       # Annotation files
├── src/                      # Source code
│   ├── dataset_loader.py     # Dataset handling
│   ├── model.py             # Model architecture
│   ├── trainer.py           # Training logic
│   └── utils.py             # Utility functions
├── config/                   # Configuration files
│   └── config.yaml          # Main configuration
├── models/                   # Saved models
└── results/                  # Training results
    └── logs/                # Training logs
```

## Getting Started

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv retail_analysis_env
source retail_analysis_env/bin/activate  # Unix/MacOS
retail_analysis_env\Scripts\activate     # Windows

# Install requirements
pip install -r requirements.txt
```

### 2. Dataset Organization
```bash
# Organize dataset into train/val/test splits
python -m src.organize_dataset
```

### 3. Training Options

#### Quick Testing
```bash
# Test mode (5 images, fast iteration)
python -m src.quick_train --mode test --stages 1
```

#### Development
```bash
# Dev mode (100 images, balanced)
python -m src.quick_train --mode dev --stages 1
```

#### Full Training
```bash
# Full training mode
python -m src.quick_train --mode full --stages 1
```

## Key Features
- Hybrid CNN-ViT architecture
- Incremental training support
- Multiple training modes (test/dev/full)
- Comprehensive logging and visualization
- Checkpoint management
- Performance metrics tracking

## Model Architecture
- **Backbone**: ResNet50 for feature extraction
- **Transformer**: Vision Transformer for spatial understanding
- **Detection Head**: Custom head for bounding box prediction

## Training Pipeline
1. Data preprocessing and augmentation
2. Feature extraction via CNN
3. Spatial processing via Vision Transformer
4. Object detection and localization
5. Loss computation and backpropagation
6. Validation and checkpoint management

## Performance Metrics
- F1 Score
- Precision
- Recall
- IoU (Intersection over Union)

## Documentation
- [Technical Approach](approach.md): Detailed explanation of methodology
- [Command Reference](commands.md): Complete list of available commands
- Configuration details in `config/config.yaml`

## Development Workflow
1. Use test mode for rapid prototyping
2. Switch to dev mode for algorithm validation
3. Run full training when satisfied with results
4. Monitor metrics via logging system
5. Analyze results in results directory


