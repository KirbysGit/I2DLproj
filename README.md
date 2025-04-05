# Retail Product Detection Model

A PyTorch-based object detection framework specifically designed for detecting retail products in densely packed shelf images. This project implements a custom object detection model with features tailored for retail environments.

## Project Overview

This project focuses on developing an object detection system for the SKU-110K dataset, which contains challenging retail shelf images with densely packed products. Our model is designed to handle:
- Dense object detection scenarios
- Multiple objects with similar appearances
- Varying scales of products
- Overlapping items

### Current Development Status

✅ Completed:
- Basic model architecture implementation
- Anchor generation and matching system
- Loss functions (classification and box regression)
- Training pipeline with validation
- Initial data loading and augmentation

🔄 In Progress:
- Improving model performance on dense object scenarios
- Enhancing visualization tools
- Documentation and code organization

## Project Structure
```
i2dlproj/
├── v1/                       # Legacy version (initial baseline implementation)
│   ├── ...                   # Old source code and training pipeline
│
├── restart/                 # Current implementation (modular, improved architecture)
│   ├── model/               # Core model components: backbone, FPN, detection head, anchors
│   │   ├── anchor_generator.py
│   │   ├── backbone.py
│   │   ├── detection_head.py
│   │   ├── detector.py
│   │   └── fpn.py
│   │ 
│   ├── data/                # Dataset loader and retail image preprocessing
│   │   └── dataset.py
│   │
│   ├── utils/               # Helper functions for box operations and plotting
│   │   ├── box_ops.py
│   │   ├── plots.py
│   │   └── visualize_detections.py
│   │
│   ├── train/               # Training loop and training logic
│   │   └── trainer.py
│   │
│   ├── test/                # Unit tests for model components
│   │   ├── test_anchor_coverage.py
│   │   ├── test_anchor_generator.py
│   │   ├── test_anchor_matching.py
│   │   ├── test_box_iou.py
│   │   ├── test_dataset.py
│   │   ├── test_detection_head.py
│   │   ├── test_detector.py
│   │   ├── test_overfitting.py
│   │   └── test_pipeline.py
│   │
│   ├── config/              # YAML configuration files for training/evaluation
│   │   ├── testing_config.yaml
│   │   └── training_config.yaml
│   │
│   ├── test.py              # Evaluation entry point for running tests
│   └── compareSOTA.py       # Evaluation script for benchmarking against SOTA (YOLOv5)
│
├── training_runs/           # Saved training checkpoints, logs, and visualizations
│   └── TR_{timestamp}/
│       ├── checkpoints/
│       ├── visualizations/
│       └── training_loss.png
│
├── test_runs/               # Evaluation outputs for specific model runs
│   └── eval_{model}_{timestamp}/
│       ├── metrics/
│       ├── visualizations/
│       └── eval_config.yaml
│
├── comparison_results/      # YOLOv5 vs ShelfVision comparison outputs
├── checkpoints/             # Manually saved model weights
├── debug_output/            # Intermediate debug images and logs
├── test_results/            # Unit test output and logs
├── docs/                    # Project documentation, slides, and notes
└── requirements.txt         # Python dependencies list


```

The project follows a modular structure where:
- `restart/`: Contains all source code and modular implementation for the latest model version
  - `model/`: Backbone, FPN, anchors, and detection logic
  - `data/`: Dataset loading and preprocessing
  - `utils/`: Visualization and box operations
  - `train/`: Core training loop logic
  - `test/`: Unit tests for model components
  - `config/`: Config files for training and testing
  - `compareSOTA.py` and `test.py`: Entry points for evaluation
- `test_runs/`: Stores model evaluation results, metrics, and visualizations
- `comparison_results/`: Output comparisons against YOLOv5 baseline models
- `training_runs/`: Contains training logs and outputs
- `debug_output/`: Debug visualizations and outputs
- `test_results/`: Test execution outputs
- `docs/`: Project documentation
- `checkpoints/`: Saved model weights and states

⚠️ **Legacy Note**: The `v1/` folder contains the original baseline version used in early development (Eval 1). 
The `restart/` directory includes the updated model logic, modular pipeline, and improvements featured in Eval 2.

## Key Features

### Model Architecture
- Feature Pyramid Network (FPN) backbone
- Multi-scale detection heads
- Anchor-based detection system
- Binary classification (object vs. background)

### Training Features
- Batch-based training with validation
- IoU-based anchor matching
- Smooth L1 loss for box regression
- Binary Cross Entropy for classification
- Learning rate scheduling

## Installation

1. Clone the repository:
```bash
git clone <repository_url>
cd retail-detection
```

2. Create and activate a virtual environment:

For Windows:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate
```

For Linux/Mac:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the SKU-110K dataset and place it in the `datasets` directory.

## Usage

The project uses YAML configuration files to manage training and testing parameters. The main workflow involves:

1. Adjusting the configuration files in `restart/config/`:
   - `training_config.yaml`: Training parameters and model settings
   - `testing_config.yaml`: Evaluation and testing parameters

2. Running the training or testing scripts:

### Training
```bash
# Edit restart/config/training_config.yaml first to set your parameters
python restart/train/trainer.py
```

### Testing/Evaluation
```bash
# Edit restart/config/testing_config.yaml first to set your parameters
python restart/test.py
```

## Configuration Files

### Training Configuration
Key parameters in `restart/config/training_config.yaml`:
```yaml
# Model Parameters
model:
  backbone: "resnet50"
  num_classes: 1
  num_anchors: 9
  pretrained: true

# Training Parameters
training:
  batch_size: 16
  num_epochs: 20
  learning_rate: 0.001
  save_freq: 5
  num_workers: 4

# Dataset Parameters
dataset:
  image_size: [640, 640]
  train_split: 0.8
  augmentation: true

# Output Settings
output_dir: "training_runs"
checkpoint_dir: "checkpoints"
```

### Testing Configuration
Key parameters in `restart/config/testing_config.yaml`:
```yaml
# Model Parameters
checkpoint_path: "checkpoints/best_model.pth"
confidence_threshold: 0.5
nms_threshold: 0.3

# Evaluation Parameters
num_images: 100
batch_size: 16
visualize: true

# Output Settings
output_dir: "test_runs"
```

### Example Workflows

1. **Training Pipeline**
```bash
# 1. Edit training configuration
vim restart/config/training_config.yaml

# 2. Run training
python restart/train/trainer.py

# 3. Edit testing configuration
vim restart/config/testing_config.yaml

# 4. Run evaluation
python restart/test.py
```

2. **Evaluation Only**
```bash
# 1. Edit testing configuration to point to your model checkpoint
vim restart/config/testing_config.yaml

# 2. Run evaluation
python restart/test.py
```

### Monitoring and Visualization

- Training metrics are saved in `training_runs/<run_name>/`
- Visualizations are saved in `debug_output/` when debug mode is enabled in config
- Evaluation results are saved in `test_runs/` directory
- Use TensorBoard for real-time monitoring:
```bash
tensorboard --logdir training_runs/
```