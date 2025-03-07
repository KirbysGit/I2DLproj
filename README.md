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
project/
├── src/                    # Source code
│   ├── model/             # Model architecture and components
│   ├── data/              # Data handling and processing
│   ├── utils/             # Utility functions and helpers
│   ├── training/          # Training components
│   ├── config/            # Configuration files
│   ├── train.py           # Main training script
│   ├── evaluate.py        # Evaluation script
│   ├── dataset_loader.py  # Dataset loading utilities
│   └── run_tests.py       # Test runner
│
├── training_runs/         # Training run outputs and logs
├── docs/                  # Documentation
├── datasets/              # Dataset storage
├── requirements.txt       # Project dependencies
└── pytest.ini            # PyTest configuration
```

The project follows a modular structure where:
- `src/`: Contains all source code and implementation
  - `model/`: Neural network architecture and components
  - `data/`: Data processing and augmentation
  - `utils/`: Helper functions and utilities
  - `training/`: Training loop and optimization
  - `config/`: Configuration files for different runs
- `evaluation_results/`: Stores model evaluation metrics and visualizations
- `training_runs/`: Contains training logs and outputs
- `debug_output/`: Debug visualizations and outputs
- `test_results/`: Test execution outputs
- `docs/`: Project documentation
- `datasets/`: Location for dataset storage
- `checkpoints/`: Saved model weights and states

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

### Training
```bash
python -m src.train --num-images 256 --batch-size 16 --epochs 20
```

### Debug Mode
```bash
python -m src.train --debug --num-images 32 --batch-size 4 --epochs 5 --debug-loss
```

## Running the Model

### Training Options

1. **Standard Training**
```bash
python -m src.train \
    --epochs 20 \
    --batch-size 8 \
    --image-size 640 \
    --learning-rate 0.001 \
    --save-freq 5
```

2. **Training with Specific Configuration**
```bash
python -m src.train \
    --config configs/train_config.yaml \
    --run-name custom_run_1 \
    --resume-from checkpoints/latest.pth
```

3. **Debug Training**
```bash
python -m src.train \
    --debug \
    --num-images 32 \
    --batch-size 4 \
    --epochs 5 \
    --verbose \
    --debug-loss
```

4. **Distributed Training**
```bash
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    src/train.py \
    --distributed \
    --batch-size 16 \
    --epochs 20
```

### Evaluation Options

1. **Quick Test**
```bash
python -m src.evaluate \
    --debug \
    --num-images 10 \
    --confidence-threshold 0.15 \
    --nms-threshold 0.3
```

2. **Standard Evaluation**
```bash
python -m src.evaluate \
    --checkpoint checkpoints/best_model.pth \
    --num-images 100 \
    --confidence-threshold 0.5 \
    --nms-threshold 0.3
```

3. **Benchmark Evaluation**
```bash
python -m src.evaluate \
    --benchmark \
    --full-dataset \
    --batch-size 16 \
    --no-visualization
```

4. **Custom Evaluation**
```bash
python -m src.evaluate \
    --checkpoint checkpoints/custom.pth \
    --image-dir custom_images/ \
    --output-dir results/ \
    --save-visualizations
```

### Key Parameters

#### Training Parameters
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size for training
- `--image-size`: Input image size
- `--learning-rate`: Initial learning rate
- `--save-freq`: Checkpoint saving frequency
- `--resume-from`: Path to checkpoint to resume from
- `--run-name`: Custom name for the training run
- `--debug`: Enable debug mode
- `--verbose`: Enable verbose logging

#### Evaluation Parameters
- `--checkpoint`: Path to model checkpoint
- `--num-images`: Number of images to evaluate
- `--confidence-threshold`: Detection confidence threshold
- `--nms-threshold`: Non-maximum suppression threshold
- `--visualize`: Enable result visualization
- `--save-visualizations`: Save visualization results
- `--output-dir`: Directory to save results

### Example Workflows

1. **Complete Training Pipeline**
```bash
# 1. Start with debug training
python -m src.train --debug --num-images 32 --batch-size 4 --epochs 2

# 2. Run full training
python -m src.train --epochs 20 --batch-size 8 --image-size 640

# 3. Evaluate the model
python -m src.evaluate --checkpoint checkpoints/best_model.pth --num-images 100
```

2. **Experimental Setup**
```bash
# 1. Train with custom configuration
python -m src.train --config configs/experimental.yaml --run-name exp1

# 2. Debug evaluation
python -m src.evaluate --debug --num-images 10 --visualize

# 3. Full evaluation
python -m src.evaluate --full-dataset --batch-size 16
```

### Monitoring and Visualization

- Training metrics are saved in `training_runs/<run_name>/`
- Visualizations are saved in `debug_output/` when using `--debug` mode
- Evaluation results are saved in `results/` directory
- Use TensorBoard for real-time monitoring:
```bash
tensorboard --logdir training_runs/
```

## Current Challenges

1. **Dense Object Detection**: Handling tightly packed products with significant overlap.
2. **Training Efficiency**: Balancing model capacity with training speed.
3. **Memory Usage**: Managing large number of anchor boxes efficiently.