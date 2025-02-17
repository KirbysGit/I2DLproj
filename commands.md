# RETAIL PRODUCT DETECTION PROJECT - COMMAND REFERENCE
==================================================

## 1. ENVIRONMENT SETUP
-------------------
```bash
# Create virtual environment
python -m venv retail_analysis_env

# Activate virtual environment
## On Windows:
retail_analysis_env\Scripts\activate
## On Unix/MacOS:
source retail_analysis_env/bin/activate
```

# Install requirements
```bash
pip install -r requirements.txt
```

## 2. PROJECT INITIALIZATION
------------------------
```bash
# Create project structure and initial files
python src/setup_project.py
```

## 3. TESTING
----------
```bash
# Run basic model and dataset tests
python -m src.test_model
```

## 4. TRAINING
-----------
```bash
# Run full training pipeline
python -m src.train_main

# Run incremental training with stages
python -m src.train_control --stages 1  # Run one stage
python -m src.train_control --stages 3  # Run three stages
python -m src.train_control --stages 1 --evaluate  # Run one stage and evaluate
```

## 5. EVALUATION
----------------
```bash
# Run standalone evaluation
python -m src.evaluate
```

## 6. DEVELOPMENT UTILITIES
-------------------------
```bash
# Clean cached files
find . -type d -name "__pycache__" -exec rm -r {} +

# Create new directories if needed
mkdir -p datasets/SKU-110K/images/{train,val,test}
mkdir -p datasets/SKU-110K/annotations
```

## 7. COMMON WORKFLOWS
--------------------
```bash
# First-time setup:
python -m venv retail_analysis_env
retail_analysis_env\Scripts\activate  # Windows
pip install -r requirements.txt
python src/setup_project.py

# Development cycle:
python -m src.test_model  # Test changes
python -m src.train_control --stages 1  # Train one stage
python -m src.evaluate  # Evaluate results

# Production training:
python -m src.train_control --stages 10 --evaluate  # Train multiple stages
```

## 8. DEBUGGING
---------------
```bash
# Test dataset loading
python -m src.test_model

# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"
```

## 9. NOTES
------------
- Always activate virtual environment before running commands
- Use --help with any command for more options
- Training can be interrupted and resumed using checkpoints
- Use Ctrl+C to safely stop training at any point

# Training Commands

## Quick Training Options

### Test Mode (Fastest)
```bash
python -m src.quick_train --mode test --stages 1
```
- Uses only 5 images
- 1 epoch per stage
- Batch size of 2
- Best for testing code changes quickly

### Development Mode
```bash
python -m src.quick_train --mode dev --stages 1
```
- Uses 100 training images and 20 validation images
- 2 epochs per stage
- Batch size of 4
- Good for development and debugging

### Full Training Mode
```bash
python -m src.quick_train --mode full --stages 1
```
- Uses full dataset
- Normal training configuration
- For final model training

## Original Training Command
```bash
python -m src.train_control --stages N
```
- N is the number of stages to train
- Uses full configuration from config.yaml

## Dataset Organization
```bash
python -m src.organize_dataset
```
- Organizes dataset images into train/val/test directories
- Run this once before training

## Notes
- The `--stages` parameter controls how many training stages to run
- Each mode can be run with multiple stages using `--stages N`
- Test mode is fastest, dev mode is balanced, full mode is complete training
- Use test mode for quick code verification
- Use dev mode for algorithm development
- Use full mode for final training