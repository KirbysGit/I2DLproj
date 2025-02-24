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

## Quick Testing
```bash
# Test with 20 images
python -m src.quick_train --mode custom --stages 1 --samples 20
```

## Development
```bash
# Train with 50 images
python -m src.quick_train --mode custom --stages 1 --samples 50
```

## Full Training
```bash
# Full dataset training
python -m src.quick_train --mode full --stages 1
```

## Configuration Modes
- `custom`: User-specified number of training images
- `dev`: 100 images for development
- `test`: 10 images for code testing
- `full`: Complete dataset

## Visualization
Results saved in:
- Training plots: `results/visualizations/`
- Logs: `results/logs/`
- Model checkpoints: `models/`