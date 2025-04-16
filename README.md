# Retail Product Detection Model

A PyTorch-based object detection framework specifically designed for detecting retail products in densely packed shelf images, developed for the Introduction to Deep Learning course.

## Dataset

The project uses the SKU-110K dataset, which contains retail shelf images with densely packed products. We provide two versions of the dataset via UCF OneDrive:

1. **Complete Dataset** (~30GB):
   **[Download Full SKU-110K Dataset](https://github.com/eg4000/SKU110K_CVPR19?tab=readme-ov-file)**
   - Training: 8,233 images
   - Validation: 588 images
   - Test: 2,941 images

2. **Small Dataset** (~4GB):
   **[Download Small SKU-110K Dataset](https://ucf-my.sharepoint.com/:u:/r/personal/co201115_ucf_edu/Documents/SKU-110K-Small.zip?csf=1&web=1&e=tzKZGI)**
   - Training: 1,000 images
   - Validation: 200 images
   - Test: 100 images
   - Perfect for quick experimentation and development

### Creating Your Own Small Dataset
If you have the full dataset, you can create your own small version:

```bash
# Create small dataset from full dataset
python restart/utils/create_small_dataset.py

# Optional: Customize the size
python restart/utils/create_small_dataset.py --train 500 --val 100 --test 50
```

This will create a new directory `datasets/SKU-110K-Small` with the reduced dataset.

### Dataset Structure
After downloading and extracting the dataset, organize it as follows:
```
datasets/
├── SKU-110K/
│   ├── images/
│   │   ├── train/          # Training images
│   │   ├── val/            # Validation images
│   │   └── test/           # Test images
│   └── annotations/
│       ├── annotations_train.csv
│       ├── annotations_val.csv
│       └── annotations_test.csv
└── sample_test/            # Sample images for quick testing
```

### Annotation Format
The CSV files contain the following columns:
```
image_name, x1, y1, x2, y2, class, image_width, image_height
```
Where:
- `image_name`: Name of the image file
- `x1, y1`: Top-left corner coordinates
- `x2, y2`: Bottom-right corner coordinates
- `class`: Always 1 (single class detection)
- `image_width, image_height`: Original image dimensions

### Quick Start with Sample Data
For quick testing without downloading the full dataset:
1. A small sample dataset is included in the repository under `datasets/sample_test/`
2. Contains 10 test images with annotations
3. Useful for testing the inference pipeline

### Setting Up the Dataset
1. Download the zip file from the OneDrive link above
2. Extract the contents to your project directory:
```bash
# Create dataset directory
mkdir -p datasets/SKU-110K

# Extract dataset
unzip SKU-110K_Dataset.zip -d datasets/SKU-110K

# Verify structure
tree datasets/SKU-110K -L 2
```

3. Verify the installation:
```bash
python restart/test/test_dataset.py
```
This will run basic checks to ensure the dataset is properly organized and readable.

## Process For Installation

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/KirbysGit/I2DLproj.git
cd retail-detection
```

2. Create and activate a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
   - Download from the provided OneDrive link
   - Extract to `datasets/SKU-110K/`
   - Ensure the following structure:
```

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


## Running the Code

### Training
1. Configure training parameters:
```bash
# Edit training configuration
vim restart/config/training_config.yaml
```

2. Start training:
```bash
python restart/train/trainer.py
```

Training outputs will be saved to `training_runs/TR_{timestamp}/`:
- Model checkpoints in `checkpoints/`
- Training visualizations in `visualizations/`
- Loss plots and metrics in root directory

### Evaluation
1. Configure evaluation parameters:
```bash
# Edit testing configuration
vim restart/config/testing_config.yaml
```

2. Run evaluation:
```bash
python restart/test.py
```

Evaluation outputs will be saved to `test_runs/eval_{model}_{timestamp}/`:
- Detection visualizations
- Precision-recall curves
- IoU histograms
- Detailed metrics summary