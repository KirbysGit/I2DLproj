# Object Detection Training Framework

A modular PyTorch-based framework for training object detection models with focal loss and IoU-based regression.

## Project Structure

src/
├── model/ # Model architecture and losses
│ ├── detector.py # Main detector implementation
│ ├── losses.py # Focal and IoU losses
│ └── test_.py # Model unit tests
├── training/ # Training infrastructure
│ ├── trainer.py # Training loop implementation
│ ├── optimizer.py # Optimizer configuration
│ ├── scheduler.py # Learning rate scheduling
│ └── test_.py # Training unit tests
└── data/ # (Coming soon) Data processing
├── target_generator.py
├── augmentation.py
└── transforms.py

## Key Features

### Detection Model
- Multi-scale feature detection
- Focal Loss for classification
- IoU Loss for box regression
- Configurable anchor system

### Training Infrastructure
- Modular training loop
- Automatic device selection (CPU/GPU)
- Gradient clipping
- Checkpoint management
- Learning rate warmup and cosine decay
- Proper weight decay configuration

## Installation

Clone repository

```bash
git clone <repository_url>
cd object_detection_framework
```

Install dependencies

## Usage

