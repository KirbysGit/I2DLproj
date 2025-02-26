Object Detection Training Framework
=================================

A modular PyTorch-based framework for training object detection models with focal loss 
and IoU-based regression.

Project Structure
---------------
src/
|-- model/                  # Model architecture and losses
|   |-- detector.py         # Main detector implementation
|   |-- losses.py          # Focal and IoU losses
|   `-- test_*.py          # Model unit tests
|-- training/              # Training infrastructure
|   |-- trainer.py         # Training loop implementation
|   |-- optimizer.py       # Optimizer configuration
|   |-- scheduler.py       # Learning rate scheduling
|   `-- test_*.py         # Training unit tests
`-- data/                  # (Coming soon) Data processing
    |-- target_generator.py
    |-- augmentation.py
    `-- transforms.py

Key Features
-----------

Detection Model:
* Multi-scale feature detection
* Focal Loss for classification
* IoU Loss for box regression
* Configurable anchor system

Training Infrastructure:
* Modular training loop
* Automatic device selection (CPU/GPU)
* Gradient clipping
* Checkpoint management
* Learning rate warmup and cosine decay
* Proper weight decay configuration

Installation
-----------
1. Clone repository:
   git clone <repository_url>
   cd object_detection_framework

2. Install dependencies:
   pip install torch torchvision pytest tqdm

Running Tests
------------
Run all tests:
   python -m pytest src/ -v

Run specific test files:
   python -m pytest src/training/test_training.py -v
   python -m pytest src/model/test_losses.py -v
   python -m pytest src/model/test_detector.py -v

Training Configuration Example
---------------------------
config = {
    'optimizer_type': 'adam',
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'epochs': 100,
    'save_freq': 5,
    'save_dir': 'checkpoints',
    'grad_clip': 1.0
}

Usage Example
-----------
from training.trainer import Trainer
from training.optimizer import OptimizerBuilder
from training.scheduler import WarmupCosineScheduler
from model.losses import DetectionLoss

# Initialize components
model = YourDetectionModel()
criterion = DetectionLoss(num_classes=num_classes)
optimizer = OptimizerBuilder.build(model, config)
scheduler = WarmupCosineScheduler(
    optimizer,
    warmup_epochs=5,
    max_epochs=config['epochs']
)

# Create trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    config=config
)

# Start training
trainer.train()

Development Status
----------------
Completed:
✓ Model architecture
✓ Loss functions
✓ Training infrastructure
✓ Optimizer configuration
✓ Learning rate scheduling
✓ Basic testing framework

In Progress:
- Data processing pipeline
- Evaluation metrics
- Configuration system
- Visualization tools