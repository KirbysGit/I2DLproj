from pathlib import Path

def get_training_config():
    return {
        # Data
        'data_dir': Path('data'),
        'train_csv': 'train.csv',
        'val_csv': 'val.csv',
        'image_size': 800,
        'batch_size': 8,
        'num_workers': 4,
        
        # Model
        'backbone': 'resnet50',
        'pretrained': True,
        'num_classes': 1,
        
        # Training
        'epochs': 50,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'grad_clip': 1.0,
        
        # Logging
        'save_dir': Path('checkpoints'),
        'save_freq': 5,
        'log_freq': 100,
        'visualize': True,
        
        # Validation
        'val_freq': 1,
        'iou_threshold': 0.5,
        'score_threshold': 0.05,
    } 