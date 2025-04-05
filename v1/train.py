import torch
import yaml
from pathlib import Path
import sys
import argparse  # Add argument parser for debug mode
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from torch.serialization import add_safe_globals  # Add this import

sys.path.append(str(Path(__file__).resolve().parent.parent))  # Add src to path

from v1.model.detector import build_detector
from v1.data.dataset import SKU110KDataset
from v1.training.trainer import Trainer
from v1.utils.augmentation import DetectionAugmentation


class MetricsTracker:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_maps = []
        self.val_maps = []
        self.train_f1s = []
        self.val_f1s = []
        self.epochs = []
    
    def update(self, epoch, train_metrics, val_metrics):
        self.epochs.append(epoch)
        self.train_losses.append(train_metrics['loss'])
        self.val_losses.append(val_metrics['loss'])
        self.train_maps.append(train_metrics['mAP'])
        self.val_maps.append(val_metrics['mAP'])
        self.train_f1s.append(train_metrics['f1'])
        self.val_f1s.append(val_metrics['f1'])
    
    def plot_metrics(self, output_dir):
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Check if we have any metrics to plot
        if not self.epochs:
            print("No metrics to plot yet.")
            return
        
        # Plot loss curves
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_losses, label='Training Loss')
        plt.plot(self.epochs, self.val_losses, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / 'loss_curves.png')
        plt.close()
        
        # Plot mAP curves
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_maps, label='Training mAP')
        plt.plot(self.epochs, self.val_maps, label='Validation mAP')
        plt.title('Training and Validation mAP')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / 'map_curves.png')
        plt.close()
        
        # Plot F1 curves
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_f1s, label='Training F1')
        plt.plot(self.epochs, self.val_f1s, label='Validation F1')
        plt.title('Training and Validation F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / 'f1_curves.png')
        plt.close()
        
        # Save metrics to file
        metrics_file = output_dir / 'training_metrics.txt'
        with open(metrics_file, 'w') as f:
            f.write('Training Results Summary\n')
            f.write('=======================\n\n')
            if self.train_losses:  # Only write if we have metrics
                f.write(f'Final Training Loss: {self.train_losses[-1]:.4f}\n')
                f.write(f'Final Validation Loss: {self.val_losses[-1]:.4f}\n')
                f.write(f'Final Training mAP: {self.train_maps[-1]:.4f}\n')
                f.write(f'Final Validation mAP: {self.val_maps[-1]:.4f}\n')
                f.write(f'Final Training F1: {self.train_f1s[-1]:.4f}\n')
                f.write(f'Final Validation F1: {self.val_f1s[-1]:.4f}\n')
                
                f.write('\nTraining Observations:\n')
                f.write('---------------------\n')
                
                # Add observations about training behavior
                loss_diff = self.train_losses[-1] - self.val_losses[-1]
                if abs(loss_diff) > 0.3:
                    if loss_diff < 0:
                        f.write('- Potential underfitting: Training loss higher than validation loss\n')
                    else:
                        f.write('- Potential overfitting: Training loss lower than validation loss\n')
                
                # Check for convergence
                if len(self.train_losses) > 5:
                    recent_loss_change = abs(self.train_losses[-1] - self.train_losses[-5])
                    if recent_loss_change < 0.01:
                        f.write('- Model converged: Loss stabilized in recent epochs\n')
                    elif self.train_losses[-1] > self.train_losses[-5]:
                        f.write('- Warning: Loss increasing in recent epochs\n')
            else:
                f.write('No metrics available yet.\n')

# Register MetricsTracker as a safe global
add_safe_globals([MetricsTracker])

# Add GPU setup function
def setup_gpu():
    """Setup GPU device if available and print GPU information."""
    if torch.cuda.is_available():
        # Get the first available GPU
        device = torch.device('cuda:0')
        
        # Enable cuDNN auto-tuner
        torch.backends.cudnn.benchmark = True
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        # Print GPU information
        gpu_name = torch.cuda.get_device_name(device)
        gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        print(f"\nGPU Setup:")
        print(f"Device: {gpu_name}")
        print(f"Total Memory: {gpu_memory:.1f}GB")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Enabled: {torch.backends.cudnn.enabled}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
        return device
    else:
        print("\nNo GPU available, using CPU")
        return torch.device('cpu')

def main():
    # Add argument parser
    parser = argparse.ArgumentParser(description='Train object detector')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with smaller dataset')
    parser.add_argument('--num-images', type=int, default=64, help='Maximum number of training images to use')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--debug-loss', action='store_true', help='Print detailed loss computation info')
    parser.add_argument('--image-size', type=int, default=800, help='Input image size')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Directory containing checkpoints')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--confidence-threshold', type=float, default=0.01, help='Confidence threshold for predictions')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage even if GPU is available')
    parser.add_argument('--warmup-epochs', type=int, default=1, help='Number of warmup epochs for learning rate')
    args = parser.parse_args()

    # Create timestamp for this training run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(f"training_runs/{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n=== Starting Training Setup ===")
    
    # Get config
    config_path = Path(__file__).parent / "config" / "train_config.yaml"
    print(f"Loading config from: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Setup device (GPU/CPU)
    device = torch.device('cpu') if args.cpu else setup_gpu()
    print(f"\nUsing device: {device}")
    
    # Modify config based on arguments
    config["training"]["batch_size"] = args.batch_size
    config["training"]["epochs"] = args.epochs
    config["training"]["save_freq"] = 1  # Save every epoch when using small dataset
    config["training"]["num_workers"] = 0 if device.type == 'cpu' else 4  # More workers for GPU
    config["training"]["debug_loss"] = args.debug_loss
    config["training"]["image_size"] = args.image_size
    config["training"]["box_key"] = "bboxes"
    config["training"]["resume"] = args.resume
    config["training"]["checkpoint_dir"] = args.checkpoint_dir
    config["training"]["learning_rate"] = args.learning_rate
    config["training"]["warmup_epochs"] = args.warmup_epochs
    config["evaluation"]["conf_threshold"] = args.confidence_threshold
    
    if args.debug:
        print("\nRunning in debug mode!")
    
    print(f"\nTraining Configuration:")
    print(f"- Max Images: {args.num_images}")
    print(f"- Batch Size: {args.batch_size}")
    print(f"- Epochs: {args.epochs}")
    print(f"- Image Size: {args.image_size}")
    print(f"- Debug Loss: {args.debug_loss}")
    print(f"- Resume Training: {args.resume}")
    print(f"- Confidence Threshold: {args.confidence_threshold}")
    
    print("\n=== Setting up Data ===")
    
    # Create augmentation with smaller image size for faster training
    print("Creating data augmentation...")
    augmentation = DetectionAugmentation(
        height=args.image_size,
        width=args.image_size
    )
    
    # Create datasets
    print("\nLoading datasets...")
    print(f"Data directory: {config['data']['data_dir']}")
    
    try:
        train_dataset = SKU110KDataset(
            data_dir=config["data"]["data_dir"],
            split='train',
            transform=augmentation,
            resize_dims=(args.image_size, args.image_size)
        )
        
        val_dataset = SKU110KDataset(
            data_dir=config["data"]["data_dir"],
            split='val',
            transform=augmentation,
            resize_dims=(args.image_size, args.image_size)
        )
        
        print("Datasets loaded successfully!")
    except Exception as e:
        print(f"\nError loading datasets: {str(e)}")
        raise
    
    # Limit dataset size
    train_dataset.image_ids = train_dataset.image_ids[:args.num_images]
    val_dataset.image_ids = val_dataset.image_ids[:max(args.num_images//10, 5)]  # At least 5 validation images
    
    print(f"\nDataset sizes:")
    print(f"Training: {len(train_dataset)} images")
    print(f"Validation: {len(val_dataset)} images")
    print(f"Steps per epoch: {len(train_dataset) // args.batch_size}")
    
    print("\n=== Building Model ===")
    
    # Build model
    model_config = {
        'pretrained_backbone': config["model"]["pretrained_backbone"],
        'fpn_out_channels': config["model"]["fpn_out_channels"],
        'num_classes': config["model"]["num_classes"],
        'num_anchors': config["model"]["num_anchors"],
        'debug': args.debug_loss
    }
    
    try:
        # Initialize model
        print("Initializing model...")
        model = build_detector(model_config)
        print("Model initialized successfully!")
        
        # Print model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel Parameters:")
        print(f"Total: {total_params:,}")
        print(f"Trainable: {trainable_params:,}")
        
    except Exception as e:
        print(f"\nError initializing model: {str(e)}")
        raise
    
    print("\n=== Setting up Training ===")
    
    # Load checkpoint if resuming training
    start_epoch = 0
    checkpoint_dir = Path(args.checkpoint_dir)
    if args.resume and checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob('*.pth'))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
            print(f"\nLoading checkpoint: {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
            
            # Create and load metrics tracker
            metrics_tracker = MetricsTracker()
            if 'metrics_tracker' in checkpoint:
                loaded_tracker = checkpoint['metrics_tracker']
                metrics_tracker.epochs = loaded_tracker.epochs
                metrics_tracker.train_losses = loaded_tracker.train_losses
                metrics_tracker.val_losses = loaded_tracker.val_losses
                metrics_tracker.train_maps = loaded_tracker.train_maps
                metrics_tracker.val_maps = loaded_tracker.val_maps
                metrics_tracker.train_f1s = loaded_tracker.train_f1s
                metrics_tracker.val_f1s = loaded_tracker.val_f1s
                print("Loaded previous training metrics")
            else:
                print("No previous metrics found, starting fresh metrics tracking")
        else:
            print("\nNo checkpoints found, starting from scratch")
            metrics_tracker = MetricsTracker()
    else:
        print("Starting training from scratch")
        metrics_tracker = MetricsTracker()
    
    # Create debug directory
    debug_dir = Path("debug_output")
    debug_dir.mkdir(exist_ok=True)
    print(f"\nDebug output will be saved to: {debug_dir.absolute()}")
    
    # Create trainer with debug options
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config={
            **config["training"],
            "debug_options": {
                "print_box_stats": True,
                "print_anchor_stats": True,
                "print_loss_components": True,
                "print_gradient_stats": True,
                "save_debug_images": True,
                "debug_dir": str(debug_dir)
            },
            "metrics_tracker": metrics_tracker,
            "run_dir": str(run_dir),
            "start_epoch": start_epoch,
            "conf_threshold": args.confidence_threshold,
            "warmup_epochs": args.warmup_epochs
        },
        device=device
    )
    
    print("\n=== Starting Training ===")
    
    try:
        trainer.train()
        metrics_tracker.plot_metrics(run_dir)
        print(f"\nTraining completed successfully! Results saved to {run_dir}")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main() 