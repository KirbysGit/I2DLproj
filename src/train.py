import torch
import yaml
from pathlib import Path
import sys
import argparse  # Add argument parser for debug mode

sys.path.append(str(Path(__file__).resolve().parent.parent))  # Add src to path

from src.model.detector import build_detector
from src.data.dataset import SKU110KDataset
from src.training.trainer import Trainer
from src.utils.augmentation import DetectionAugmentation

def main():
    # Add argument parser
    parser = argparse.ArgumentParser(description='Train object detector')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode with smaller dataset')
    parser.add_argument('--num-images', type=int, default=64, help='Maximum number of training images to use')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--debug-loss', action='store_true', help='Print detailed loss computation info')
    parser.add_argument('--image-size', type=int, default=800, help='Input image size')
    args = parser.parse_args()

    # Get config
    config_path = Path(__file__).parent / "config" / "train_config.yaml"
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Modify config based on arguments
    config["training"]["batch_size"] = args.batch_size
    config["training"]["epochs"] = args.epochs
    config["training"]["save_freq"] = 1  # Save every epoch when using small dataset
    config["training"]["num_workers"] = 0  # Easier debugging with small dataset
    config["training"]["debug_loss"] = args.debug_loss  # Add loss debugging flag
    config["training"]["image_size"] = args.image_size  # Allow image size adjustment
    
    if args.debug:
        print("\nRunning in debug mode!")
    
    print(f"\nTraining Configuration:")
    print(f"- Max Images: {args.num_images}")
    print(f"- Batch Size: {args.batch_size}")
    print(f"- Epochs: {args.epochs}")
    print(f"- Image Size: {args.image_size}")
    print(f"- Debug Loss: {args.debug_loss}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create augmentation with smaller image size for faster training
    augmentation = DetectionAugmentation(
        height=args.image_size,
        width=args.image_size
    )
    
    # Create datasets
    train_dataset = SKU110KDataset(
        data_dir=config["data"]["data_dir"],
        split='train',
        transform=augmentation.train_transform
    )
    
    val_dataset = SKU110KDataset(
        data_dir=config["data"]["data_dir"],
        split='val',
        transform=augmentation.val_transform
    )
    
    # Limit dataset size
    train_dataset.image_ids = train_dataset.image_ids[:args.num_images]
    val_dataset.image_ids = val_dataset.image_ids[:max(args.num_images//10, 5)]  # At least 5 validation images
    
    print(f"\nDataset sizes:")
    print(f"Training: {len(train_dataset)} images")
    print(f"Validation: {len(val_dataset)} images")
    print(f"Steps per epoch: {len(train_dataset) // args.batch_size}")
    
    # Build model
    model_config = {
        'pretrained_backbone': config["model"]["pretrained_backbone"],
        'fpn_out_channels': config["model"]["fpn_out_channels"],
        'num_classes': config["model"]["num_classes"],
        'num_anchors': config["model"]["num_anchors"],
        'debug': args.debug_loss  # Add debug flag from arguments
    }
    
    # Initialize model
    print("\nInitializing model...")
    model = build_detector(model_config)
    print("Model initialized successfully!")
    
    # Create trainer
    debug_dir = Path("debug_output")
    debug_dir.mkdir(exist_ok=True)
    print(f"\nDebug output will be saved to: {debug_dir.absolute()}")
    
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config={
            **config["training"],
            "debug_options": {
                "print_box_stats": True,        # Print statistics about box coordinates
                "print_anchor_stats": True,     # Print statistics about generated anchors
                "print_loss_components": True,  # Print detailed loss computation
                "print_gradient_stats": True,   # Print statistics about gradients
                "save_debug_images": True,      # Save debug visualizations
                "debug_dir": str(debug_dir)     # Specify debug output directory
            }
        },
        device=device
    )
    
    print("\nStarting training...")
    # Train
    try:
        trainer.train()
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

if __name__ == '__main__':
    main() 