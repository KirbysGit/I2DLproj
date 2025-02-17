import torch
from .dataset_loader import SKU110KDataset
from .model import CNNViTHybrid
from .utils import load_config
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def test_dataset():
    """
    Test dataset loading and visualization
    """
    print("Testing dataset loading...")
    
    # Load configuration
    config = load_config('config/config.yaml')
    
    # Create dataset instance
    dataset = SKU110KDataset(config, split='train')
    print(f"Dataset size: {len(dataset)} images")
    
    # Test loading a single sample
    sample = dataset[0]
    print("\nSample contents:")
    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            print(f"{k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"{k}: {v}")
    
    # Visualize sample with bounding boxes
    image = sample['image']
    boxes = sample['boxes']
    
    # Convert image for visualization (handle both CHW and HWC formats)
    if image.shape[0] == 3:  # If in CHW format
        image = image.transpose(1, 2, 0)
    
    plt.figure(figsize=(12, 8))
    plt.imshow(image)  # Matplotlib expects HWC format
    
    # Draw bounding boxes
    for box in boxes:
        x1, y1, x2, y2 = box
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'r-', linewidth=2)
    
    plt.title(f"Sample Image: {sample['image_name']}")
    plt.axis('off')
    plt.savefig('results/sample_visualization.png')
    print("\nSample visualization saved to results/sample_visualization.png")

def test_model():
    """
    Test model architecture and forward pass
    """
    print("\nTesting model architecture...")
    
    # Load configuration
    config = load_config('config/config.yaml')
    
    # Create model instance
    model = CNNViTHybrid(config)
    print("\nModel architecture:")
    print(model)
    
    # Test forward pass
    batch_size = 2
    input_shape = config['preprocessing']['image_size']
    dummy_input = torch.randn(batch_size, 3, input_shape[0], input_shape[1])
    
    print(f"\nTesting forward pass with input shape: {dummy_input.shape}")
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print("\nOutput contains:")
    print(f"- Bounding box coordinates (x1, y1, x2, y2): {output[..., :4].shape}")
    print(f"- Objectness score: {output[..., 4].shape}")

def main():
    """
    Run all tests
    """
    # Create required directories if they don't exist
    import os
    os.makedirs('results', exist_ok=True)
    os.makedirs('datasets/SKU-110K/images/train', exist_ok=True)
    os.makedirs('datasets/SKU-110K/images/val', exist_ok=True)
    os.makedirs('datasets/SKU-110K/images/test', exist_ok=True)
    os.makedirs('datasets/SKU-110K/annotations', exist_ok=True)
    
    try:
        test_dataset()
    except Exception as e:
        print(f"Dataset test failed: {str(e)}")
        print(f"Error details: {type(e).__name__}")
        import traceback
        traceback.print_exc()
    
    try:
        test_model()
    except Exception as e:
        print(f"Model test failed: {str(e)}")
        print(f"Error details: {type(e).__name__}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 