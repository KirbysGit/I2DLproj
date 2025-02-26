import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from dataset_loader import SKU110KDataset

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def visualize_sample(image, boxes, class_labels, title=None, save_path=None):
    """Visualize an image with its bounding boxes"""
    if isinstance(image, torch.Tensor):
        # Denormalize the image
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
        image = image * std + mean
        image = image.permute(1, 2, 0).cpu().numpy()
        
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    
    # Ensure image is in correct range
    image = np.clip(image, 0, 1)
    height, width = image.shape[:2]
    
    # Create figure with reasonable size
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    # Draw boxes
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        # Convert normalized coordinates to pixel coordinates
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
        
        rect = plt.Rectangle(
            (x1, y1), 
            x2 - x1, 
            y2 - y1,
            fill=False,
            color='lime',
            linewidth=1,
            alpha=0.8
        )
        plt.gca().add_patch(rect)
    
    plt.title(f"{title}\nBoxes: {len(boxes)}")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
        plt.close()
    else:
        plt.show()

def analyze_boxes(boxes, image_shape):
    """Analyze and print statistics about the boxes"""
    height, width = image_shape[:2]
    
    # Convert normalized coordinates to pixels
    boxes_pixels = boxes.copy()
    boxes_pixels[:, [0, 2]] *= width
    boxes_pixels[:, [1, 3]] *= height
    
    # Calculate box dimensions
    widths = boxes_pixels[:, 2] - boxes_pixels[:, 0]
    heights = boxes_pixels[:, 3] - boxes_pixels[:, 1]
    areas = widths * heights
    
    print("\nBox Statistics:")
    print(f"- Width (px): min={widths.min():.1f}, max={widths.max():.1f}, mean={widths.mean():.1f}")
    print(f"- Height (px): min={heights.min():.1f}, max={heights.max():.1f}, mean={heights.mean():.1f}")
    print(f"- Area (px²): min={areas.min():.1f}, max={areas.max():.1f}, mean={areas.mean():.1f}")

def test_dataset_loading():
    """Test dataset loading and visualization"""
    # Load configuration
    config = load_config('config/config.yaml')
    
    # Initialize dataset
    dataset = SKU110KDataset(config, split='train')
    
    print("\nDataset Loading Test Results:")
    print(f"Total samples: {len(dataset)}")
    
    # Test a few samples
    num_test_samples = 5
    save_dir = Path('test_results/dataset_visualization')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nTesting {num_test_samples} random samples...")
    
    for i in range(num_test_samples):
        try:
            # Get a random sample
            idx = np.random.randint(len(dataset))
            sample = dataset[idx]
            
            # Extract data
            image = sample['image']
            boxes = sample['boxes']
            class_labels = sample['class_labels']
            image_name = sample['image_name']
            
            # Print sample info
            print(f"\nSample {i+1}:")
            print(f"- Image: {image_name}")
            print(f"- Number of objects: {len(boxes)}")
            print(f"- Image shape: {tuple(image.shape)}")
            print(f"- Box coordinates shape: {tuple(boxes.shape)}")
            print(f"- Class labels shape: {tuple(class_labels.shape)}")
            
            # Add box analysis
            analyze_boxes(boxes.cpu().numpy(), image.permute(1, 2, 0).shape)
            
            # Visualize
            title = f"Sample {i+1}: {image_name}"
            save_path = save_dir / f"sample_{i+1}.png"
            
            visualize_sample(
                image, boxes, class_labels,
                title=title,
                save_path=save_path
            )
            
            print(f"- Visualization saved to: {save_path}")
            
        except Exception as e:
            print(f"Error processing sample {i+1}: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_dataset_loading() 