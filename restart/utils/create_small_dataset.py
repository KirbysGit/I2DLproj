import os
import shutil
import pandas as pd
import random
from pathlib import Path
from tqdm import tqdm

def create_small_dataset(
    source_dir: str,
    target_dir: str,
    num_train: int = 1000,
    num_val: int = 200,
    num_test: int = 100
):
    """
    Create a smaller version of the SKU-110K dataset.
    
    Args:
        source_dir: Path to original SKU-110K dataset
        target_dir: Path to create small dataset
        num_train: Number of training images to include
        num_val: Number of validation images to include
        num_test: Number of test images to include
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    
    # Create directory structure
    print("Creating directory structure...")
    for split in ['train', 'val', 'test']:
        (target_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
    (target_dir / 'annotations').mkdir(parents=True, exist_ok=True)
    
    # Process each split
    splits = {
        'train': num_train,
        'val': num_val,
        'test': num_test
    }
    
    for split, num_images in splits.items():
        print(f"\nProcessing {split} split...")
        
        # Read original annotations
        ann_file = source_dir / 'annotations' / f'annotations_{split}.csv'
        df = pd.read_csv(ann_file, names=[
            'image_name', 'x1', 'y1', 'x2', 'y2', 'class', 'image_width', 'image_height'
        ])
        
        # Get unique image names
        unique_images = df['image_name'].unique()
        print(f"Found {len(unique_images)} original images")
        
        # Randomly select images
        selected_images = random.sample(list(unique_images), min(num_images, len(unique_images)))
        print(f"Selected {len(selected_images)} images")
        
        # Copy images and create new annotations
        new_annotations = []
        for img_name in tqdm(selected_images, desc=f"Copying {split} images"):
            # Copy image
            src_img = source_dir / 'images' / split / img_name
            dst_img = target_dir / 'images' / split / img_name
            shutil.copy2(src_img, dst_img)
            
            # Get annotations for this image
            img_anns = df[df['image_name'] == img_name]
            new_annotations.append(img_anns)
        
        # Save new annotations
        new_df = pd.concat(new_annotations)
        new_ann_file = target_dir / 'annotations' / f'annotations_{split}.csv'
        new_df.to_csv(new_ann_file, index=False, header=False)
        
        print(f"Completed {split} split:")
        print(f"- Images: {len(selected_images)}")
        print(f"- Annotations: {len(new_df)}")

def main():
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define paths
    source_dir = "datasets/SKU-110K"
    target_dir = "datasets/SKU-110K-Small"
    
    # Create small dataset
    create_small_dataset(
        source_dir=source_dir,
        target_dir=target_dir,
        num_train=1000,
        num_val=200,
        num_test=100
    )
    
    print("\nSmall dataset created successfully!")
    print(f"Output directory: {target_dir}")
    
    # Calculate total size
    total_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(target_dir)
                    for filename in filenames)
    print(f"Total size: {total_size / (1024*1024):.1f} MB")

if __name__ == "__main__":
    main() 