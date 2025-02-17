from pathlib import Path
import shutil
import os

def organize_dataset():
    # Setup paths
    base_dir = Path('datasets/SKU-110K')
    source_dir = base_dir / 'images'
    
    # Create destination directories
    train_dir = base_dir / 'images/train'
    val_dir = base_dir / 'images/val'
    test_dir = base_dir / 'images/test'
    
    # Create directories if they don't exist
    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("Moving files to their respective directories...")
    moved_counts = {'train': 0, 'val': 0, 'test': 0, 'unknown': 0}
    
    # Move files based on their prefixes
    for img_file in source_dir.glob('*.jpg'):
        if not img_file.is_file():  # Skip if it's a directory
            continue
            
        filename = img_file.name
        
        # Determine destination based on prefix
        if filename.startswith('train_'):
            dest_dir = train_dir
            moved_counts['train'] += 1
        elif filename.startswith('val_'):
            dest_dir = val_dir
            moved_counts['val'] += 1
        elif filename.startswith('test_'):
            dest_dir = test_dir
            moved_counts['test'] += 1
        else:
            print(f"Unknown file prefix: {filename}")
            moved_counts['unknown'] += 1
            continue
        
        # Move the file
        try:
            shutil.move(str(img_file), str(dest_dir / filename))
        except Exception as e:
            print(f"Error moving {filename}: {str(e)}")
    
    print("\nOrganization complete!")
    print(f"Moved {moved_counts['train']} training images")
    print(f"Moved {moved_counts['val']} validation images")
    print(f"Moved {moved_counts['test']} test images")
    if moved_counts['unknown'] > 0:
        print(f"Found {moved_counts['unknown']} files with unknown prefixes")

if __name__ == '__main__':
    # Confirm before proceeding
    print("This will organize your SKU-110K dataset images into train/val/test directories.")
    response = input("Proceed? (y/n): ")
    if response.lower() == 'y':
        organize_dataset()
    else:
        print("Operation cancelled.") 