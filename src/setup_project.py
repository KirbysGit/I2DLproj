import os
import shutil

def create_directory_structure():
    # Define the directory structure (excluding data directories that already exist)
    directories = [
        'notebooks',
        'models',
        'src',
        'config',
        'results/logs'
    ]
    
    # Create directories
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

    # Create symbolic links to existing dataset
    data_dirs = {
        'data/raw': 'datasets/SKU-110K',
        'data/annotations': 'datasets/SKU-110K/annotations'
    }
    
    for target, source in data_dirs.items():
        if os.path.exists(source):
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if os.path.exists(target):
                print(f"Link already exists: {target}")
            else:
                # On Windows, might need administrator privileges for symlinks
                try:
                    os.symlink(os.path.abspath(source), target, target_is_directory=True)
                    print(f"Created link: {target} -> {source}")
                except OSError:
                    # Fallback to creating a directory reference file if symlink fails
                    with open(f"{target}_reference.txt", 'w') as f:
                        f.write(f"Dataset location: {os.path.abspath(source)}")
                    print(f"Created reference file for: {target}")

    # Create empty files
    files_to_create = [
        'notebooks/01_data_preprocessing.ipynb',
        'notebooks/02_model_training.ipynb',
        'notebooks/03_evaluation.ipynb',
        'src/dataset_loader.py',
        'src/model.py',
        'src/train.py',
        'src/evaluate.py',
        'src/utils.py',
        'README.md'
    ]
    
    for file_path in files_to_create:
        with open(file_path, 'w') as f:
            pass
        print(f"Created file: {file_path}")

if __name__ == "__main__":
    create_directory_structure() 