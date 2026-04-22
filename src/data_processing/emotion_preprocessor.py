"""
Emotion Dataset Preprocessor
Parses raw emotion dataset format and splits into train/val/test subsets.
NO preprocessing (resize, normalize, etc.) - just format parsing and splitting.

Dataset Structure:
data/raw/emotion_dataset/
├── alert/      # Alert emotion images
├── angry/      # Angry emotion images
├── frown/      # Frown emotion images
├── happy/      # Happy emotion images
└── relax/      # Relax emotion images

Each folder contains images (*.jpg) representing that emotion class.

Output:
Split metadata saved to: data/splitting/emotion_split/
Images are loaded on-the-fly during training.
"""

import os
from pathlib import Path
import numpy as np
import json
import yaml
from tqdm import tqdm
from typing import Tuple, List, Dict


class EmotionPreprocessor:
    """
    Parses dog emotion classification dataset and splits into train/val/test.
    Only organizes folder structure and creates split indices.
    No image preprocessing - images loaded during training.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_data_dir = Path(self.config['paths']['raw_data']) / "emotion_dataset"
        self.splitting_dir = Path("data/splitting/emotion_split")
        self.classes = self.config['datasets']['emotion']['classes']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Create directory
        self.splitting_dir.mkdir(parents=True, exist_ok=True)
    
    def is_processed(self) -> bool:
        """Check if data has already been parsed and split."""
        required_files = [
            self.splitting_dir / "train_split.json",
            self.splitting_dir / "val_split.json",
            self.splitting_dir / "test_split.json",
            self.splitting_dir / "metadata.json"
        ]
        return all(f.exists() for f in required_files)
    
    def process(self):
        """Main parsing and splitting pipeline."""
        print("=" * 80)
        print("EMOTION DATASET PARSING AND SPLITTING")
        print("=" * 80)
        
        # Load and organize data by class
        print("\n[1/3] Loading and organizing dataset...")
        images, labels = self._load_raw_data()
        print(f"  Loaded {len(images)} images across {len(self.classes)} classes")
        
        # Display class distribution
        print("\n  Class distribution:")
        from collections import Counter
        label_counts = Counter(labels)
        for cls in self.classes:
            count = label_counts.get(cls, 0)
            print(f"    {cls}: {count}")
        
        # Split dataset
        print("\n[2/3] Splitting dataset (70/20/10)...")
        splits = self._split_dataset(images, labels)
        
        # Save split metadata
        print("\n[3/3] Saving split metadata...")
        self._save_splits(splits)
        
        print("\n" + "=" * 80)
        print("PARSING AND SPLITTING COMPLETE")
        print("=" * 80)
        print(f"Split metadata: {self.splitting_dir}")
        total_samples = len(splits['train_images']) + len(splits['val_images']) + len(splits['test_images'])
        print(f"Total samples: {total_samples}")
        print(f"  Train: {len(splits['train_images'])} images")
        print(f"  Valid: {len(splits['val_images'])} images")
        print(f"  Test: {len(splits['test_images'])} images")
        print("\nNote: Images are NOT preprocessed. They will be loaded during training.")
    
    def _load_raw_data(self) -> Tuple[List[str], List[str]]:
        """
        Load raw emotion dataset from folder structure.
        Expected structure:
        emotion_dataset/
            alert/
                img1.jpg
                img2.jpg
            angry/
                img1.jpg
                ...
        
        Returns:
            Tuple of (image_paths, labels)
        """
        images = []
        labels = []
        
        for class_name in self.classes:
            class_dir = self.raw_data_dir / class_name
            
            if not class_dir.exists():
                print(f"Warning: Class directory not found: {class_dir}")
                continue
            
            # Get all image files (support multiple extensions)
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(class_dir.glob(ext))
            
            image_files = sorted(image_files)  # Sort for reproducibility
            
            for img_file in tqdm(image_files, desc=f"Loading {class_name}"):
                images.append(str(img_file))
                labels.append(class_name)
        
        return images, labels
    
    def _split_dataset(self, images: List[str], labels: List[str]) -> Dict[str, List]:
        """
        Split dataset into train/valid/test sets with stratification.
        
        Args:
            images: List of image paths
            labels: List of class labels
            
        Returns:
            Dictionary with split datasets
        """
        from sklearn.model_selection import train_test_split
        
        n_samples = len(images)
        train_ratio = self.config['datasets']['emotion']['train_ratio']
        val_ratio = self.config['datasets']['emotion']['val_ratio']
        test_ratio = 1 - train_ratio - val_ratio
        
        # First split: separate test set (stratified)
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, labels, test_size=test_ratio, random_state=42, stratify=labels
        )
        
        # Second split: separate train and validation from remaining (stratified)
        val_adjusted = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_adjusted, random_state=42, stratify=y_temp
        )
        
        return {
            'train_images': X_train,
            'train_labels': y_train,
            'val_images': X_val,
            'val_labels': y_val,
            'test_images': X_test,
            'test_labels': y_test
        }
    
    def _save_splits(self, splits: Dict[str, List]):
        """
        Save split metadata as JSON files to data/splitting/emotion_split/.
        
        Args:
            splits: Dictionary containing split data
        """
        # Save each split as JSON
        for split_name in ['train', 'val', 'test']:
            split_data = {
                'images': splits[f'{split_name}_images'],
                'labels': splits[f'{split_name}_labels']
            }
            output_file = self.splitting_dir / f"{split_name}_split.json"
            with open(output_file, 'w') as f:
                json.dump(split_data, f, indent=2)
            print(f"  Saved {split_name} split: {len(split_data['images'])} images")
        
        # Save overall metadata
        from collections import Counter
        metadata = {
            'dataset_type': 'emotion_classification',
            'classes': self.classes,
            'class_to_idx': self.class_to_idx,
            'total_samples': len(splits['train_images']) + len(splits['val_images']) + len(splits['test_images']),
            'splits': {
                'train': len(splits['train_images']),
                'val': len(splits['val_images']),
                'test': len(splits['test_images'])
            },
            'class_distribution': {
                'train': dict(Counter(splits['train_labels'])),
                'val': dict(Counter(splits['val_labels'])),
                'test': dict(Counter(splits['test_labels']))
            },
            'format': 'folder_based',
            'preprocessing': 'none',
            'note': 'Images are stored as paths and loaded during training'
        }
        
        metadata_file = self.splitting_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved metadata")


def main():
    """Main function to run emotion dataset parsing."""
    preprocessor = EmotionPreprocessor()
    
    if preprocessor.is_processed():
        print("✓ Emotion dataset already parsed and split.")
        print(f"  Output directory: {preprocessor.splitting_dir}")
        print("  Auto-overwriting existing files...")
        # Automatically proceed without asking
    
    try:
        preprocessor.process()
        print("\n✓ Parsing complete! You can now run experiments.")
    except Exception as e:
        print(f"\n✗ Error during parsing: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
