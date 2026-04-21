"""
Emotion Dataset Preprocessor
Converts raw emotion dataset to unified format (X_train, X_valid, X_test, y_*)

Dataset Structure:
data/raw/emotion_dataset/
├── alert/      # Alert emotion images
├── angry/      # Angry emotion images
├── frown/      # Frown emotion images
├── happy/      # Happy emotion images
└── relax/      # Relaxed emotion images

Each folder contains images (*.jpg) representing that emotion class.
"""

import os
from pathlib import Path
import numpy as np
import json
import yaml
from tqdm import tqdm
import cv2
from typing import Tuple, List, Dict


class EmotionPreprocessor:
    """
    Preprocesses dog emotion classification dataset.
    Converts folder-based structure to unified numpy arrays.
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
        self.processed_dir = Path(self.config['paths']['processed_data']) / "emotion"
        self.image_size = self.config['datasets']['emotion']['image_size']
        self.classes = self.config['datasets']['emotion']['classes']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Create processed directory
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def is_processed(self) -> bool:
        """Check if data has already been preprocessed."""
        required_files = [
            self.processed_dir / "X_train.npy",
            self.processed_dir / "X_valid.npy",
            self.processed_dir / "X_test.npy",
            self.processed_dir / "y_train.npy",
            self.processed_dir / "y_valid.npy",
            self.processed_dir / "y_test.npy",
            self.processed_dir / "metadata.json"
        ]
        return all(f.exists() for f in required_files)
    
    def process(self):
        """Main preprocessing pipeline."""
        print("=" * 80)
        print("EMOTION DATASET PREPROCESSING")
        print("=" * 80)
        
        # Load and organize data by class
        print("\n[1/4] Loading and organizing dataset...")
        images, labels = self._load_raw_data()
        print(f"  Loaded {len(images)} images across {len(self.classes)} classes")
        
        # Display class distribution
        print("\n  Class distribution:")
        from collections import Counter
        label_counts = Counter(labels)
        for cls in self.classes:
            count = label_counts.get(cls, 0)
            print(f"    {cls}: {count}")
        
        # Preprocess images
        print("\n[2/4] Preprocessing images...")
        X, y = self._preprocess_data(images, labels)
        print(f"  Preprocessed shape: X={X.shape}, y={y.shape}")
        
        # Split dataset
        print("\n[3/4] Splitting dataset (70/20/10)...")
        splits = self._split_dataset(X, y)
        
        # Save processed data
        print("\n[4/4] Saving processed data...")
        self._save_splits(splits)
        
        print("\n" + "=" * 80)
        print("PREPROCESSING COMPLETE")
        print("=" * 80)
        print(f"Output directory: {self.processed_dir}")
        print(f"Total samples: {len(X)}")
        print(f"  Train: {splits['X_train'].shape[0]}")
        print(f"  Valid: {splits['X_valid'].shape[0]}")
        print(f"  Test: {splits['X_test'].shape[0]}")
    
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
            
            for img_file in image_files:
                images.append(str(img_file))
                labels.append(class_name)
        
        return images, labels
    
    def _preprocess_data(self, images: List[str], labels: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess images and encode labels.
        
        Args:
            images: List of image paths
            labels: List of class labels
            
        Returns:
            Tuple of (images_array, labels_array)
        """
        X_list = []
        y_list = []
        
        failed_count = 0
        for img_path, label in tqdm(zip(images, labels), total=len(images), desc="Processing"):
            try:
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    failed_count += 1
                    continue
                
                # Resize to target size
                img_resized = cv2.resize(img, (self.image_size, self.image_size))
                
                # Normalize to [0, 1]
                img_normalized = img_resized.astype(np.float32) / 255.0
                
                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(img_normalized, cv2.COLOR_BGR2RGB)
                
                X_list.append(img_rgb)
                y_list.append(self.class_to_idx[label])
                
            except Exception as e:
                print(f"Warning: Error processing {img_path}: {e}")
                failed_count += 1
                continue
        
        if failed_count > 0:
            print(f"  Warning: Failed to process {failed_count} images")
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        return X, y
    
    def _split_dataset(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Split dataset into train/valid/test sets with stratification.
        
        Args:
            X: Images array
            y: Labels array
            
        Returns:
            Dictionary with split datasets
        """
        from sklearn.model_selection import train_test_split
        
        train_ratio = self.config['datasets']['emotion']['train_ratio']
        val_ratio = self.config['datasets']['emotion']['val_ratio']
        test_ratio = 1 - train_ratio - val_ratio
        
        # First split: separate test set (stratified)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=test_ratio, 
            random_state=42,
            stratify=y
        )
        
        # Second split: separate train and validation (stratified)
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_temp, y_temp, 
            test_size=val_ratio_adjusted, 
            random_state=42,
            stratify=y_temp
        )
        
        return {
            'X_train': X_train,
            'X_valid': X_valid,
            'X_test': X_test,
            'y_train': y_train,
            'y_valid': y_valid,
            'y_test': y_test
        }
    
    def _save_splits(self, splits: Dict[str, np.ndarray]):
        """Save split datasets to disk."""
        for key, data in splits.items():
            filepath = self.processed_dir / f"{key}.npy"
            np.save(filepath, data)
        
        # Save metadata
        metadata = {
            'total_samples': int(len(splits['X_train']) + len(splits['X_valid']) + len(splits['X_test'])),
            'train_samples': int(len(splits['X_train'])),
            'valid_samples': int(len(splits['X_valid'])),
            'test_samples': int(len(splits['X_test'])),
            'image_size': self.image_size,
            'num_classes': len(self.classes),
            'classes': self.classes,
            'class_distribution': {
                cls: int(np.sum(splits['y_train'] == idx) + 
                        np.sum(splits['y_valid'] == idx) + 
                        np.sum(splits['y_test'] == idx))
                for cls, idx in self.class_to_idx.items()
            },
            'split_ratios': {
                'train': self.config['datasets']['emotion']['train_ratio'],
                'valid': self.config['datasets']['emotion']['val_ratio'],
                'test': 1 - self.config['datasets']['emotion']['train_ratio'] - self.config['datasets']['emotion']['val_ratio']
            }
        }
        
        with open(self.processed_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_split(self, split_name: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a specific data split.
        
        Args:
            split_name: 'train', 'valid', or 'test'
            
        Returns:
            Tuple of (X, y) arrays
        """
        X = np.load(self.processed_dir / f"X_{split_name}.npy")
        y = np.load(self.processed_dir / f"y_{split_name}.npy")
        return X, y


if __name__ == "__main__":
    preprocessor = EmotionPreprocessor()
    if not preprocessor.is_processed():
        preprocessor.process()
    else:
        print("Data already preprocessed. Skipping.")
