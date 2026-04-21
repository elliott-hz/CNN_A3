"""
Detection Dataset Preprocessor
Converts raw detection dataset to unified format (X_train, X_valid, X_test, y_*)

Dataset Structure:
data/raw/detection_dataset/
├── train_img/          # Training images (*.jpg)
├── train_label/        # Training labels (*.txt, YOLO format)
├── val_img/           # Validation images (*.jpg)
└── val_label/         # Validation labels (*.txt, YOLO format)

Label Format (YOLO):
class_id x_center y_center width height  (all normalized 0-1)
"""

import os
import sys
from pathlib import Path
import numpy as np
import json
import yaml
from tqdm import tqdm
import cv2
from typing import Tuple, List, Dict


class DetectionPreprocessor:
    """
    Preprocesses dog face detection dataset.
    Converts YOLO format annotations to unified numpy arrays.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_data_dir = Path(self.config['paths']['raw_data']) / "detection_dataset"
        self.processed_dir = Path(self.config['paths']['processed_data']) / "detection"
        self.image_size = self.config['datasets']['detection']['image_size']
        
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
        print("DETECTION DATASET PREPROCESSING")
        print("=" * 80)
        
        # Load training data
        print("\n[1/5] Loading training data...")
        train_images, train_annotations = self._load_split_data('train')
        print(f"  Loaded {len(train_images)} training images")
        
        # Load validation data
        print("\n[2/5] Loading validation data...")
        val_images, val_annotations = self._load_split_data('val')
        print(f"  Loaded {len(val_images)} validation images")
        
        # Combine and split into train/val/test
        print("\n[3/5] Combining and splitting dataset (70/20/10)...")
        all_images = train_images + val_images
        all_annotations = train_annotations + val_annotations
        splits = self._split_dataset(all_images, all_annotations)
        
        # Preprocess images and annotations
        print("\n[4/5] Preprocessing images and annotations...")
        processed_splits = {}
        for split_name in ['train', 'valid', 'test']:
            print(f"  Processing {split_name} split...")
            X, y = self._preprocess_data(
                splits[f'{split_name}_images'], 
                splits[f'{split_name}_annotations']
            )
            processed_splits[f'X_{split_name}'] = X
            processed_splits[f'y_{split_name}'] = y
            print(f"    {split_name}: X={X.shape}, y={len(y)} samples")
        
        # Save processed data
        print("\n[5/5] Saving processed data...")
        self._save_splits(processed_splits)
        
        print("\n" + "=" * 80)
        print("PREPROCESSING COMPLETE")
        print("=" * 80)
        print(f"Output directory: {self.processed_dir}")
        total_samples = sum(processed_splits[f'X_{split}'].shape[0] for split in ['train', 'valid', 'test'])
        print(f"Total samples: {total_samples}")
        print(f"  Train: {processed_splits['X_train'].shape[0]}")
        print(f"  Valid: {processed_splits['X_valid'].shape[0]}")
        print(f"  Test: {processed_splits['X_test'].shape[0]}")
    
    def _load_split_data(self, split: str) -> Tuple[List[str], List[List]]:
        """
        Load images and annotations for a specific split (train or val).
        
        Args:
            split: 'train' or 'val'
            
        Returns:
            Tuple of (image_paths, annotations)
            annotations is a list where each element is a list of bboxes for that image
        """
        img_dir = self.raw_data_dir / f"{split}_img"
        label_dir = self.raw_data_dir / f"{split}_label"
        
        if not img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {img_dir}")
        if not label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {label_dir}")
        
        images = []
        annotations = []
        
        # Get all label files
        label_files = sorted(label_dir.glob("*.txt"))
        
        for label_file in tqdm(label_files, desc=f"Loading {split} data"):
            # Find corresponding image file
            img_file = img_dir / f"{label_file.stem}.jpg"
            if not img_file.exists():
                # Try other extensions
                img_file = img_dir / f"{label_file.stem}.jpeg"
                if not img_file.exists():
                    img_file = img_dir / f"{label_file.stem}.png"
                    if not img_file.exists():
                        continue
            
            # Read YOLO annotations
            bboxes = []
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 5:
                        # YOLO format: class x_center y_center width height (normalized)
                        class_id, x_c, y_c, w, h = map(float, parts[:5])
                        bboxes.append([int(class_id), x_c, y_c, w, h])
            
            images.append(str(img_file))
            annotations.append(bboxes)
        
        return images, annotations
    
    def _preprocess_data(self, images: List[str], annotations: List[List]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess images and annotations.
        
        Args:
            images: List of image paths
            annotations: List of bounding box lists (each bbox: [class, x_c, y_c, w, h])
            
        Returns:
            Tuple of (images_array, annotations_array)
        """
        X_list = []
        y_list = []
        
        for img_path, bboxes in tqdm(zip(images, annotations), total=len(images), desc="Processing"):
            try:
                # Load and resize image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                original_h, original_w = img.shape[:2]
                
                # Resize to target size
                img_resized = cv2.resize(img, (self.image_size, self.image_size))
                
                # Normalize to [0, 1]
                img_normalized = img_resized.astype(np.float32) / 255.0
                
                # Convert BGR to RGB
                img_rgb = cv2.cvtColor(img_normalized, cv2.COLOR_BGR2RGB)
                
                # Process annotations - scale bounding boxes to match resized image
                scaled_bboxes = []
                for bbox in bboxes:
                    class_id, x_c, y_c, w, h = bbox
                    # Scale normalized coordinates to image size
                    x_c_scaled = x_c * self.image_size
                    y_c_scaled = y_c * self.image_size
                    w_scaled = w * self.image_size
                    h_scaled = h * self.image_size
                    scaled_bboxes.append([class_id, x_c_scaled, y_c_scaled, w_scaled, h_scaled])
                
                X_list.append(img_rgb)
                y_list.append(scaled_bboxes if scaled_bboxes else [])
                
            except Exception as e:
                print(f"Warning: Error processing {img_path}: {e}")
                continue
        
        # Convert to numpy arrays
        X = np.array(X_list)
        
        # For y, save as object array since each image can have different number of boxes
        y = np.empty(len(y_list), dtype=object)
        for i, bboxes in enumerate(y_list):
            if bboxes:
                y[i] = np.array(bboxes, dtype=np.float32)
            else:
                y[i] = np.array([]).reshape(0, 5).astype(np.float32)
        
        return X, y
    
    def _split_dataset(self, images: List[str], annotations: List[List]) -> Dict[str, List]:
        """
        Split combined dataset into train/valid/test sets.
        
        Args:
            images: List of all image paths
            annotations: List of all annotations
            
        Returns:
            Dictionary with split datasets
        """
        from sklearn.model_selection import train_test_split
        
        n_samples = len(images)
        train_ratio = self.config['datasets']['detection']['train_ratio']
        val_ratio = self.config['datasets']['detection']['val_ratio']
        test_ratio = 1 - train_ratio - val_ratio
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, annotations, test_size=test_ratio, random_state=42
        )
        
        # Second split: separate train and validation
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_temp, y_temp, test_size=val_ratio_adjusted, random_state=42
        )
        
        return {
            'train_images': X_train,
            'valid_images': X_valid,
            'test_images': X_test,
            'train_annotations': y_train,
            'valid_annotations': y_valid,
            'test_annotations': y_test
        }
    
    def _save_splits(self, splits: Dict[str, np.ndarray]):
        """Save split datasets to disk."""
        for key, data in splits.items():
            filepath = self.processed_dir / f"{key}.npy"
            np.save(filepath, data, allow_pickle=True)
        
        # Save metadata
        metadata = {
            'total_samples': sum(splits[f'X_{split}'].shape[0] for split in ['train', 'valid', 'test']),
            'train_samples': int(splits['X_train'].shape[0]),
            'valid_samples': int(splits['X_valid'].shape[0]),
            'test_samples': int(splits['X_test'].shape[0]),
            'image_size': self.image_size,
            'split_ratios': {
                'train': self.config['datasets']['detection']['train_ratio'],
                'valid': self.config['datasets']['detection']['val_ratio'],
                'test': 1 - self.config['datasets']['detection']['train_ratio'] - self.config['datasets']['detection']['val_ratio']
            },
            'annotation_format': 'YOLO (class, x_center, y_center, width, height)',
            'coordinate_system': 'Absolute pixel coordinates after resizing'
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
        X = np.load(self.processed_dir / f"X_{split_name}.npy", allow_pickle=True)
        y = np.load(self.processed_dir / f"y_{split_name}.npy", allow_pickle=True)
        return X, y


if __name__ == "__main__":
    preprocessor = DetectionPreprocessor()
    if not preprocessor.is_processed():
        preprocessor.process()
    else:
        print("Data already preprocessed. Skipping.")
