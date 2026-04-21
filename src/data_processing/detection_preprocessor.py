"""
Detection Dataset Preprocessor with Albumentations Letterbox
Preprocesses detection dataset using letterbox resize to preserve aspect ratio.
Saves processed images as individual files (not numpy arrays) for memory efficiency.

Dataset Structure:
data/raw/detection_dataset/
├── train_img/          # Training images (*.jpg)
├── train_label/        # Training labels (*.txt, YOLO format)
├── val_img/           # Validation images (*.jpg)
└── val_label/         # Validation labels (*.txt, YOLO format)

Output:
- Processed images saved to: data/processed/detection/{train,valid,test}/
- Annotations saved to: data/processed/detection/annotations/{train,valid,test}/
- Split metadata saved to: data/splitting/detection_split/
"""

import os
import sys
from pathlib import Path
import numpy as np
import json
import yaml
from tqdm import tqdm
import cv2
import gc
import warnings
import contextlib
import io
from typing import Tuple, List, Dict


class DetectionPreprocessor:
    """
    Preprocesses dog face detection dataset using Albumentations letterbox.
    Preserves aspect ratio with padding to avoid image distortion.
    Saves processed images as individual files for memory efficiency.
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
        self.splitting_dir = Path("data/splitting/detection_split")
        self.image_size = self.config['datasets']['detection']['image_size']
        
        # Create directories
        for split in ['train', 'valid', 'test']:
            (self.processed_dir / split).mkdir(parents=True, exist_ok=True)
            (self.processed_dir / 'annotations' / split).mkdir(parents=True, exist_ok=True)
        self.splitting_dir.mkdir(parents=True, exist_ok=True)
    
    def is_processed(self) -> bool:
        """Check if data has already been preprocessed."""
        # Check if processed directories have images
        train_dir = self.processed_dir / 'train'
        valid_dir = self.processed_dir / 'valid'
        test_dir = self.processed_dir / 'test'
        
        if not train_dir.exists() or not valid_dir.exists() or not test_dir.exists():
            return False
        
        # Check if there are any images
        train_images = list(train_dir.glob('*.jpg'))
        valid_images = list(valid_dir.glob('*.jpg'))
        test_images = list(test_dir.glob('*.jpg'))
        
        return len(train_images) > 0 and len(valid_images) > 0 and len(test_images) > 0
    
    def process(self):
        """Main preprocessing pipeline with letterbox resize."""
        print("=" * 80)
        print("DETECTION DATASET PREPROCESSING (WITH LETTERBOX)")
        print("=" * 80)
        
        # Step 1: Load all data using streaming approach
        print("\n[1/5] Loading all data (streaming mode)...")
        all_images, all_annotations = self._load_all_data_streaming()
        print(f"  Loaded {len(all_images)} total images")
        gc.collect()
        
        # Step 2: Split dataset
        print("\n[2/5] Splitting dataset (70/20/10)...")
        splits = self._split_dataset(all_images, all_annotations)
        
        # Clear original data to free memory
        del all_images, all_annotations
        gc.collect()
        
        # Step 3: Save split metadata
        print("\n[3/5] Saving split metadata...")
        self._save_splits(splits)
        
        # Step 4: Process and save each split as individual image files
        processed_info = {}
        
        # Map split names to match the keys from _split_dataset
        split_mapping = {
            'train': 'train',
            'valid': 'val',  # Map 'valid' to 'val' to match _split_dataset output
            'test': 'test'
        }
        
        for idx, split_name in enumerate(['train', 'valid', 'test']):
            step_num = idx + 4
            print(f"\n[{step_num}/5] Processing {split_name} split...")
            
            # Get the correct key name from splits dictionary
            split_key = split_mapping[split_name]
            image_paths = splits[f'{split_key}_images']
            annotations = splits[f'{split_key}_annotations']
            
            # Process and save images
            saved_count = self._preprocess_and_save_split(
                image_paths, 
                annotations, 
                split_name  # Use 'valid' for directory naming
            )
            
            processed_info[split_name] = {
                'count': saved_count,
                'image_dir': str(self.processed_dir / split_name),
                'annotation_dir': str(self.processed_dir / 'annotations' / split_name)
            }
            
            print(f"    Saved {saved_count} images to {self.processed_dir / split_name}")
            
            # Clear memory after each split
            del splits[f'{split_key}_images'], splits[f'{split_key}_annotations']
            gc.collect()
        
        # Save overall metadata
        print("\n[5/5] Saving processing metadata...")
        self._save_processing_metadata(processed_info)
        
        print("\n" + "=" * 80)
        print("PREPROCESSING COMPLETE")
        print("=" * 80)
        print(f"Processed images: {self.processed_dir}")
        print(f"Split metadata: {self.splitting_dir}")
        total_samples = sum(info['count'] for info in processed_info.values())
        print(f"Total samples: {total_samples}")
        for split_name, info in processed_info.items():
            print(f"  {split_name.capitalize()}: {info['count']} images")

    def _load_all_data_streaming(self) -> Tuple[List[str], List[List]]:
        """
        Load all image paths and annotations using streaming approach.
        Processes train and val sequentially to minimize memory usage.
        
        Returns:
            Tuple of (image_paths, annotations)
        """
        all_images = []
        all_annotations = []
        
        # Process train split
        print("  Loading train data...")
        train_images, train_annotations = self._load_split_data_generator('train')
        all_images.extend(train_images)
        all_annotations.extend(train_annotations)
        print(f"    Train: {len(train_images)} images")
        
        # Clear train data from memory
        del train_images, train_annotations
        gc.collect()
        
        # Process val split
        print("  Loading val data...")
        val_images, val_annotations = self._load_split_data_generator('val')
        all_images.extend(val_images)
        all_annotations.extend(val_annotations)
        print(f"    Val: {len(val_images)} images")
        
        # Clear val data from memory
        del val_images, val_annotations
        gc.collect()
        
        return all_images, all_annotations
    
    def _load_split_data_generator(self, split: str) -> Tuple[List[str], List[List]]:
        """
        Load image paths and annotations for a specific split (train or val).
        
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
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        bboxes.append([class_id, x_center, y_center, width, height])
            
            images.append(str(img_file))
            annotations.append(bboxes)
        
        return images, annotations
    
    def _split_dataset(self, images: List[str], annotations: List[List]) -> Dict[str, List]:
        """
        Split dataset respecting original train/val structure.
        
        Strategy (Option A):
        - Original val_img (230 images) → valid set (kept as-is)
        - Original train_img (5924 images) → split into train (90%) and test (10%)
          - train: ~5332 images
          - test: ~592 images
        
        This preserves the original professional train/val split while creating
        a proper test set from the training data.
        
        Args:
            images: List of all image paths (train first, then val)
            annotations: List of all annotations
            
        Returns:
            Dictionary with split datasets
        """
        from sklearn.model_selection import train_test_split
        
        # Separate train and val based on path
        train_images = []
        train_annotations = []
        val_images = []
        val_annotations = []
        
        for img_path, ann in zip(images, annotations):
            if '/train_img/' in img_path or '\\train_img\\' in img_path:
                train_images.append(img_path)
                train_annotations.append(ann)
            elif '/val_img/' in img_path or '\\val_img\\' in img_path:
                val_images.append(img_path)
                val_annotations.append(ann)
        
        print(f"  Original train: {len(train_images)} images")
        print(f"  Original val: {len(val_images)} images")
        
        # Use original val as validation set (no splitting)
        X_val = val_images
        y_val = val_annotations
        
        # Split original train into train (90%) and test (10%)
        test_ratio = 0.10  # 10% for test
        X_train, X_test, y_train, y_test = train_test_split(
            train_images, 
            train_annotations, 
            test_size=test_ratio, 
            random_state=42
        )
        
        print(f"  After splitting:")
        print(f"    Train: {len(X_train)} images (from original train)")
        print(f"    Valid: {len(X_val)} images (original val, kept as-is)")
        print(f"    Test: {len(X_test)} images (from original train)")
        
        return {
            'train_images': X_train,
            'train_annotations': y_train,
            'val_images': X_val,
            'val_annotations': y_val,
            'test_images': X_test,
            'test_annotations': y_test
        }
    
    def _save_splits(self, splits: Dict[str, List]):
        """
        Save split metadata as JSON files to data/splitting/detection_split/.
        
        Note: For detection dataset, images are preprocessed and saved with new names.
        The split files record which original images belong to each split for reference.
        
        Args:
            splits: Dictionary containing split data
        """
        # Save each split as JSON with original image paths (for reference only)
        for split_name in ['train', 'val', 'test']:
            # Determine source of this split
            if split_name == 'val':
                source_info = "Original val_img folder (kept as-is)"
            elif split_name == 'train':
                source_info = "From original train_img folder (90% split)"
            else:  # test
                source_info = "From original train_img folder (10% split)"
            
            split_data = {
                'original_image_paths': splits[f'{split_name}_images'],
                'num_samples': len(splits[f'{split_name}_images']),
                'source': source_info,
                'note': f'Original raw image paths. Processed images are in data/processed/detection/{split_name if split_name != "val" else "valid"}/'
            }
            output_file = self.splitting_dir / f"{split_name}_split.json"
            with open(output_file, 'w') as f:
                json.dump(split_data, f, indent=2)
            print(f"  Saved {split_name} split: {len(split_data['original_image_paths'])} images ({source_info})")
        
        # Save overall metadata
        metadata = {
            'dataset_type': 'detection',
            'total_samples': len(splits['train_images']) + len(splits['val_images']) + len(splits['test_images']),
            'split_strategy': 'Option A: Preserve original val, split train into train/test',
            'splits': {
                'train': {
                    'count': len(splits['train_images']),
                    'source': 'train_img (90%)'
                },
                'val': {
                    'count': len(splits['val_images']),
                    'source': 'val_img (100%, kept as-is)'
                },
                'test': {
                    'count': len(splits['test_images']),
                    'source': 'train_img (10%)'
                }
            },
            'format': 'YOLO',
            'preprocessing': 'letterbox_resize',
            'target_size': self.image_size,
            'note': 'Images are preprocessed with letterbox resize and saved as individual files. Use dataset.yaml for training.'
        }
        
        metadata_file = self.splitting_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved metadata")
    
    def _letterbox_resize(self, img: np.ndarray, bboxes: List, target_size: int = 640) -> Tuple[np.ndarray, List]:
        """
        Resize image with letterbox (preserve aspect ratio + padding).
        
        Args:
            img: Original image (H, W, C)
            bboxes: YOLO format bboxes [[class, x_c, y_c, w, h], ...]
            target_size: Target size (default 640)
        
        Returns:
            Resized image and adjusted bboxes
        """
        h, w = img.shape[:2]
        
        # Calculate scale ratio (keep aspect ratio)
        scale = min(target_size / w, target_size / h)
        
        # Resize
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h))
        
        # Calculate padding
        pad_w = (target_size - new_w) / 2
        pad_h = (target_size - new_h) / 2
        
        # Add padding (gray color for YOLO: 114, 114, 114)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img_padded = cv2.copyMakeBorder(
            img_resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=[114, 114, 114]
        )
        
        # Adjust bounding boxes
        adjusted_bboxes = []
        for bbox in bboxes:
            class_id, x_c, y_c, bw, bh = bbox
            
            # Scale coordinates
            x_c_new = x_c * scale
            y_c_new = y_c * scale
            bw_new = bw * scale
            bh_new = bh * scale
            
            # Add padding offset and normalize to padded image size
            x_c_final = (x_c_new + pad_w) / target_size
            y_c_final = (y_c_new + pad_h) / target_size
            bw_final = bw_new / target_size
            bh_final = bh_new / target_size
            
            adjusted_bboxes.append([class_id, x_c_final, y_c_final, bw_final, bh_final])
        
        return img_padded, adjusted_bboxes
    
    def _preprocess_and_save_split(self, images: List[str], annotations: List[List], split_name: str) -> int:
        """
        Preprocess images using letterbox resize and save as individual files.
        Processes in batches to avoid memory issues.
        
        Args:
            images: List of image paths
            annotations: List of bounding box lists
            split_name: 'train', 'valid', or 'test'
            
        Returns:
            Number of successfully processed images
        """
        output_img_dir = self.processed_dir / split_name
        output_ann_dir = self.processed_dir / 'annotations' / split_name
        
        n_images = len(images)
        batch_size = 100  # Process 100 images at a time
        
        n_batches = (n_images + batch_size - 1) // batch_size
        print(f"  Processing {n_images} images in {n_batches} batches...")
        
        saved_count = 0
        failed_images = []  # Track failed images
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_images)
            
            for i in tqdm(range(start_idx, end_idx), desc=f"  Batch {batch_idx+1}/{n_batches}", leave=False):
                img_path = images[i]
                bboxes = annotations[i]
                
                try:
                    # Load image with error handling for corrupt files
                    # Suppress OpenCV warnings by redirecting stderr temporarily
                    f = io.StringIO()
                    with contextlib.redirect_stderr(f):
                        img = cv2.imread(str(img_path))
                    
                    if img is None:
                        failed_images.append(img_path)
                        continue
                    
                    # Apply letterbox resize
                    img_processed, bboxes_processed = self._letterbox_resize(
                        img, bboxes, self.image_size
                    )
                    
                    # Generate output filename
                    img_filename = f"img_{saved_count:05d}.jpg"
                    ann_filename = f"img_{saved_count:05d}.txt"
                    
                    # Save processed image
                    output_img_path = output_img_dir / img_filename
                    cv2.imwrite(str(output_img_path), img_processed)
                    
                    # Save annotations in YOLO format
                    output_ann_path = output_ann_dir / ann_filename
                    with open(output_ann_path, 'w') as f:
                        for bbox in bboxes_processed:
                            class_id, x_c, y_c, w, h = bbox
                            f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                    
                    saved_count += 1
                    
                    # Clear memory for this image
                    del img, img_processed, bboxes_processed
                    gc.collect()
                    
                except Exception as e:
                    # Log the specific error for debugging
                    print(f"\n  ⚠ Error processing image {img_path}: {str(e)}")
                    failed_images.append(img_path)
                    # Ensure cleanup before continuing to next image
                    try:
                        del img, img_processed, bboxes_processed
                    except:
                        pass
                    gc.collect()
                    continue
            
            # Clear batch memory
            gc.collect()
        
        # Report failed images
        if failed_images:
            print(f"\n  ⚠ Warning: {len(failed_images)} images failed to process:")
            for img_path in failed_images[:10]:  # Show first 10
                print(f"    - {img_path}")
            if len(failed_images) > 10:
                print(f"    ... and {len(failed_images) - 10} more")
            
            # Save failed images list to file
            failed_log = self.processed_dir / f"failed_{split_name}_images.txt"
            with open(failed_log, 'w') as f:
                for img_path in failed_images:
                    f.write(f"{img_path}\n")
            print(f"  Full list saved to: {failed_log}")
        
        return saved_count
    
    def _save_processing_metadata(self, processed_info: Dict):
        """
        Save metadata about the preprocessing results.
        
        Args:
            processed_info: Dictionary with processing statistics
        """
        metadata = {
            'dataset_type': 'detection',
            'image_size': self.image_size,
            'preprocessing_method': 'letterbox_resize',
            'padding_color': [114, 114, 114],
            'output_format': 'individual_jpeg_files',
            'splits': {
                split_name: {
                    'count': info['count'],
                    'image_directory': info['image_dir'],
                    'annotation_directory': info['annotation_dir']
                }
                for split_name, info in processed_info.items()
            },
            'total_samples': sum(info['count'] for info in processed_info.values()),
            'note': 'Images are saved as individual JPEG files with corresponding YOLO annotations'
        }
        
        metadata_file = self.processed_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved metadata to {metadata_file}")
        
        # Generate YOLOv8 dataset configuration file
        self._generate_yolo_dataset_config(processed_info)
    
    def _generate_yolo_dataset_config(self, processed_info: Dict):
        """
        Generate YOLOv8 dataset configuration YAML file.
        This file is required for YOLOv8 training.
        Uses relative paths for portability across different machines.
        
        Args:
            processed_info: Dictionary with processing statistics
        """
        # Use relative path ('.' means current directory where dataset.yaml is located)
        # This makes the config portable across different machines
        yolo_config = {
            'path': '.',           # Current directory (where dataset.yaml is located)
            'train': 'train',      # Relative path to train images
            'val': 'valid',        # Relative path to validation images
            'test': 'test',        # Relative path to test images
            
            # Number of classes
            'nc': 1,
            
            # Class names (dog face detection has only one class)
            'names': ['dog_face']
        }
        
        # Save as YAML file
        config_file = self.processed_dir / "dataset.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(yolo_config, f, default_flow_style=False, sort_keys=False)
        
        print(f"  ✓ Generated YOLOv8 dataset config: {config_file}")
        print(f"    Using relative paths for portability")
        print(f"    Classes: {yolo_config['nc']} ({yolo_config['names']})")

def main():
    """Main function to run detection dataset preprocessing."""
    preprocessor = DetectionPreprocessor()
    
    if preprocessor.is_processed():
        print("✓ Detection dataset already preprocessed.")
        print(f"  Output directory: {preprocessor.processed_dir}")
        print("  Auto-overwriting existing files...")
        # Automatically proceed without asking
    
    try:
        preprocessor.process()
        print("\n✓ Preprocessing complete! You can now run experiments.")
    except Exception as e:
        print(f"\n✗ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
