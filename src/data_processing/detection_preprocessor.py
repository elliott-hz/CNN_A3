"""
Detection Dataset Preprocessor with Albumentations Letterbox
Preprocesses detection dataset using letterbox resize to preserve aspect ratio.
Saves processed images as individual files (not numpy arrays) for memory efficiency.

Dataset Structure (YOLOv8 Standard Format):
data/processed/detection/
├── images/
│   ├── train/          # Training images (*.jpg)
│   ├── val/            # Validation images (*.jpg)
│   └── test/           # Test images (*.jpg)
├── labels/
│   ├── train/          # Training labels (*.txt, YOLO format)
│   ├── val/            # Validation labels (*.txt, YOLO format)
│   └── test/           # Test labels (*.txt, YOLO format)
├── dataset.yaml        # YOLOv8 configuration file
└── metadata.json       # Processing metadata

Input:
data/raw/detection_dataset/
├── train_img/          # Training images (*.jpg)
├── train_label/        # Training labels (*.txt, YOLO format)
├── val_img/           # Validation images (*.jpg)
└── val_label/         # Validation labels (*.txt, YOLO format)

Output:
- Processed images saved to: data/processed/detection/images/{train,val,test}/
- Annotations saved to: data/processed/detection/labels/{train,val,test}/
- Split metadata saved to: data/splitting/detection_split/
"""

import os
import shutil
from sklearn.model_selection import train_test_split
import yaml


class DetectionPreprocessor:
    """
    A preprocessor for detection datasets that organizes images and labels
    into train/val/test splits for YOLOv8 training.
    """
    
    def __init__(self, raw_data_path, processed_data_path="data/processed/detection", train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
        """
        Initializes the DetectionPreprocessor.
        
        Args:
            raw_data_path: Path to the raw dataset containing images and labels
            processed_data_path: Path to save the processed dataset
            train_ratio: Proportion of data for training (default: 0.7 → 70%)
            val_ratio: Proportion of data for validation (default: 0.2 → 20%)
            test_ratio: Proportion of data for testing (default: 0.1 → 10%)
            
        Note:
            Ratios should sum to 1.0. The actual split uses two-stage process:
            1. First split: extract test set (test_ratio)
            2. Second split: from remaining, extract val set (val_ratio / (1 - test_ratio))
        """
        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        
        # Store original ratios for reference
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Calculate internal split parameters for sklearn's train_test_split
        # Stage 1: Extract test set
        self._test_size = test_ratio
        
        # Stage 2: From remaining (1 - test_ratio), extract val set
        # We want val_ratio of total, so from remaining it's: val_ratio / (1 - test_ratio)
        self._val_size = val_ratio / (1 - test_ratio) if (1 - test_ratio) > 0 else 0.5
        
        # Create directory structure (YOLOv8 standard: images/split, labels/split)
        self.directories = {
            'train': {
                'images': os.path.join(self.processed_data_path, 'images', 'train'),
                'labels': os.path.join(self.processed_data_path, 'labels', 'train')
            },
            'val': {
                'images': os.path.join(self.processed_data_path, 'images', 'val'),
                'labels': os.path.join(self.processed_data_path, 'labels', 'val')
            },
            'test': {
                'images': os.path.join(self.processed_data_path, 'images', 'test'),
                'labels': os.path.join(self.processed_data_path, 'labels', 'test')
            }
        }
        
        # Create all required directories
        for dirs in self.directories.values():
            os.makedirs(dirs['images'], exist_ok=True)
            os.makedirs(dirs['labels'], exist_ok=True)
    
    def _find_image_label_pairs(self):
        """
        Find all image-label pairs in the raw data directory.
        
        Returns:
            List of tuples (image_path, label_path) for valid pairs
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_dir = os.path.join(self.raw_data_path, 'train_img')  # Training images
        label_dir = os.path.join(self.raw_data_path, 'train_label')  # Training labels
        
        # Collect all training pairs
        pairs = self._collect_pairs(image_dir, label_dir, image_extensions)
        
        # Add validation data if it exists separately
        val_image_dir = os.path.join(self.raw_data_path, 'val_img')
        val_label_dir = os.path.join(self.raw_data_path, 'val_label')
        
        if os.path.exists(val_image_dir) and os.path.exists(val_label_dir):
            val_pairs = self._collect_pairs(val_image_dir, val_label_dir, image_extensions)
            pairs.extend(val_pairs)
        
        print(f"Found {len(pairs)} valid image-label pairs")
        return pairs
    
    def _collect_pairs(self, image_dir, label_dir, image_extensions):
        """
        Helper method to collect image-label pairs from specific directories.
        Validates YOLO labels and filters out images with invalid annotations.
        """
        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            return []

        image_files = []
        for ext in image_extensions:
            image_files.extend([f for f in os.listdir(image_dir) if f.lower().endswith(ext)])
        
        pairs = []
        skipped_invalid_label = 0
        skipped_empty_label = 0
        
        for img_file in image_files:
            base_name = os.path.splitext(img_file)[0]
            label_file = base_name + '.txt'
            img_path = os.path.join(image_dir, img_file)
            label_path = os.path.join(label_dir, label_file)
            
            # Check if label file exists
            if not os.path.exists(label_path):
                continue
            
            # Check if label file is empty
            if os.path.getsize(label_path) == 0:
                skipped_empty_label += 1
                continue
            
            # Validate label content
            if self._validate_yolo_label(label_path, img_path):
                pairs.append((img_path, label_path))
            else:
                skipped_invalid_label += 1
        
        if skipped_invalid_label > 0 or skipped_empty_label > 0:
            print(f"  Skipped {skipped_invalid_label} images with invalid labels, {skipped_empty_label} with empty labels")
        
        return pairs
    
    def _validate_yolo_label(self, label_path, img_path):
        """
        Validate YOLO format label file and filter invalid bounding boxes.
        Rewrites the label file to only contain valid annotations.
        
        Args:
            label_path: Path to the YOLO label file
            img_path: Path to the corresponding image file
            
        Returns:
            True if the label file has at least one valid annotation, False otherwise
            
        Validation criteria:
        - All values must be in [0, 1] range
        - Width and height must be > 0.001 (at least 0.1% of image size)
        - Bounding box must not exceed image boundaries
        """
        try:
            # Get image dimensions
            from PIL import Image
            with Image.open(img_path) as img:
                img_width, img_height = img.size
            
            valid_lines = []
            total_annotations = 0
            filtered_count = 0
            
            with open(label_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    original_line = line
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        print(f"    ⚠️  Invalid format in {os.path.basename(label_path)} line {line_num}: expected 5 values, got {len(parts)}")
                        filtered_count += 1
                        continue
                    
                    try:
                        cls, x_center, y_center, width, height = map(float, parts)
                    except ValueError:
                        print(f"    ⚠️  Invalid number format in {os.path.basename(label_path)} line {line_num}")
                        filtered_count += 1
                        continue
                    
                    total_annotations += 1
                    
                    # Validate class ID
                    if cls < 0 or cls != int(cls):
                        print(f"    ⚠️  Invalid class ID {cls} in {os.path.basename(label_path)} line {line_num}")
                        filtered_count += 1
                        continue
                    
                    # Validate normalized coordinates are in [0, 1]
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1):
                        print(f"    ⚠️  Coordinates out of [0,1] range in {os.path.basename(label_path)} line {line_num}")
                        filtered_count += 1
                        continue
                    
                    # Validate minimum size (at least 0.1% of image dimensions)
                    min_size_threshold = 0.001
                    if width < min_size_threshold or height < min_size_threshold:
                        print(f"    ⚠️  Bbox too small (w={width:.6f}, h={height:.6f}) in {os.path.basename(label_path)} line {line_num}")
                        filtered_count += 1
                        continue
                    
                    # Validate bbox doesn't exceed image boundaries
                    x_min = x_center - width / 2
                    y_min = y_center - height / 2
                    x_max = x_center + width / 2
                    y_max = y_center + height / 2
                    
                    if x_min < 0 or y_min < 0 or x_max > 1 or y_max > 1:
                        print(f"    ⚠️  Bbox exceeds image boundaries in {os.path.basename(label_path)} line {line_num}")
                        filtered_count += 1
                        continue
                    
                    # Annotation is valid, keep it
                    valid_lines.append(original_line)
            
            if len(valid_lines) == 0:
                if total_annotations > 0:
                    print(f"  ⚠️  All {total_annotations} annotations invalid in {os.path.basename(label_path)}, skipping image")
                else:
                    print(f"  ⚠️  No valid annotations in {os.path.basename(label_path)}, skipping image")
                return False
            
            # Rewrite label file with only valid annotations
            if filtered_count > 0:
                with open(label_path, 'w') as f:
                    f.writelines(valid_lines)
                print(f"  ✓ Filtered {filtered_count} invalid annotations from {os.path.basename(label_path)} ({len(valid_lines)}/{total_annotations} remain)")
            
            return True
            
        except Exception as e:
            print(f"  ⚠️  Error validating {os.path.basename(label_path)}: {str(e)}")
            return False
    
    def _split_data(self, pairs):
        """
        Split the data into train/val/test sets.
        
        Args:
            pairs: List of tuples (image_path, label_path)
            
        Returns:
            Dictionary with keys 'train', 'val', 'test' containing the split pairs
        """
        # First, split into train+val and test
        train_val_pairs, test_pairs = train_test_split(
            pairs, 
            test_size=self._test_size, 
            random_state=42
        )
        
        # Then, split train+val into train and val
        train_pairs, val_pairs = train_test_split(
            train_val_pairs,
            test_size=self._val_size,
            random_state=42
        )
        
        # Print actual split statistics
        total = len(pairs)
        print(f"\n📊 Dataset Split Summary:")
        print(f"   Total samples: {total}")
        print(f"   Train: {len(train_pairs)} ({len(train_pairs)/total*100:.1f}%)")
        print(f"   Val:   {len(val_pairs)} ({len(val_pairs)/total*100:.1f}%)")
        print(f"   Test:  {len(test_pairs)} ({len(test_pairs)/total*100:.1f}%)")
        print(f"   Expected ratio: {self.train_ratio:.0%} / {self.val_ratio:.0%} / {self.test_ratio:.0%}\n")
        
        return {
            'train': train_pairs,
            'val': val_pairs,
            'test': test_pairs
        }
    
    def preprocess(self):
        """
        Performs the preprocessing by copying images and labels to the appropriate directories.
        """
        # Find all valid image-label pairs
        all_pairs = self._find_image_label_pairs()
        
        if len(all_pairs) == 0:
            raise ValueError("No valid image-label pairs found in the raw data directory.")
        
        # Split the data
        split_data = self._split_data(all_pairs)
        
        # Copy files to appropriate directories
        for split_name, pairs in split_data.items():
            print(f"Processing {split_name} data ({len(pairs)} pairs)...")
            
            for img_path, label_path in pairs:
                # Copy image
                img_filename = os.path.basename(img_path)
                new_img_path = os.path.join(self.directories[split_name]['images'], img_filename)
                shutil.copy2(img_path, new_img_path)
                
                # Copy label
                label_filename = os.path.basename(label_path)
                new_label_path = os.path.join(self.directories[split_name]['labels'], label_filename)
                shutil.copy2(label_path, new_label_path)
        
        # Generate YAML configuration file for YOLOv8
        self._generate_yaml_config()
        
        print(f"Preprocessing completed! Dataset saved to {self.processed_data_path}")
    
    def _generate_yaml_config(self):
        """
        Generates a YAML configuration file for YOLOv8 training.
        """
        yaml_config = {
            'path': self.processed_data_path,
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 1,  # Number of classes (dog)
            'names': ['dog']  # Class names
        }
        
        yaml_path = os.path.join(self.processed_data_path, 'dataset.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_config, f, default_flow_style=False)
        
        print(f"YAML configuration saved to {yaml_path}")


# Example usage:
if __name__ == "__main__":
    # Define paths
    raw_data_path = "data/raw/detection_dataset/"  # Path to your raw dataset
    processed_data_path = "data/processed/detection"  # Path for processed dataset (matches exp script expectation)
    
    # Create preprocessor instance
    preprocessor = DetectionPreprocessor(raw_data_path, processed_data_path)
    
    # Run preprocessing
    preprocessor.preprocess()
