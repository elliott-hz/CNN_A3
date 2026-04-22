"""
Detection Dataset Subset Creator
Creates a small subset from the processed detection dataset for quick testing.

This script randomly selects N samples from each split (train/valid/test) and
copies both images and their corresponding label files to maintain data integrity.

Usage:
    python src/data_processing/create_detection_subset.py \
        --train_samples 50 \
        --val_samples 10 \
        --test_samples 10 \
        --output_dir data/processed/detection_small
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm


class DetectionSubsetCreator:
    """
    Creates a small subset from processed detection dataset.
    Maintains image-label pairing and directory structure.
    """
    
    def __init__(self, input_dir: str = "data/processed/detection", 
                 output_dir: str = "data/processed/detection_small"):
        """
        Initialize subset creator.
        
        Args:
            input_dir: Path to full processed detection dataset
            output_dir: Path where subset will be created
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
    
    def create_subset(self, train_samples: int = 50, val_samples: int = 10, 
                     test_samples: int = 10):
        """
        Create a small subset with specified number of samples per split.
        
        Args:
            train_samples: Number of training samples to select
            val_samples: Number of validation samples to select
            test_samples: Number of test samples to select
        """
        print("=" * 80)
        print("CREATING DETECTION DATASET SUBSET")
        print("=" * 80)
        print(f"\nInput: {self.input_dir}")
        print(f"Output: {self.output_dir}")
        print(f"\nTarget samples:")
        print(f"  Train: {train_samples}")
        print(f"  Valid: {val_samples}")
        print(f"  Test: {test_samples}")
        
        # Create output directory structure
        self._create_output_structure()
        
        # Process each split
        splits_config = {
            'train': train_samples,
            'valid': val_samples,
            'test': test_samples
        }
        
        actual_counts = {}
        
        for split_name, target_count in splits_config.items():
            print(f"\n[1/3] Processing {split_name} split...")
            actual_count = self._process_split(split_name, target_count)
            actual_counts[split_name] = actual_count
        
        # Generate dataset.yaml for the subset
        print("\n[2/3] Generating dataset.yaml...")
        self._generate_dataset_yaml(actual_counts)
        
        # Save metadata
        print("\n[3/3] Saving subset metadata...")
        self._save_metadata(actual_counts, {'train': train_samples, 'valid': val_samples, 'test': test_samples})
        
        print("\n" + "=" * 80)
        print("SUBSET CREATION COMPLETE")
        print("=" * 80)
        print(f"\nSubset location: {self.output_dir}")
        print(f"\nActual samples created:")
        for split_name, count in actual_counts.items():
            print(f"  {split_name.capitalize()}: {count} images")
        total = sum(actual_counts.values())
        print(f"  Total: {total} images")
        print(f"\nYou can now use this subset for quick testing!")
    
    def _create_output_structure(self):
        """Create the output directory structure matching YOLOv8 format."""
        for split in ['train', 'valid', 'test']:
            (self.output_dir / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        print(f"✓ Created output directory structure at {self.output_dir}")
    
    def _process_split(self, split_name: str, target_count: int) -> int:
        """
        Process a single split: select samples and copy files.
        
        Args:
            split_name: 'train', 'valid', or 'test'
            target_count: Target number of samples
            
        Returns:
            Actual number of samples copied
        """
        input_split_dir = self.input_dir / split_name
        input_labels_dir = input_split_dir / 'labels'
        output_split_dir = self.output_dir / split_name
        output_labels_dir = output_split_dir / 'labels'
        
        if not input_split_dir.exists():
            print(f"  ⚠ Warning: {input_split_dir} does not exist, skipping...")
            return 0
        
        # Get all image files
        image_files = sorted(list(input_split_dir.glob('*.jpg')) + 
                           list(input_split_dir.glob('*.jpeg')) +
                           list(input_split_dir.glob('*.png')))
        
        if not image_files:
            print(f"  ⚠ Warning: No images found in {input_split_dir}")
            return 0
        
        available_count = len(image_files)
        actual_count = min(target_count, available_count)
        
        if actual_count < target_count:
            print(f"  ⚠ Warning: Only {available_count} images available, using all of them")
        
        # Randomly select samples
        selected_images = random.sample(image_files, actual_count)
        
        print(f"  Available: {available_count} images")
        print(f"  Selecting: {actual_count} images")
        
        # Copy selected images and labels
        for img_path in tqdm(selected_images, desc=f"  Copying {split_name}", leave=False):
            # Copy image
            dest_img = output_split_dir / img_path.name
            shutil.copy2(img_path, dest_img)
            
            # Copy corresponding label file
            label_path = input_labels_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                dest_label = output_labels_dir / f"{img_path.stem}.txt"
                shutil.copy2(label_path, dest_label)
            else:
                print(f"  ⚠ Warning: Label file not found for {img_path.name}")
        
        print(f"  ✓ Copied {actual_count} images and labels")
        return actual_count
    
    def _generate_dataset_yaml(self, actual_counts: dict):
        """
        Generate YOLOv8 dataset configuration YAML for the subset.
        
        Args:
            actual_counts: Dictionary with actual sample counts per split
        """
        yolo_config = {
            'path': str(self.output_dir),
            'train': 'train',
            'val': 'valid',
            'test': 'test',
            'nc': 1,
            'names': ['dog_face']
        }
        
        config_file = self.output_dir / "dataset.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(yolo_config, f, default_flow_style=False, sort_keys=False)
        
        print(f"  ✓ Generated dataset.yaml at {config_file}")
    
    def _save_metadata(self, actual_counts: dict, target_counts: dict):
        """
        Save metadata about the subset creation.
        
        Args:
            actual_counts: Actual sample counts per split
            target_counts: Target sample counts per split
        """
        import json
        from datetime import datetime
        
        metadata = {
            'created_at': datetime.now().isoformat(),
            'source_dataset': str(self.input_dir),
            'output_dataset': str(self.output_dir),
            'target_samples': target_counts,
            'actual_samples': actual_counts,
            'total_samples': sum(actual_counts.values()),
            'note': 'Random subset created for quick testing and debugging'
        }
        
        metadata_file = self.output_dir / "subset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Saved metadata to {metadata_file}")


def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Create a small subset from processed detection dataset for quick testing'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='data/processed/detection',
        help='Path to full processed detection dataset (default: data/processed/detection)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed/detection_small',
        help='Path where subset will be created (default: data/processed/detection_small)'
    )
    parser.add_argument(
        '--train_samples',
        type=int,
        default=50,
        help='Number of training samples to select (default: 50)'
    )
    parser.add_argument(
        '--val_samples',
        type=int,
        default=10,
        help='Number of validation samples to select (default: 10)'
    )
    parser.add_argument(
        '--test_samples',
        type=int,
        default=10,
        help='Number of test samples to select (default: 10)'
    )
    
    args = parser.parse_args()
    
    try:
        creator = DetectionSubsetCreator(
            input_dir=args.input_dir,
            output_dir=args.output_dir
        )
        creator.create_subset(
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            test_samples=args.test_samples
        )
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
