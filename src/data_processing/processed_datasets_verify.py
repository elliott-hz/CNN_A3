"""
Processed Datasets Verification Module
Checks if datasets have been preprocessed and split, ready for training.
Does NOT download or process data - only verifies existence.
"""

import os
from pathlib import Path
import yaml


def verify_processed_datasets(config_path: str = "config.yaml") -> bool:
    """
    Verify that both detection and emotion datasets have been processed.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        True if both datasets are ready, False otherwise
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    processed_data_dir = Path(config['paths']['processed_data'])
    splitting_dir = Path("data/splitting")
    
    detection_processed_dir = processed_data_dir / "detection"
    detection_split_dir = splitting_dir / "detection_split"
    emotion_split_dir = splitting_dir / "emotion_split"
    
    print("=" * 80)
    print("PROCESSED DATASETS VERIFICATION")
    print("=" * 80)
    
    # Check detection dataset
    print("\n[1/2] Checking Detection Dataset...")
    detection_ok = _check_detection_dataset(detection_processed_dir, detection_split_dir)
    
    # Check emotion dataset
    print("\n[2/2] Checking Emotion Dataset...")
    emotion_ok = _check_emotion_dataset(emotion_split_dir)
    
    print("\n" + "=" * 80)
    if detection_ok and emotion_ok:
        print("✓ ALL DATASETS READY")
        print("=" * 80)
        print(f"Detection processed data: {detection_processed_dir}")
        print(f"Detection split metadata: {detection_split_dir}")
        print(f"Emotion split metadata: {emotion_split_dir}")
        return True
    else:
        print("✗ DATASETS NOT READY")
        print("=" * 80)
        if not detection_ok:
            print("  - Detection dataset needs preprocessing")
        if not emotion_ok:
            print("  - Emotion dataset needs parsing and splitting")
        print("\nPlease run preprocessing first:")
        print("  bash scripts/run_data_preprocessing.sh")
        print("\nOr run individually:")
        print("  python src/data_processing/detection_preprocessor.py")
        print("  python src/data_processing/emotion_preprocessor.py")
        return False


def _check_detection_dataset(processed_dir: Path, split_dir: Path) -> bool:
    """
    Check if detection dataset is processed and split.
    
    Args:
        processed_dir: Directory containing processed image files
        split_dir: Directory containing split metadata
        
    Returns:
        True if dataset is complete, False otherwise
    """
    # Check processed image directories (with letterbox padding)
    required_dirs = ['train', 'valid', 'test']
    annotation_dirs = ['annotations/train', 'annotations/valid', 'annotations/test']
    
    missing_dirs = []
    for dir_name in required_dirs + annotation_dirs:
        dir_path = processed_dir / dir_name
        if not dir_path.exists():
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"  ✗ Missing directories in {processed_dir}:")
        for d in missing_dirs:
            print(f"      - {d}")
        return False
    
    # Check if there are images in each split
    total_images = 0
    split_counts = {}
    
    for split_name in required_dirs:
        img_dir = processed_dir / split_name
        images = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
        count = len(images)
        split_counts[split_name] = count
        total_images += count
        
        if count == 0:
            print(f"  ✗ No images found in {img_dir}")
            return False
    
    # Check annotation files
    for split_name in required_dirs:
        ann_dir = processed_dir / 'annotations' / split_name
        annotations = list(ann_dir.glob('*.txt'))
        ann_count = len(annotations)
        
        if ann_count != split_counts[split_name]:
            print(f"  ⚠ Warning: {split_name} has {split_counts[split_name]} images but {ann_count} annotations")
    
    # Read metadata and display info
    import json
    metadata_file = processed_dir / "metadata.json"
    if not metadata_file.exists():
        print(f"  ✗ Metadata file not found: {metadata_file}")
        return False
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    print(f"  ✓ Detection dataset ready")
    print(f"    Total samples: {total_images}")
    for split_name, count in split_counts.items():
        print(f"    {split_name.capitalize()}: {count} images")
    print(f"    Preprocessing: {metadata.get('preprocessing_method', 'unknown')}")
    print(f"    Image size: {metadata.get('image_size', 'unknown')}x{metadata.get('image_size', 'unknown')}")
    
    return True


def _check_emotion_dataset(split_dir: Path) -> bool:
    """
    Check if emotion dataset is parsed and split.
    
    Args:
        split_dir: Directory containing split metadata
        
    Returns:
        True if dataset is complete, False otherwise
    """
    if not split_dir.exists():
        print(f"  ✗ Emotion split directory not found: {split_dir}")
        return False
    
    # Check required files
    required_files = [
        "train_split.json",
        "val_split.json",
        "test_split.json",
        "metadata.json"
    ]
    
    missing_files = []
    for filename in required_files:
        filepath = split_dir / filename
        if not filepath.exists():
            missing_files.append(filename)
    
    if missing_files:
        print(f"  ✗ Missing files in {split_dir}:")
        for f in missing_files:
            print(f"      - {f}")
        return False
    
    # Read metadata and display info
    import json
    metadata_file = split_dir / "metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    total_samples = metadata.get('total_samples', 0)
    splits = metadata.get('splits', {})
    
    print(f"  ✓ Emotion dataset ready")
    print(f"    Total samples: {total_samples}")
    print(f"    Train: {splits.get('train', 0)}")
    print(f"    Valid: {splits.get('val', 0)}")
    print(f"    Test: {splits.get('test', 0)}")
    
    return True


if __name__ == "__main__":
    import sys
    
    success = verify_processed_datasets()
    sys.exit(0 if success else 1)
