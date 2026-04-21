"""
Processed Datasets Verification Module
Checks if datasets have been preprocessed and are ready for use.
Does NOT download or process data - only verifies existence.
"""

import os
from pathlib import Path
import yaml


def verify_processed_datasets(config_path: str = "config.yaml") -> bool:
    """
    Verify that both detection and emotion datasets have been preprocessed.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        True if both datasets are processed, False otherwise
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    processed_data_dir = Path(config['paths']['processed_data'])
    detection_dir = processed_data_dir / "detection"
    emotion_dir = processed_data_dir / "emotion"
    
    print("=" * 80)
    print("PROCESSED DATASETS VERIFICATION")
    print("=" * 80)
    
    # Check detection dataset
    print("\n[1/2] Checking Detection Dataset...")
    detection_ok = _check_processed_dataset(detection_dir, "Detection")
    
    # Check emotion dataset
    print("\n[2/2] Checking Emotion Dataset...")
    emotion_ok = _check_processed_dataset(emotion_dir, "Emotion")
    
    print("\n" + "=" * 80)
    if detection_ok and emotion_ok:
        print("✓ ALL DATASETS READY")
        print("=" * 80)
        print(f"Detection dataset: {detection_dir}")
        print(f"Emotion dataset: {emotion_dir}")
        return True
    else:
        print("✗ DATASETS NOT READY")
        print("=" * 80)
        if not detection_ok:
            print("  - Detection dataset needs preprocessing")
        if not emotion_ok:
            print("  - Emotion dataset needs preprocessing")
        print("\nPlease run preprocessing first:")
        print("  python src/data_processing/detection_preprocessor.py")
        print("  python src/data_processing/emotion_preprocessor.py")
        return False


def _check_processed_dataset(dataset_dir: Path, dataset_name: str) -> bool:
    """
    Check if a processed dataset exists and contains required files.
    
    Args:
        dataset_dir: Directory containing processed dataset
        dataset_name: Name for display purposes
        
    Returns:
        True if dataset is complete, False otherwise
    """
    if not dataset_dir.exists():
        print(f"  ✗ {dataset_name} directory not found: {dataset_dir}")
        return False
    
    # Check required files
    required_files = [
        "X_train.npy",
        "X_valid.npy",
        "X_test.npy",
        "y_train.npy",
        "y_valid.npy",
        "y_test.npy",
        "metadata.json"
    ]
    
    missing_files = []
    for filename in required_files:
        filepath = dataset_dir / filename
        if not filepath.exists():
            missing_files.append(filename)
    
    if missing_files:
        print(f"  ✗ {dataset_name} dataset incomplete. Missing files:")
        for f in missing_files:
            print(f"    - {f}")
        return False
    
    # Load and display metadata
    try:
        import json
        with open(dataset_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        print(f"  ✓ {dataset_name} dataset verified")
        print(f"    Total samples: {metadata.get('total_samples', 'N/A')}")
        print(f"    Train: {metadata.get('train_samples', 'N/A')}")
        print(f"    Valid: {metadata.get('valid_samples', 'N/A')}")
        print(f"    Test: {metadata.get('test_samples', 'N/A')}")
        return True
        
    except Exception as e:
        print(f"  ⚠ {dataset_name} metadata error: {e}")
        return False


if __name__ == "__main__":
    verify_processed_datasets()
