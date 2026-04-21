"""
Test script for data preprocessing modules
Validates that preprocessors can correctly load and process the raw datasets
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_processing.detection_preprocessor import DetectionPreprocessor
from src.data_processing.emotion_preprocessor import EmotionPreprocessor


def test_detection_preprocessor():
    """Test detection dataset preprocessing."""
    print("=" * 80)
    print("TESTING DETECTION PREPROCESSOR")
    print("=" * 80)
    
    preprocessor = DetectionPreprocessor()
    
    # Check if already processed
    if preprocessor.is_processed():
        print("\n✓ Detection data already preprocessed")
        print("Loading splits to verify...")
        
        # Try loading each split
        for split in ['train', 'valid', 'test']:
            X, y = preprocessor.load_split(split)
            print(f"  {split}: X shape={X.shape}, y samples={len(y)}")
            
            # Verify data integrity
            assert X.ndim == 4, f"X should be 4D array, got {X.ndim}D"
            assert X.shape[1] == 640 and X.shape[2] == 640, f"Image size should be 640x640"
            assert X.shape[3] == 3, f"Should have 3 channels (RGB)"
            assert len(X) == len(y), f"X and y should have same length"
            
            # Check a few samples
            if len(y) > 0:
                sample_bboxes = y[0]
                print(f"    Sample bbox shape: {sample_bboxes.shape if hasattr(sample_bboxes, 'shape') else 'N/A'}")
                if len(sample_bboxes) > 0:
                    print(f"    Sample bbox format: {sample_bboxes[0]}")
        
        print("\n✓ Detection data validation passed!")
        return True
    else:
        print("\n⚠ Detection data not preprocessed yet")
        print("Run the preprocessor to generate processed data")
        return False


def test_emotion_preprocessor():
    """Test emotion dataset preprocessing."""
    print("\n" + "=" * 80)
    print("TESTING EMOTION PREPROCESSOR")
    print("=" * 80)
    
    preprocessor = EmotionPreprocessor()
    
    # Check if already processed
    if preprocessor.is_processed():
        print("\n✓ Emotion data already preprocessed")
        print("Loading splits to verify...")
        
        # Try loading each split
        for split in ['train', 'valid', 'test']:
            X, y = preprocessor.load_split(split)
            print(f"  {split}: X shape={X.shape}, y shape={y.shape}")
            
            # Verify data integrity
            assert X.ndim == 4, f"X should be 4D array, got {X.ndim}D"
            assert X.shape[1] == 224 and X.shape[2] == 224, f"Image size should be 224x224"
            assert X.shape[3] == 3, f"Should have 3 channels (RGB)"
            assert len(X) == len(y), f"X and y should have same length"
            assert y.ndim == 1, f"y should be 1D array"
            
            # Check label range
            assert y.min() >= 0 and y.max() <= 4, f"Labels should be in range [0, 4]"
        
        print("\n✓ Emotion data validation passed!")
        return True
    else:
        print("\n⚠ Emotion data not preprocessed yet")
        print("Run the preprocessor to generate processed data")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("DATA PREPROCESSING VALIDATION")
    print("=" * 80)
    
    detection_ok = test_detection_preprocessor()
    emotion_ok = test_emotion_preprocessor()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Detection Preprocessor: {'✓ PASS' if detection_ok else '⚠ NEEDS PROCESSING'}")
    print(f"Emotion Preprocessor: {'✓ PASS' if emotion_ok else '⚠ NEEDS PROCESSING'}")
    
    if detection_ok and emotion_ok:
        print("\n🎉 All data preprocessing tests passed!")
    else:
        print("\n💡 To process data, run:")
        print("   python src/data_processing/detection_preprocessor.py")
        print("   python src/data_processing/emotion_preprocessor.py")


if __name__ == "__main__":
    main()
