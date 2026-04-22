#!/bin/bash
# Run data preprocessing script
# This script preprocesses both detection and emotion datasets
# Optionally creates a small subset for quick testing

# Parse command line arguments
CREATE_SUBSET=false
TRAIN_SAMPLES=50
VAL_SAMPLES=10
TEST_SAMPLES=10

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --create-subset) CREATE_SUBSET=true ;;
        --train-samples) TRAIN_SAMPLES="$2"; shift ;;
        --val-samples) VAL_SAMPLES="$2"; shift ;;
        --test-samples) TEST_SAMPLES="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo "=========================================="
echo "Running Data Preprocessing"
echo "=========================================="

echo ""
echo "[1/2] Preprocessing Detection Dataset..."
echo "  - Applying letterbox resize (640x640)"
echo "  - Saving processed images as JPEG files"
echo "  - Saving split metadata to data/splitting/"
python src/data_processing/detection_preprocessor.py

if [ $? -ne 0 ]; then
    echo "Error: Detection preprocessing failed!"
    exit 1
fi

echo ""
echo "[2/2] Parsing and Splitting Emotion Dataset..."
echo "  - Organizing folder structure"
echo "  - Creating train/val/test splits"
echo "  - Saving split metadata to data/splitting/"
python src/data_processing/emotion_preprocessor.py

if [ $? -ne 0 ]; then
    echo "Error: Emotion preprocessing failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Data preprocessing complete!"
echo "=========================================="
echo ""
echo "Output directories:"
echo "  Processed images:"
echo "    - data/processed/detection/train/"
echo "    - data/processed/detection/valid/"
echo "    - data/processed/detection/test/"
echo ""
echo "  Split metadata:"
echo "    - data/splitting/detection_split/"
echo "    - data/splitting/emotion_split/"
echo ""

# Optional: Create small subset for testing
if [ "$CREATE_SUBSET" = true ]; then
    echo "=========================================="
    echo "Creating Small Subset for Testing"
    echo "=========================================="
    echo ""
    echo "Creating subset with:"
    echo "  Train samples: $TRAIN_SAMPLES"
    echo "  Val samples: $VAL_SAMPLES"
    echo "  Test samples: $TEST_SAMPLES"
    echo ""
    
    python src/data_processing/create_detection_subset.py \
        --input_dir data/processed/detection \
        --output_dir data/processed/detection_small \
        --train_samples $TRAIN_SAMPLES \
        --val_samples $VAL_SAMPLES \
        --test_samples $TEST_SAMPLES
    
    if [ $? -ne 0 ]; then
        echo "Warning: Subset creation failed, but main preprocessing succeeded."
    else
        echo ""
        echo "✓ Small subset created successfully!"
        echo "  Location: data/processed/detection_small/"
        echo "  Use this for quick testing and debugging."
    fi
    echo ""
fi

echo "You can now run experiments."
