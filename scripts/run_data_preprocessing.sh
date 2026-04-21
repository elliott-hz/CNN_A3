#!/bin/bash
# Run data preprocessing script
# This script preprocesses both detection and emotion datasets

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
echo "You can now run experiments."
