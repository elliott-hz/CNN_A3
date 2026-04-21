#!/bin/bash
# Run data preprocessing script
# This script preprocesses both detection and emotion datasets

echo "=========================================="
echo "Running Data Preprocessing"
echo "=========================================="

echo ""
echo "[1/2] Preprocessing Detection Dataset..."
python src/data_processing/detection_preprocessor.py

if [ $? -ne 0 ]; then
    echo "Error: Detection preprocessing failed!"
    exit 1
fi

echo ""
echo "[2/2] Preprocessing Emotion Dataset..."
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
echo "Processed data saved to:"
echo "  - data/processed/detection/"
echo "  - data/processed/emotion/"
echo ""
echo "You can now run experiments."
