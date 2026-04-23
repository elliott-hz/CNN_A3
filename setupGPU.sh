#!/bin/bash
# Simple GPU Setup Script for CNN_A3 Project
# This script installs PyTorch with CUDA support and verifies GPU availability

echo "=========================================="
echo "Setting up GPU environment for CNN_A3 project..."
echo "=========================================="

# Install PyTorch with CUDA support
echo "[STEP 1/4] Installing PyTorch with CUDA 11.8 support..."
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
if [ $? -eq 0 ]; then
    echo "✓ PyTorch with CUDA installed successfully"
else
    echo "✗ Failed to install PyTorch with CUDA"
    exit 1
fi

# Install project dependencies
echo "[STEP 2/4] Installing project dependencies from requirements.txt..."
pip install -r requirements.txt
if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "✗ Failed to install dependencies"
    exit 1
fi

# Fix NumPy compatibility
echo "[STEP 3/4] Ensuring NumPy compatibility (< 2.0.0)..."
pip install 'numpy>=1.24.0,<2.0.0' --force-reinstall
if [ $? -eq 0 ]; then
    echo "✓ NumPy compatibility ensured"
else
    echo "✗ Failed to ensure NumPy compatibility"
    exit 1
fi

# Verify GPU setup
echo "[STEP 4/4] Verifying GPU setup..."
python -c "
import torch
print('Checking CUDA availability...')
cuda_available = torch.cuda.is_available()
print(f'CUDA Available: {cuda_available}')

if cuda_available:
    gpu_count = torch.cuda.device_count()
    print(f'Number of GPUs available: {gpu_count}')
    if gpu_count > 0:
        gpu_name = torch.cuda.get_device_name(0)
        print(f'GPU 0: {gpu_name}')
        # Test basic CUDA operation
        try:
            x = torch.randn(3, 3).cuda()
            y = torch.randn(3, 3).cuda()
            z = x + y
            print('Basic CUDA operation test: PASSED')
        except Exception as e:
            print(f'Basic CUDA operation test: FAILED - {str(e)}')
else:
    print('CUDA is not available. Make sure you have a compatible GPU.')
"

echo "=========================================="
echo "Setup completed! Ready for GPU training."
echo "To run experiments:"
echo "  python experiments/exp01_detection_baseline.py"
echo "  python experiments/exp02_detection_modified_v1.py"
echo "  python experiments/exp03_detection_modified_v2.py"
echo "=========================================="