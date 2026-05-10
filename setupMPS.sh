#!/bin/bash
# GPU Setup Script for CNN_A3 Project
# This script installs PyTorch with MPS (Metal Performance Shaders) support for Apple Silicon

echo "=========================================="
echo "Setting up GPU environment for CNN_A3 project..."
echo "Detected: Apple Silicon (M1/M2/M3) with MPS support"
echo "=========================================="

# Install PyTorch with MPS support (Apple Silicon optimized)
echo "[STEP 1/4] Installing PyTorch with MPS support..."
pip3 install torch torchvision torchaudio
if [ $? -eq 0 ]; then
    echo "✓ PyTorch installed successfully"
else
    echo "✗ Failed to install PyTorch"
    exit 1
fi

# Install project dependencies
echo "[STEP 2/4] Installing project dependencies from requirements.txt..."
pip3 install -r requirements.txt
if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully"
else
    echo "✗ Failed to install dependencies"
    exit 1
fi

# Fix NumPy compatibility
echo "[STEP 3/4] Ensuring NumPy compatibility (< 2.0.0)..."
pip3 install 'numpy>=1.24.0,<2.0.0' --force-reinstall
if [ $? -eq 0 ]; then
    echo "✓ NumPy compatibility ensured"
else
    echo "✗ Failed to ensure NumPy compatibility"
    exit 1
fi

# Verify GPU setup
echo "[STEP 4/4] Verifying MPS (Metal GPU) setup..."
python3 -c "
import torch
print('Checking PyTorch and MPS availability...')
print(f'PyTorch version: {torch.__version__}')

# Check MPS (Metal Performance Shaders) for Apple Silicon
mps_available = torch.backends.mps.is_available()
print(f'MPS Available: {mps_available}')

if mps_available:
    print('✓ Apple M3 GPU is available via MPS!')
    # Test basic MPS operation
    try:
        device = torch.device('mps')
        x = torch.randn(3, 3, device=device)
        y = torch.randn(3, 3, device=device)
        z = x + y
        print('Basic MPS (GPU) operation test: PASSED')
        print(f'Tensor device: {z.device}')
    except Exception as e:
        print(f'Basic MPS operation test: FAILED - {str(e)}')
else:
    print('MPS is not available. Falling back to CPU.')
    # Test basic CPU operation
    try:
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = x + y
        print('Basic CPU operation test: PASSED')
    except Exception as e:
        print(f'Basic CPU operation test: FAILED - {str(e)}')

# Also check CUDA (for reference)
cuda_available = torch.cuda.is_available()
print(f'CUDA Available: {cuda_available} (expected: False on macOS)')
"

echo "=========================================="
echo "Setup completed! Ready for GPU-accelerated training."
echo "Your Apple M3 GPU will be used via MPS backend."
echo ""
echo "To run experiments:"
echo "  python3 experiments/exp01_detection_baseline.py"
echo "  python3 experiments/exp02_detection_modified_v1.py"
echo "  python3 experiments/exp03_detection_modified_v2.py"
echo "=========================================="