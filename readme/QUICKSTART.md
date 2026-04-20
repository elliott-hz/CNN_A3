# Quick Start Guide - Visual Dog Emotion Recognition

Get up and running in 10 minutes! This guide covers both **CPU (local testing)** and **GPU (AWS SageMaker)** setups.

---

## 🚀 Quick Overview

This project implements a two-stage pipeline:
1. **Dog Detection** (YOLOv8) - Find dogs in images
2. **Emotion Classification** (ResNet50) - Classify emotions: Angry, Fearful, Happy, Sad

---

## 📋 Prerequisites

- Python 3.9 - 3.11 (⚠️ **Python 3.12+ not supported yet**)
- Git
- Kaggle account (for dataset download)

---

## 🔧 Installation

### Option A: CPU Setup (Local Testing) ✅ Recommended for First-Time Users

Perfect for validating code logic before GPU training.

#### Step 1: Create Virtual Environment

```bash
cd CNN_A3
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

#### Step 2: Install PyTorch (CPU Version)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### Step 3: Install Other Dependencies

```bash
pip install -r requirements.txt
```

#### Step 4: Fix NumPy Compatibility

Some packages may upgrade NumPy to 2.x, which is incompatible. Force downgrade:

```bash
pip install 'numpy>=1.24.0,<2.0.0' --force-reinstall
```

✅ **Verify Installation:**
```bash
python test_logic_validation.py
```

Expected output: `🎉 ALL TESTS PASSED!`

---

### Option B: GPU Setup (AWS SageMaker) 🚀 For Production Training

Use this for full-scale model training with NVIDIA T4 GPU (16GB VRAM).

#### Step 1: Activate Conda Environment (SageMaker Default)

```bash
conda activate pytorch_p310
```

#### Step 2: Install Dependencies

```bash
cd CNN_A3
pip install -r requirements.txt
```

✅ **Verify GPU Availability:**
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

Expected output: `CUDA Available: True`

---

## 📊 Dataset Setup

### Step 1: Configure Kaggle API

```bash
# Create directory
mkdir -p ~/.kaggle

# Download kaggle.json from:
# https://www.kaggle.com/<your-username>/account

# Place it in ~/.kaggle/ and set permissions
chmod 600 ~/.kaggle/kaggle.json
```

### Step 2: Accept Dataset Rules

Visit these pages and click "Accept Rules":
1. [Dog Emotion Dataset](https://www.kaggle.com/datasets) *(replace with actual link)*
2. [Dog Detection Dataset](https://www.kaggle.com/datasets) *(replace with actual link)*

### Step 3: Download Datasets (Automatic)

The first experiment will automatically download and preprocess datasets. No manual action needed!

---

## 🧪 Test Your Setup

### Quick Logic Validation (CPU)

Before training, verify all components work:

```bash
source .venv/bin/activate  # If using venv
python test_logic_validation.py
```

This tests:
- ✅ Module imports
- ✅ Model creation & forward pass
- ✅ Data loading utilities
- ✅ Output directory structure

---

## 🎯 Run Your First Experiment

### Start with Classification Baseline (Simplest)

```bash
python experiments/exp04_classification_baseline.py
```

**What happens:**
1. Downloads datasets (first time only)
2. Preprocesses data
3. Trains ResNet50 model
4. Evaluates on test set
5. Saves results to `outputs/exp04_classification_baseline/run_TIMESTAMP/`

**Expected runtime:**
- CPU: ~30-60 minutes (small dataset)
- GPU: ~5-10 minutes

---

## 📂 Understanding Outputs

After running an experiment:

```
outputs/exp04_classification_baseline/run_20260420_201500/
├── model/
│   ├── best_model.pth          ← Best model weights
│   └── model_config.json       ← Configuration used
├── logs/
│   ├── training_log.csv        ← Epoch-by-epoch metrics
│   ├── experiment_report.md    ← Human-readable summary
│   └── evaluation_metrics.json ← Test set performance
└── figures/                    ← Visualizations
    ├── confusion_matrix.png
    ├── training_curves.png
    └── ...
```

**View Results:**
```bash
# Read the report
cat outputs/exp04_classification_baseline/run_*/logs/experiment_report.md

# Check metrics
cat outputs/exp04_classification_baseline/run_*/logs/evaluation_metrics.json
```

---

## 🔄 All Experiments

### Detection Experiments (YOLOv8)

```bash
python experiments/exp01_detection_baseline.py      # Baseline YOLOv8
python experiments/exp02_detection_modified_v1.py   # Modified v1
python experiments/exp03_detection_modified_v2.py   # Modified v2
```

### Classification Experiments (ResNet50)

```bash
python experiments/exp04_classification_baseline.py      # Baseline ResNet50
python experiments/exp05_classification_modified_v1.py   # Modified v1
python experiments/exp06_classification_modified_v2.py   # Modified v2
```

### Run All Experiments

```bash
bash scripts/run_all_experiments.sh
```

---

## 🔍 Inference Demo

Test trained models on your own images:

```bash
bash scripts/inference_demo.sh path/to/your/image.jpg
```

---

## ⚙️ Configuration

### Modify Hyperparameters

Edit any experiment script to change settings:

```python
training_config = {
    'learning_rate': 0.001,
    'batch_size': 16,           # Reduce if OOM
    'epochs': 30,
    'use_amp': True,            # Enable mixed precision (GPU only)
    'gradient_accumulation_steps': 1,
}
```

### Global Settings

Edit [`config.yaml`](../config.yaml) for project-wide defaults.

---

## 🐛 Troubleshooting

### Issue: "No module named 'torch'"

**Solution:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

### Issue: "NumPy compatibility error" or "_ARRAY_API not found"

**Cause:** Some packages upgraded NumPy to 2.x

**Solution:**
```bash
pip install 'numpy>=1.24.0,<2.0.0' --force-reinstall
```

---

### Issue: "Kaggle API authentication error"

**Solution:**
1. Ensure `~/.kaggle/kaggle.json` exists
2. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`
3. Accept dataset rules on Kaggle website
4. Test: `kaggle datasets list`

---

### Issue: "Out of Memory" (GPU)

**Solution:** Reduce batch size and enable optimizations:

```python
training_config = {
    'batch_size': 8,                  # Reduce from 32
    'gradient_accumulation_steps': 2, # Simulate larger batch
    'use_amp': True,                  # Enable mixed precision
    'freeze_backbone': True,          # Freeze ResNet backbone initially
}
```

---

### Issue: "CUDA not available"

**Check:**
```python
import torch
print(torch.cuda.is_available())  # Should be True on GPU
```

**Note:** Code automatically falls back to CPU if CUDA unavailable, but training will be slower.

---

### Issue: Slow training on CPU

**Normal!** CPU is 10-20x slower than GPU. Use CPU only for:
- Code validation
- Small-scale testing
- Debugging

For production training, use AWS SageMaker GPU.

---

## 💡 Pro Tips

### 1. Test Locally First (CPU)
```bash
# Quick validation with minimal epochs
# Edit experiment script:
training_config['epochs'] = 2  # Just verify it works
python experiments/exp04_classification_baseline.py
```

### 2. Monitor GPU Usage (SageMaker)
```bash
nvidia-smi  # Real-time GPU monitoring
```

### 3. Compare Experiments
```bash
# List all runs
ls outputs/exp04_classification_baseline/

# Compare reports
diff outputs/exp04_classification_baseline/run_001/logs/experiment_report.md \
     outputs/exp04_classification_baseline/run_002/logs/experiment_report.md
```

### 4. Resume Interrupted Training
Best model is auto-saved. Check:
```bash
ls outputs/*/run_*/model/best_model.pth
```

### 5. Clean Up Old Runs
```bash
# Remove old experiment runs (keep latest)
find outputs/ -type d -name "run_*" | sort | head -n -1 | xargs rm -rf
```

---

## 📚 Documentation

- **[README.md](README.md)** - Complete project overview
- **[PartC-Project Structure.md](PartC-Project%20Structure.md)** - Architecture details

---

## 🆘 Still Stuck?

1. Check error messages carefully
2. Verify all dependencies: `pip list`
3. Review experiment script that failed
4. Check [`test_logic_validation.py`](test_logic_validation.py) passes

---

## ✅ Checklist Before GPU Training

- [ ] Local CPU tests pass (`python test_logic_validation.py`)
- [ ] At least one experiment completed successfully on CPU
- [ ] Kaggle API configured
- [ ] GPU environment ready (AWS SageMaker)
- [ ] Sufficient storage space (datasets ~2-3GB)

---

**Happy Training! 🐕✨**

*Next: Run your first experiment and explore the outputs!*
