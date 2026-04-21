# Quick Start Guide - Visual Dog Emotion Recognition

Get up and running in 10 minutes! This guide covers both **CPU (local testing)** and **GPU (AWS SageMaker)** setups.

---

## 🚀 Quick Overview

This project implements a two-stage pipeline:
1. **Dog Detection** (YOLOv8) - Find dogs in images
2. **Emotion Classification** (ResNet50) - Classify emotions: Angry, Happy, Relaxed, Frown, Alert

---

## 📋 Prerequisites

- Python 3.9 - 3.11 (⚠️ **Python 3.12+ not supported yet**)
- Git
- Datasets already downloaded and extracted to `data/raw/` directory

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
python test_setup.py
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

### ⚠️ Important: Simplified Data Workflow

This project uses a **minimal data preparation workflow**:
1. **Manual Download**: You must download and extract datasets to `data/raw/` before running any experiments
2. **Lightweight Parsing**: Run the parsing script to organize data paths and create train/val/test splits
3. **No Preprocessing**: Images are NOT resized, normalized, or augmented - they're loaded on-the-fly during training

**Key Benefits:**
- ✅ No memory issues (images not loaded into RAM)
- ✅ Fast setup (seconds instead of minutes)
- ✅ Flexible (preprocessing happens in training pipeline with augmentations)

### Dataset Structure

Your `data/raw/` directory should contain:

```
data/raw/
├── detection_dataset/          # Dog face detection dataset
│   ├── train_img/             # Training images (~5924 .jpg files)
│   ├── train_label/           # Training labels (YOLO format .txt files)
│   ├── val_img/               # Validation images (~230 .jpg files)
│   └── val_label/             # Validation labels (YOLO format .txt files)
│
└── emotion_dataset/            # Dog emotion classification dataset
    ├── alert/                 # ~1865 images
    ├── angry/                 # ~1865 images
    ├── frown/                 # ~1865 images
    ├── happy/                 # ~1865 images
    └── relax/                 # ~1865 images
```

### Step 1: Verify Raw Data Exists

Make sure you have downloaded and extracted both datasets:
- **Detection Dataset**: Dog Face Detection from Kaggle
- **Emotion Dataset**: Dog Emotions (5 classes) from Kaggle

Place them in the correct directories as shown above.

### Step 2: Run Data Parsing and Splitting

Once raw data is in place, run the parsing script:

```bash
bash scripts/run_data_preprocessing.sh
```

This will:
1. Parse YOLO-format annotations (for detection dataset)
2. Organize image paths by class (for emotion dataset)
3. Split data into train/valid/test sets (70/20/10)
4. Save lightweight JSON metadata to `data/processed/`

**Expected output:**
```
==========================================
Running Data Preprocessing
==========================================

[1/2] Parsing Detection Dataset...
================================================================================
DETECTION DATASET PARSING AND SPLITTING
================================================================================

[1/4] Loading training data...
  Loaded 5924 training images

[2/4] Loading validation data...
  Loaded 230 validation images

[3/4] Combining and splitting dataset (70/20/10)...

[4/4] Saving split metadata...
  Saved train split: 4307 images
  Saved val split: 1231 images
  Saved test split: 616 images
  Saved metadata

PARSING AND SPLITTING COMPLETE
Total samples: 6154
  Train: 4307 images
  Valid: 1231 images
  Test: 616 images

Note: Images are NOT preprocessed. They will be loaded during training.

[2/2] Parsing Emotion Dataset...
================================================================================
EMOTION DATASET PARSING AND SPLITTING
================================================================================

[1/3] Loading and organizing dataset...
  Loaded 9325 images across 5 classes

  Class distribution:
    alert: 1865
    angry: 1865
    frown: 1865
    happy: 1865
    relax: 1865

[2/3] Splitting dataset (70/20/10)...

[3/3] Saving split metadata...
  Saved train split: 6527 images
  Saved val split: 1865 images
  Saved test split: 933 images
  Saved metadata

PARSING AND SPLITTING COMPLETE
Total samples: 9325
  Train: 6527 images
  Valid: 1865 images
  Test: 933 images

Note: Images are NOT preprocessed. They will be loaded during training.

==========================================
Data parsing complete!
==========================================
```

### Step 3: Verify Parsed Data

After parsing, verify that data is ready:

```bash
python src/data_processing/processed_datasets_verify.py
```

You should see:
```
✓ ALL DATASETS READY
Detection dataset: data/processed/detection
Emotion dataset: data/processed/emotion
```

**Note:** Parsing only needs to be done **once**. All experiments will reuse the split metadata.

---

## 🧪 Test Your Setup

### Quick Logic Validation (CPU)

Before training, verify all components work:

```bash
source .venv/bin/activate  # If using venv
python test_setup.py
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
1. Verifies processed datasets exist
2. Loads preprocessed emotion data
3. Trains ResNet50 model
4. Evaluates on test set
5. Saves results to `outputs/exp04_classification_baseline/run_TIMESTAMP/`

**Expected runtime:**
- CPU: ~30-60 minutes (small dataset, few epochs)
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

### Issue: "Datasets not ready. Please run preprocessing first."

**Cause:** You haven't run the preprocessing script yet, or processed data is missing.

**Solution:**
```bash
# Step 1: Verify raw data exists
ls data/raw/detection_dataset/
ls data/raw/emotion_dataset/

# Step 2: Run preprocessing
bash scripts/run_data_preprocessing.sh

# Step 3: Verify processed data
python src/data_processing/processed_datasets_verify.py
```

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
4. Check [`test_setup.py`](test_setup.py) passes
5. Ensure data preprocessing completed: `python src/data_processing/processed_datasets_verify.py`

---

## ✅ Checklist Before GPU Training

- [ ] Local CPU tests pass (`python test_setup.py`)
- [ ] Raw datasets downloaded to `data/raw/`
- [ ] Data preprocessing completed (`bash scripts/run_data_preprocessing.sh`)
- [ ] Processed data verified (`python src/data_processing/processed_datasets_verify.py`)
- [ ] At least one experiment completed successfully on CPU
- [ ] GPU environment ready (AWS SageMaker)
- [ ] Sufficient storage space (datasets ~2-3GB)

---

**Happy Training! 🐕✨**

*Next: Run your first experiment and explore the outputs!*
