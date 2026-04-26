# Visual Dog Emotion Recognition System

A computer vision system for detecting dogs and recognizing their emotional states using deep learning.

---

## 🎯 Project Overview

This project implements a two-stage pipeline:
1. **Dog Detection**: Locate dogs in images using YOLOv8
2. **Emotion Classification**: Classify detected dog emotions (5 classes) using CNN architectures

**Key Features:**
- Modular architecture with separate detection and classification modules
- Multiple model architectures for comparison (ResNet50, AlexNet, GoogLeNet)
- Comprehensive training strategies with data augmentation
- Detailed evaluation metrics and visualization
- Reproducible experiments with fixed random seeds

---

## 📚 Documentation

### Task-Specific Training Guides

- **[📊 Classification Training Guide](CLASSIFICATION_TRAINING.md)** - Complete guide for emotion classification models (ResNet50, AlexNet, GoogLeNet)
  - Model architectures and configurations
  - Training strategies and optimization
  - Evaluation metrics and results
  - Bug fixes and troubleshooting

- **[🔍 Detection Training Guide](DETECTION_TRAINING.md)** - Complete guide for dog detection model (YOLOv8)
  - YOLOv8 architecture and backbone selection
  - Training configuration and augmentations
  - mAP evaluation and performance expectations
  - Advanced configuration options

### Common Information (Below)

- Quick Start
- Environment Setup
- Running Experiments
- Output Organization
- Best Practices

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Install PyTorch first (choose based on your hardware)
# CPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# GPU (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Verify setup
python test_setup.py
```

### 2. Download Data

```bash
bash scripts/download_data.sh
```

### 3. Run Experiments

```bash
# Run all experiments
bash scripts/run_all_experiments.sh

# Run single experiment
python experiments/exp04_classification_ResNet50_baseline.py
```

### 4. Inference Demo

```bash
bash scripts/inference_demo.sh
```

---

## 📊 Experiment Overview

The project includes **4 main experiments**:

### Detection (Exp01)

| Experiment | Model | Purpose |
|------------|-------|---------|
| Exp01 | YOLOv8 (Medium) | Dog face detection baseline |

### Classification (Exp04-06)

| Experiment | Model | Parameters | Key Feature |
|------------|-------|------------|-------------|
| Exp04 | ResNet50 | ~25.6M | Modern residual network |
| Exp05 | AlexNet | ~60M | Classic CNN architecture |
| Exp06 | GoogLeNet | ~7M | Efficient inception modules |

**Detailed configurations and comparisons**: See [CLASSIFICATION_TRAINING.md](CLASSIFICATION_TRAINING.md) and [DETECTION_TRAINING.md](DETECTION_TRAINING.md)

---

## 💻 Environment Requirements

### Core Dependencies

- **Python**: 3.8+
- **PyTorch**: 2.0+ (install separately based on hardware)
- **Ultralytics**: YOLOv8 >= 8.0.0
- **NumPy**: >= 1.24.0, < 2.0.0 (version constraint important!)

### Installation Notes

⚠️ **Important**: 
1. Install PyTorch **first** with appropriate CUDA version
2. NumPy must be < 2.0.0 for compatibility
3. See [DETECTION_TRAINING.md](DETECTION_TRAINING.md) for detailed environment setup

---

## 🏗️ Project Structure

```
CNN_A3/
├── experiments/              # Experiment scripts
│   ├── exp01_detection_YOLOv8_baseline.py
│   ├── exp04_classification_ResNet50_baseline.py
│   ├── exp05_classification_AlexNet.py
│   └── exp06_classification_GoogLeNet.py
│
├── src/                      # Source code
│   ├── data_processing/      # Data download and preprocessing
│   ├── models/               # Model definitions
│   ├── training/             # Training logic
│   ├── evaluation/           # Evaluation metrics
│   └── inference/            # Inference pipeline
│
├── scripts/                  # Shell scripts for automation
├── outputs/                  # Experiment results (auto-generated)
├── config.yaml               # Global configuration
│
├── CLASSIFICATION_TRAINING.md    # Classification training guide
├── DETECTION_TRAINING.md         # Detection training guide
└── README.md                     # This file
```

---

## 📂 Output Organization

Each experiment saves results to timestamped directories:

```
outputs/
├── exp01_detection_YOLOv8_baseline/
│   └── run_20260420_193045/
│       ├── model/          # Model weights and config
│       ├── logs/           # Training logs and reports
│       └── figures/        # Visualization plots
│
├── exp04_classification_ResNet50_baseline/
│   └── run_timestamp/
│       ├── model/
│       ├── logs/
│       └── figures/
│
└── ... (other experiments)
```

**Output Contents:**
- `model/best_model.pt` or `.pth`: Best model weights
- `logs/training_log.csv`: Epoch-by-epoch metrics
- `logs/experiment_report.md`: Comprehensive markdown report
- `figures/*.png`: Confusion matrices, ROC curves, PR curves, etc.

---

## 💡 Best Practices

### 1. Reproducibility

All experiments use fixed random seeds:
```python
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

### 2. Resource Efficiency

- Use mixed precision training (`use_amp=True`) for 2x speedup
- Choose appropriate model size for your hardware
- Monitor GPU memory: `watch -n 1 nvidia-smi`

### 3. Quick Validation

Use small subset for rapid testing:
```bash
python experiments/exp04_classification_ResNet50_baseline.py --use_small_subset
```

### 4. Monitoring Training

```bash
# Watch training progress
tail -f outputs/<experiment>/run_<timestamp>/logs/training_log.csv

# View visualizations
open outputs/<experiment>/run_<timestamp>/figures/*.png
```

---

## 🔧 Troubleshooting

### Common Issues

**Problem**: Out of Memory (OOM)
```
Solution:
- Reduce batch_size
- Use smaller model backbone
- Enable gradient accumulation
- Enable AMP if not already enabled
```

**Problem**: Training loss not decreasing
```
Solution:
- Check learning rate (try lower/higher)
- Verify data loading is correct
- Ensure model is in training mode: model.train()
- Check for label errors in dataset
```

**Problem**: Overfitting (val loss increasing)
```
Solution:
- Increase dropout rate
- Add more data augmentation
- Enable early stopping
- Increase weight_decay
```

**Detailed troubleshooting guides**:
- Classification issues: See [CLASSIFICATION_TRAINING.md](CLASSIFICATION_TRAINING.md) - Section "Bug Fixes & Troubleshooting"
- Detection issues: See [DETECTION_TRAINING.md](DETECTION_TRAINING.md) - Section "Troubleshooting"

---

## 📖 Additional Resources

- **[Classification Training Guide](CLASSIFICATION_TRAINING.md)**: Detailed model architectures, training strategies, optimization plans, and bug fixes for classification experiments
- **[Detection Training Guide](DETECTION_TRAINING.md)**: YOLOv8 configuration, backbone selection, evaluation metrics, and advanced settings for detection experiments

---

## 📝 License

[Add your license information here]

---

## 👥 Contributors

[Add contributor information here]

---

**Last Updated**: 2026-04-26
