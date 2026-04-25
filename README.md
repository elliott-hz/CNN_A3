# Visual Dog Emotion Recognition - Complete Project Documentation

This repository contains a complete solution for visual dog emotion recognition using a two-stage deep learning pipeline:
1. **Dog face detection** using YOLOv8
2. **Emotion classification** using ResNet50
3. **Web Application** with React + FastAPI for real-time inference

---

## Table of Contents

### Part 1: Core ML Pipeline
1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [Module Descriptions](#3-module-descriptions)
4. [Dataset Sources](#4-dataset-sources)
5. [Data Processing Workflow](#5-data-processing-workflow)
6. [Experiment Descriptions](#6-experiment-descriptions)
7. [Data Flow](#7-data-flow)
8. [Output Organization](#8-output-organization)
9. [Key Design Principles](#9-key-design-principles)

### Part 2: Web Application
10. [Web App Overview](#10-web-app-overview)
11. [Web App Quick Start](#11-web-app-quick-start)
12. [Web App Architecture](#12-web-app-architecture)
13. [Using the Web Application](#13-using-the-web-application)
14. [API Documentation](#14-api-documentation)
15. [Web App Troubleshooting](#15-web-app-troubleshooting)
16. [Performance & Deployment](#16-performance--deployment)

---

## 1. Project Overview

This project implements a **two-stage deep learning pipeline** for visual dog emotion recognition:
- **Stage 1**: Dog face detection using YOLOv8
- **Stage 2**: Emotion classification using ResNet50

### Core Architecture
```
Input Image/Video
       ↓
┌─────────────────────┐
│  Dog Face Detection  │ ← YOLOv8 (configurable variants)
│  (Bounding Box)      │
└─────────────────────┘
       ↓ (crop face)
┌─────────────────────┐
│ Emotion Classification│ ← ResNet50 (configurable variants)
│  (5 emotions)        │
└─────────────────────┘
       ↓
Output: BBox + Emotion Label
```

### Technical Stack
- **Framework**: PyTorch 2.0+ with torchvision
- **Hardware**: AWS SageMaker JupyterLab with NVIDIA T4 GPU (16GB VRAM)
- **Detection Model**: YOLOv8 (single base model with configurable parameters)
- **Classification Model**: ResNet50 (single base model with configurable parameters)
- **Datasets**: 
  - Dog Face Detection Dataset (~6,000 images)
  - Dog Emotion Dataset (~9,000 images, 5 classes)

---

## 2. Directory Structure

```
CNN_A3/
│
├── README.md                          # This file
├── config.yaml                        # Global configuration (paths, defaults)
├── requirements.txt                   # Python dependencies
├── test_setup.py                      # Test script to verify environment
├── check_image_dimensions.py          # Utility to check image dimensions
├── xxxx.md                            # Additional notes
│
├── data/                              # Data directory (auto-created)
│   ├── raw/                           # Original downloaded datasets
│   │   ├── detection_dataset/         # Dog Face Detection (Kaggle)
│   │   └── emotion_dataset/           # Dog Emotion (Kaggle)
│   │
│   ├── processed/                     # Preprocessed & split datasets
│   │   ├── detection/                 # Detection splits (YOLO format)
│   │   │   ├── images/
│   │   │   │   ├── train/
│   │   │   │   ├── val/
│   │   │   │   └── test/
│   │   │   ├── labels/
│   │   │   │   ├── train/
│   │   │   │   ├── val/
│   │   │   │   └── test/
│   │   │   ├── dataset.yaml           # YOLOv8 config
│   │   │   └── metadata.json          # Processing metadata
│   │   └── detection_small/           # Small subset for testing
│   │       ├── images/
│   │       ├── labels/
│   │       ├── dataset.yaml
│   │       └── subset_metadata.json
│   │
│   └── splitting/                     # Split indices for emotion dataset
│       ├── detection_split/           # Detection dataset splits
│       │   ├── train_split.json
│       │   ├── val_split.json
│       │   ├── test_split.json
│       │   └── metadata.json
│       └── emotion_split/             # Emotion dataset splits
│           ├── train_split.json
│           ├── val_split.json
│           ├── test_split.json
│           └── metadata.json
│
├── src/                               # Source code package
│   ├── __init__.py
│   │
│   ├── data_processing/               # Data preprocessing modules
│   │   ├── __init__.py
│   │   ├── detection_preprocessor.py  # Detection dataset preprocessing
│   │   ├── emotion_preprocessor.py    # Emotion dataset preprocessing
│   │   └── create_detection_subset.py # Create small subset for testing
│   │
│   ├── models/                        # Model definitions (SIMPLIFIED)
│   │   ├── __init__.py
│   │   ├── detection_model.py         # ONE model: YOLOv8Detector
│   │   └── classification_model.py    # ONE model: ResNet50Classifier
│   │
│   ├── training/                      # Training frameworks
│   │   ├── __init__.py
│   │   ├── detection_trainer.py       # Detection training logic
│   │   └── classification_trainer.py  # Classification training logic
│   │
│   ├── evaluation/                    # Evaluation frameworks
│   │   ├── __init__.py
│   │   ├── detection_evaluator.py     # Detection metrics (mAP, IoU)
│   │   └── classification_evaluator.py # Classification metrics
│   │
│   ├── inference/                     # Inference pipeline
│   │   ├── __init__.py
│   │   ├── detection_inference.py     # Detection-only inference
│   │   ├── classification_inference.py # Classification-only inference
│   │   └── pipeline_inference.py      # End-to-end stacked inference
│   │
│   └── utils/                         # Utility functions
│       ├── __init__.py
│       ├── file_utils.py              # File I/O helpers
│       └── logger.py                  # Logging setup
│
├── experiments/                       # 6 Experiment scripts
│   ├── __init__.py
│   ├── exp01_detection_baseline.py        # YOLOv8 baseline
│   ├── exp02_detection_modified_v1.py     # YOLOv8 modified v1
│   ├── exp03_detection_modified_v2.py     # YOLOv8 modified v2
│   ├── exp04_classification_baseline.py   # ResNet50 baseline
│   ├── exp05_classification_modified_v1.py # ResNet50 modified v1
│   └── exp06_classification_modified_v2.py # ResNet50 modified v2
│
├── outputs/                           # Experiment outputs (timestamped runs)
│   ├── exp01_detection_baseline/
│   ├── exp02_detection_modified_v1/
│   ├── exp03_detection_modified_v2/
│   ├── exp04_classification_baseline/
│   ├── exp05_classification_modified_v1/
│   └── exp06_classification_modified_v2/
│
├── scripts/                           # Convenience scripts
│   ├── inference_demo.sh              # Demo inference
│   └── run_data_preprocessing.sh      # Run data preprocessing
│
└── notebooks/                         # Jupyter notebooks (if any)
```

---

## 3. Module Descriptions

### 3.1 Data Processing (`src/data_processing/`)

#### Purpose
Handle dataset preprocessing, formatting, and splitting into train/val/test sets.

#### Key Features
- **Detection Preprocessor**: Processes detection dataset with letterbox resize to preserve aspect ratio
- **Emotion Preprocessor**: Organizes emotion dataset and creates train/val/test splits
- **Subset Creator**: Creates small subsets for quick testing

#### Files
- [`detection_preprocessor.py`](src/data_processing/detection_preprocessor.py): Preprocesses detection dataset
- [`emotion_preprocessor.py`](src/data_processing/emotion_preprocessor.py): Preprocesses emotion dataset  
- [`create_detection_subset.py`](src/data_processing/create_detection_subset.py): Creates small subsets for testing

### 3.2 Model Definitions (`src/models/`)

#### **Simplified Design Philosophy**
Each task uses **ONE base model class** with **configurable parameters** to create different variants across experiments.

#### Detection Model: [`detection_model.py`](src/models/detection_model.py)
```python
class YOLOv8Detector:
    """
    Single YOLOv8 wrapper with configurable parameters.
    
    Configuration options:
    - backbone_depth: 'n', 's', 'm', 'l', 'x' (model size)
    - input_size: 640, 1280, etc.
    - confidence_threshold: 0.3, 0.5, 0.7
    - nms_iou_threshold: 0.45, 0.5, 0.6
    - anchor_settings: custom anchor boxes
    """
    def __init__(self, config: dict):
        self.config = config
        # Initialize YOLOv8 with specified parameters
```

#### Classification Model: [`classification_model.py`](src/models/classification_model.py)
```python
class ResNet50Classifier:
    """
    Single ResNet50 wrapper with configurable parameters.
    
    Configuration options:
    - dropout_rate: 0.3, 0.5, 0.7
    - additional_fc_layers: True/False
    - freeze_strategy: 'all', 'partial', 'none'
    - num_classes: 5 (fixed for this task)
    - use_batch_norm: True/False
    """
    def __init__(self, config: dict):
        self.config = config
        # Initialize ResNet50 with specified parameters
```

---

## 4. Dataset Sources

### Detection Dataset

**Source**: Dog face detection dataset

**Processed Structure**:
```
data/processed/detection/
├── images/
│   ├── train/          # Training images (.jpg)
│   ├── val/            # Validation images (.jpg)
│   └── test/           # Test images (.jpg)
├── labels/
│   ├── train/          # Training labels (.txt, YOLO format)
│   ├── val/            # Validation labels (.txt, YOLO format)
│   └── test/           # Test labels (.txt, YOLO format)
├── dataset.yaml        # YOLOv8 configuration
└── metadata.json       # Processing metadata
```

**Label Format** (YOLO):
```
class_id x_center y_center width height
```
- All coordinates normalized to [0, 1]
- Each `.txt` file may contain multiple bounding boxes (multi-dog images)
- Single class: dog (class_id = 0)

### Emotion Dataset

**Source**: Dog emotion classification dataset

**Raw Structure**:
```
data/raw/emotion_dataset/
├── alert/      # Alert emotion images
├── angry/      # Angry emotion images
├── frown/      # Frown emotion images
├── happy/      # Happy emotion images
└── relax/      # Relax emotion images
```

**Classes**: 5 emotion categories
- Total: ~9,325 images
- Balanced distribution across classes

**Split Structure**:
```
data/splitting/emotion_split/
├── train_split.json    # Training split metadata
├── val_split.json      # Validation split metadata
├── test_split.json     # Test split metadata
└── metadata.json       # Overall dataset metadata
```

---

## 5. Data Processing Workflow

### Preprocessing Scripts

To prepare the datasets for training, run the preprocessing script:

```bash
bash scripts/run_data_preprocessing.sh
```

This script:
1. Runs [`detection_preprocessor.py`](src/data_processing/detection_preprocessor.py) to process the detection dataset
2. Runs [`emotion_preprocessor.py`](src/data_processing/emotion_preprocessor.py) to parse and split the emotion dataset
3. Optionally creates a small subset for quick testing

#### Detection Dataset Processing

- Preserves original aspect ratios using letterbox resize
- Saves processed images as JPEG files
- Maintains YOLO format for annotations
- Generates `dataset.yaml` for YOLOv8 training

#### Emotion Dataset Processing

- Organizes images into train/val/test splits
- Creates JSON files with image paths and labels
- No preprocessing (images loaded during training)
- Maintains class balance across splits

### Creating a Small Subset for Testing

To create a small subset of the detection dataset for quick testing:

```bash
bash scripts/run_data_preprocessing.sh --create-subset
```

Customize the number of samples:
```bash
bash scripts/run_data_preprocessing.sh \
    --create-subset \
    --train-samples 50 \
    --val-samples 10 \
    --test-samples 10
```

---

## 6. Quick Start Guide

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- NVIDIA GPU with CUDA support (recommended)

### Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Prepare datasets (see Data Processing section)

### Running Preprocessing
```bash
bash scripts/run_data_preprocessing.sh
```

### Running Experiments
```bash
# Detection baseline
python experiments/exp01_detection_baseline.py

# Classification baseline
python experiments/exp04_classification_baseline.py

# With small subset for quick testing
python experiments/exp01_detection_baseline.py --use-small-subset
python experiments/exp04_classification_baseline.py --use_small_subset
```

### Inference Demo
```bash
bash scripts/inference_demo.sh
```

---

## 7. Experiment Descriptions

The project includes 6 experiments divided into two categories:

### 7.1 Detection Experiments (Exp01-03)

#### Exp01: Detection Baseline ([`exp01_detection_baseline.py`](experiments/exp01_detection_baseline.py))
- **Model**: YOLOv8 medium (m)
- **Configuration**: backbone='m', input_size=640, confidence=0.5
- **Purpose**: Establish baseline performance for dog face detection

#### Exp02: Detection Modified V1 ([`exp02_detection_modified_v1.py`](experiments/exp02_detection_modified_v1.py))
- **Model**: YOLOv8 large (l)
- **Configuration**: backbone='l', input_size=1280, confidence=0.6
- **Purpose**: Test larger model with higher resolution input

#### Exp03: Detection Modified V2 ([`exp03_detection_modified_v2.py`](experiments/exp03_detection_modified_v2.py))
- **Model**: YOLOv8 small (s)
- **Configuration**: backbone='s', input_size=640, confidence=0.4
- **Purpose**: Test smaller model for faster inference

### 7.2 Classification Experiments (Exp04-06)

#### Exp04: Classification Baseline ([`exp04_classification_baseline.py`](experiments/exp04_classification_baseline.py))
- **Model**: ResNet50 with partial freezing
- **Configuration**: dropout=0.5, freeze_backbone=True
- **Purpose**: Establish baseline performance for emotion classification

#### Exp05: Classification Modified V1 ([`exp05_classification_modified_v1.py`](experiments/exp05_classification_modified_v1.py))
- **Model**: ResNet50 with additional layers
- **Configuration**: dropout=0.7, additional_fc_layers=True, pretrained=True
- **Purpose**: Test model with additional fully connected layers

#### Exp06: Classification Modified V2 ([`exp06_classification_modified_v2.py`](experiments/exp06_classification_modified_v2.py))
- **Model**: ResNet50 without freezing
- **Configuration**: dropout=0.3, freeze_backbone=False, all layers trainable
- **Purpose**: Test fine-tuning of the entire model

### Common Training Configuration
All experiments use the following enhanced training configuration:
- **Epochs**: 120 (increased from initial values)
- **Early stopping**: Patience of 15 epochs (12.5% of total epochs)
- **Mixed precision**: Enabled (use_amp=True) for faster training and reduced memory usage
- **Label smoothing**: 0.05-0.1 depending on experiment
- **Class weighting**: Enabled to handle potential imbalances

---

## 8. Data Flow

### Training Data Flow

#### Detection Training
1. Load dataset configuration from `data/processed/detection/dataset.yaml`
2. YOLOv8 automatically loads images and labels from specified paths
3. Apply built-in augmentations and preprocessing
4. Train the detection model
5. Evaluate on test set and generate metrics

#### Classification Training
1. Load preprocessed split metadata from `data/splitting/emotion_split/`
2. Load images on-the-fly during training
3. Apply transforms (resize, normalize, augmentations)
4. Train the classification model
5. Evaluate on test set and generate metrics

### Inference Data Flow
1. Detect dog faces using trained detection model
2. Crop detected faces from original image
3. Classify emotions using trained classification model
4. Combine results and output bounding boxes with emotion labels

---

## 9. Output Organization

Each experiment saves outputs to timestamped directories:

```
outputs/
├── exp01_detection_baseline/
│   ├── run_YYYYMMDD_HHMMSS/
│   │   ├── model/
│   │   │   ├── best_model.pt
│   │   │   └── model_config.json
│   │   ├── logs/
│   │   │   ├── training_log.csv
│   │   │   └── experiment_report.md
│   │   └── figures/
│   │       ├── precision_recall_curve.png
│   │       ├── IoU_distribution.png
│   │       └── sample_detections.png
│   └── run_YYYYMMDD_HHMMSS/
│
├── exp02_detection_modified_v1/
├── exp03_detection_modified_v2/
├── exp04_classification_baseline/
├── exp05_classification_modified_v1/
└── exp06_classification_modified_v2/
```

This organization ensures that:
- Multiple runs of the same experiment are preserved
- Results are organized by timestamp for easy tracking
- All artifacts (models, logs, figures) are grouped together

---

## 10. Key Design Principles

### 10.1 Simplified Architecture
- One base model class per task with configurable parameters
- Avoid creating multiple similar model classes
- Use configuration dictionaries to create model variants

### 10.2 Reproducible Research
- Fixed random seeds for reproducible results
- Detailed logging of experimental conditions
- Automatic saving of model configurations

### 10.3 Resource Efficiency
- Mixed precision training to reduce memory usage
- Optimized batch sizes for available GPU memory
- Two-stage training (freeze backbone first, then fine-tune)

### 10.4 Scalability
- Modular design allowing easy addition of new experiments
- Consistent API across different model variants
- Separation of data processing from model training

### 10.5 Best Practices
- Early stopping to prevent overfitting
- Comprehensive evaluation metrics
- Visualization of results for analysis
- Proper train/validation/test separation

**Format**: JPEG (quality ~95, default OpenCV)
**Size**: 640x640 pixels (with padding)
**Color Space**: BGR (OpenCV default)
**Naming**: Sequential `img_XXXXX.jpg`

##### Annotations

**Format**: YOLO text format
**Encoding**: UTF-8
**Structure**: One line per bounding box
```
class_id x_center y_center width height
```

**Example** (`img_00000.txt`):
```
0 0.523456 0.412345 0.156789 0.234567
0 0.234567 0.678901 0.098765 0.123456
```

---

## 6. Quick Start Guide

### 🚀 Quick Overview

This project implements a two-stage pipeline:
1. **Dog Detection** (YOLOv8) - Find dogs in images
2. **Emotion Classification** (ResNet50) - Classify emotions: Angry, Happy, Relaxed, Frown, Alert

### 📋 Prerequisites

- Python 3.9 - 3.11 (⚠️ **Python 3.12+ not supported yet**)
- Git
- Datasets already downloaded and extracted to `data/raw/` directory

### 🔧 Installation

#### Option A: CPU Setup (Local Testing) ✅ Recommended for First-Time Users

Perfect for validating code logic before GPU training.

##### Step 1: Create Virtual Environment

```bash
cd CNN_A3
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

##### Step 2: Install PyTorch (CPU Version)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

##### Step 3: Install Other Dependencies

```bash
pip install -r requirements.txt
```

##### Step 4: Fix NumPy Compatibility

Some packages may upgrade NumPy to 2.x, which is incompatible. Force downgrade:

```bash
pip install 'numpy>=1.24.0,<2.0.0' --force-reinstall
```

✅ **Verify Installation:**
```bash
python test_setup.py
```

Expected output: `🎉 ALL TESTS PASSED!`

#### Option B: GPU Setup (AWS SageMaker) 🚀 For Production Training

Use this for full-scale model training with NVIDIA T4 GPU (16GB VRAM).

##### Step 1: Activate Conda Environment (SageMaker Default)

```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
```

##### Step 2: Install Dependencies

```bash
cd CNN_A3
pip install -r requirements.txt
```

##### Step 3: Fix NumPy Compatibility

Some packages may upgrade NumPy to 2.x, which is incompatible. Force downgrade:

```bash
pip install 'numpy>=1.24.0,<2.0.0' --force-reinstall
```

✅ **Verify GPU Availability:**
```python
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

Expected output: `CUDA Available: True`

### 📊 Dataset Setup

#### ⚠️ Important: Simplified Data Workflow

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


### 🧪 Test Your Setup

#### Quick Logic Validation (CPU)

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

### 🎯 Run Your First Experiment

#### Start with Classification Baseline (Simplest)

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

### 📂 Understanding Outputs

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

### 🔄 All Experiments

#### Detection Experiments (YOLOv8)

```bash
python experiments/exp01_detection_baseline.py      # Baseline YOLOv8
python experiments/exp02_detection_modified_v1.py   # Modified v1
python experiments/exp03_detection_modified_v2.py   # Modified v2
```

#### Classification Experiments (ResNet50)

```bash
python experiments/exp04_classification_baseline.py      # Baseline ResNet50
python experiments/exp05_classification_modified_v1.py   # Modified v1
python experiments/exp06_classification_modified_v2.py   # Modified v2
```

### Run All Experiments

```bash
bash scripts/run_all_experiments.sh
```

### 🔍 Inference Demo

Test trained models on your own images:

```bash
bash scripts/inference_demo.sh path/to/your/image.jpg
```

### ⚙️ Configuration

#### Modify Hyperparameters

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

Edit [`config.yaml`](config.yaml) for project-wide defaults.


---

## 7. Experiment Workflow

### 7.1 Six Experiments Overview

| Experiment | Task | Model | Variant | Key Differences |
|------------|------|-------|---------|-----------------|
| **Exp01** | Detection | YOLOv8 | Baseline | Default params (backbone='m', size=640) |
| **Exp02** | Detection | YOLOv8 | Modified v1 | Larger model (backbone='l', size=1280) |
| **Exp03** | Detection | YOLOv8 | Modified v2 | Smaller model + custom anchors |
| **Exp04** | Classification | ResNet50 | Baseline | Default params (dropout=0.5, partial freeze) |
| **Exp05** | Classification | ResNet50 | Modified v1 | Higher dropout + additional FC layers |
| **Exp06** | Classification | ResNet50 | Modified v2 | Lower dropout + no freeze |


### 7.2 Running Experiments

#### Option 1: Run Single Experiment
```bash
cd experiments
python exp01_detection_baseline.py
```

#### Option 2: Run All Experiments Sequentially
```bash
bash scripts/run_all_experiments.sh
```

#### Option 3: Run with Custom Parameters
```bash
python exp01_detection_baseline.py --lr 0.001 --batch_size 32 --epochs 100
```

---

## 8. Output Organization

### 8.1 Simplified Output Structure

Each experiment has its own folder. Each run creates a timestamped sub-folder containing all outputs.

```
outputs/
├── exp01_detection_baseline/
│   ├── run_20260420_193045/       ← First run
│   │   ├── model/
│   │   │   ├── best_model.pt      ← Best model weights
│   │   │   └── model_config.json  ← Model configuration used
│   │   │
│   │   ├── logs/
│   │   │   ├── training_log.csv   ← Epoch-by-epoch metrics
│   │   │   └── experiment_report.md ← Full markdown report
│   │   │
│   │   └── figures/
│   │       ├── precision_recall_curve.png
│   │       ├── IoU_distribution.png
│   │       └── sample_detections.png
│   │
│   └── run_20260421_101523/       ← Second run (new timestamp)
│       └── ... (same structure)
│
├── exp02_detection_modified_v1/
│   └── run_TIMESTAMP/
│       ├── model/
│       ├── logs/
│       └── figures/
│
... (exp03-exp06 follow same pattern)
```

### 8.2 Output Contents

#### Model Folder
- `best_model.pt` / `best_model.pth`: Trained model weights
- `model_config.json`: JSON file with all hyperparameters used

#### Logs Folder
- `training_log.csv`: CSV with columns:
  ```
  epoch, train_loss, val_loss, train_metric, val_metric, learning_rate
  1, 2.345, 2.123, 0.45, 0.48, 0.001
  2, 1.987, 1.876, 0.52, 0.55, 0.001
  ...
  ```
- `experiment_report.md`: Comprehensive markdown report including:
  - Experiment configuration
  - Training summary
  - Evaluation metrics
  - Figure references
  - Hardware info
  - Execution time

#### Figures Folder
- **Detection experiments**:
  - `precision_recall_curve.png`: PR curve for each class
  - `IoU_distribution.png`: Histogram of IoU values
  - `sample_detections.png`: Example predictions on test images
  - `confusion_matrix.png`: If multi-class detection
  
- **Classification experiments**:
  - `confusion_matrix.png`: 5×5 confusion matrix
  - `roc_curve.png`: ROC curves (one-vs-rest for each class)
  - `per_class_metrics.png`: Bar chart of precision/recall/F1 per class
  - `training_curves.png`: Loss and accuracy over epochs

---

## 10. Web App Overview

### 🎯 Features

The web application provides a user-friendly interface for real-time dog emotion recognition with **three interaction modes**:

#### 📷 Mode 1: Upload Image
- ✅ **Image Upload**: Drag & drop or click to upload images
- ✅ **Instant Analysis**: Automatic detection upon upload
- ✅ **Visual Annotations**: Bounding boxes drawn directly on uploaded images
- ✅ **Detailed Results**: Confidence scores and emotion probabilities

#### 🎬 Mode 2: Upload Video (Enhanced in v3.1.0)
- ✅ **Video Upload**: Support for MP4, WebM, AVI files (max 20 seconds, 50MB)
- ✅ **Video Playback**: Native HTML5 video player with controls
- ✅ **Smooth Video Annotations**: **ENHANCED!** Bounding boxes smoothly follow dog movement using linear interpolation
- ✅ **Pre-processing Analysis**: Backend analyzes all frames upfront at 5fps (every 200ms)
- ✅ **Fluent Animation**: Real-time boundary box interpolation for buttery-smooth tracking
- ✅ **Progress Indicator**: Shows analysis progress during preprocessing phase

#### 📹 Mode 3: Live Stream
- ✅ **Camera Access**: Real-time webcam feed using getUserMedia API
- ✅ **Live Indicator**: Visual feedback showing active stream
- ✅ **Future Ready**: Framework for real-time emotion detection

### Core Capabilities

- ✅ **Dog Detection**: YOLOv8-based dog face detection with bounding boxes
- ✅ **Emotion Classification**: ResNet50-based emotion recognition (5 emotions)
- ✅ **Multi-Dog Support**: Detect and classify multiple dogs in a single image/frame
- ✅ **Beautiful UI**: Modern, responsive React interface with mode switching
- ✅ **CPU Compatible**: Works on CPU (no GPU required for inference)

### Supported Emotions

- 😊 **Happy**: Joyful, playful expression
- 😠 **Angry**: Aggressive, threatening posture
- 😌 **Relaxed**: Calm, peaceful state
- 😟 **Frown**: Sad, concerned look
- 👀 **Alert**: Attentive, watchful stance

---

## 11. Web App Quick Start

### ✅ System Status

All components are installed and tested successfully!

- ✅ Backend API (FastAPI) - Running on port 8000
- ✅ Frontend App (React + Vite) - Running on port 5173
- ✅ Models loaded successfully on CPU
- ✅ API documentation available

### Prerequisites

- Python 3.9+
- Node.js 16+ and npm
- PyTorch (CPU or GPU version)

### Installation

#### 1. Install Backend Dependencies

```bash
cd api_service

# First, install PyTorch (choose based on your hardware)
# For CPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For GPU (CUDA 11.8):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Ensure NumPy compatibility
pip install 'numpy>=1.24.0,<2.0.0' --force-reinstall
```

#### 2. Install Frontend Dependencies

```bash
cd web_intf
npm install
```

### Running the Application

#### Option A: One-Command Start (Recommended)

```bash
chmod +x start_web_app.sh
./start_web_app.sh
```

This script will automatically:
- Check model files exist
- Start backend API on port 8000
- Start frontend dev server on port 5173
- Handle cleanup when you press Ctrl+C

#### Option B: Manual Start (Two Terminals)

**Terminal 1 - Backend:**
```bash
cd api_service
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd web_intf
npm run dev
```

### Access Points

Once started, open these URLs in your browser:

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend App** | http://localhost:5173 | Main user interface |
| **Backend API** | http://localhost:8000 | API root endpoint |
| **API Docs** | http://localhost:8000/docs | Interactive Swagger UI |
| **Health Check** | http://localhost:8000/health | API status check |

### Stopping the Application

**If using start_web_app.sh:**
Press `Ctrl+C` in the terminal

**If running manually:**
```bash
# Stop backend
pkill -f "python main.py"

# Stop frontend
pkill -f "npm run dev"
```

---

## 12. Web App Architecture

### Tech Stack

**Frontend:**
- React 18.x
- Vite (build tool with hot reload)
- Axios (HTTP client)
- CSS Modules (styling)

**Backend:**
- FastAPI (async web framework)
- PyTorch (deep learning)
- Ultralytics YOLOv8 (detection)
- OpenCV & PIL (image processing)

### Architecture Diagram

```
┌─────────────────┐         ┌──────────────┐         ┌─────────────────┐
│  React Frontend │  HTTP   │ FastAPI      │  Python │ Model Pipeline  │
│  (Vite + Axios) │ ◄─────► │ Backend      │ ◄─────► │ (YOLO+ResNet)   │
│  localhost:5173 │  JSON   │ localhost:8000│        │                 │
└─────────────────┘         └──────────────┘         └─────────────────┘
```

### Directory Structure

```
CNN_A3/
├── api_service/              # Backend API service
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   └── README.md            # API documentation
│
├── web_intf/                # Frontend React app
│   ├── src/
│   │   ├── components/      # React components
│   │   │   ├── ImageUploader.jsx    # Upload & preview
│   │   │   ├── ResultsDisplay.jsx   # Results with canvas
│   │   │   └── *.css                # Component styles
│   │   ├── services/
│   │   │   └── api.js       # API client
│   │   ├── App.jsx          # Main app component
│   │   └── App.css          # Global styles
│   ├── package.json
│   └── vite.config.js
│
├── best_models/             # Trained models (shared)
│   ├── detection_YOLOv8_baseline.pt
│   └── emotion_ResNet50_baseline.pth
│
├── src/                     # Existing ML code (reused)
│   └── inference/
│       └── pipeline_inference.py
│
├── start_web_app.sh         # One-command startup script
└── test_web_app.py          # Automated test suite
```

---

## 13. Using the Web Application

### Step-by-Step Guide

#### 📷 Mode 1: Upload Image Mode

1. **Select Mode**: Click "📷 Upload Image" button in header (default mode)
2. **Upload an image**: 
   - Click the upload area, OR
   - Drag & drop an image file
   - **Detection starts automatically** - no button click needed!
3. **View results**: See annotated image with:
   - Colored bounding boxes around detected dogs
   - Emotion labels at top-left of each box
   - Dog ID tags at bottom-left
   - Detailed metrics cards below
4. **Analyze another image**: Simply click the upload area again or drag & drop a new image (previous results auto-clear)

#### 🎬 Mode 2: Upload Video Mode (NEW!)

1. **Select Mode**: Click "🎬 Upload Video" button in header
2. **Upload a video**:
   - Click the upload area, OR
   - Drag & drop a video file
   - Supported formats: MP4, WebM, AVI (max 50MB)
3. **Video Playback**: 
   - Video loads and plays automatically
   - Use play/pause controls to manage playback
4. **Automatic Analysis**:
   - System extracts frames every 3 seconds
   - Processing status indicator shows current state
   - Detection results appear in cards below video
5. **Switch Videos**: Click "🔄 Change Video" to upload a different video

#### 📹 Mode 3: Live Stream Mode

1. **Select Mode**: Click "📹 Live Stream" button in header
2. **Grant Camera Permission**: Browser will request camera access - click "Allow"
3. **View Live Feed**: 
   - Real-time camera feed displays
   - "LIVE" indicator shows stream is active
   - Future enhancement: Real-time emotion detection overlay

### Supported File Formats

**Images:**
- JPEG/JPG
- PNG
- Maximum size: 10MB
- Auto-resized to 640px max dimension for optimal performance

**Videos:**
- MP4 (recommended)
- WebM
- AVI
- Maximum size: 50MB

### Visual Annotations

When results are displayed, you'll see:

**On Images/Video Frames:**
- **Colored Bounding Boxes**: Each emotion has a unique color
  - 😊 Happy: Green (#4CAF50)
  - 😠 Angry: Red (#f44336)
  - 😌 Relaxed: Blue (#2196F3)
  - 😟 Frown: Orange (#FF9800)
  - 👀 Alert: Purple (#9C27B0)

- **Emotion Labels**: Smart positioning at top-left of each box
  - Shows emoji + emotion name + confidence %
  - Example: "😊 Happy (87.3%)"
  - **Auto-adjusts position**: If box is near image edge, label moves inside to stay visible

- **Dog ID Tags**: Smart positioning at bottom-left of each box
  - Shows "Dog #1", "Dog #2", etc.
  - **Auto-adjusts position**: If box is near bottom edge, label moves inside to stay visible

### UI Optimization

**Three-Mode Interface**:
- Clear mode buttons in header with active state highlighting
- Smooth transitions between modes
- Independent state management for each mode
- Automatic cleanup when switching modes

**Compact Upload Interface**:
- Upload area uses minimal vertical space (80px height)
- Horizontal layout with icon + text side-by-side
- File info shown inline (name + size)
- No duplicate rendering

**Frontend Image Resizing**:
- Images >640px automatically resized to 640px max dimension
- Aspect ratio preserved
- 90% JPEG quality for optimal size/quality balance
- Reduces inference time by ~60% on CPU

**Smart Label Positioning**:
- Labels automatically repositioned when near image edges
- Top label (emotion): Moves inside box if <30px space above
- Bottom label (dog ID): Moves inside box if near canvas bottom
- Ensures 100% label visibility in all scenarios

### Expected Results

For each detected dog, you'll see:
- **Bounding Box**: Location coordinates [x1, y1, x2, y2]
- **Detection Confidence**: How sure the model is about the detection
- **Emotion Label**: Predicted emotion with confidence
- **Probability Distribution**: Breakdown across all 5 emotion classes

---

## 14. API Documentation

### POST /api/detect

Upload an image to detect dog faces and classify emotions.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@dog_image.jpg"
```

**Response:**
```json
{
  "success": true,
  "results": [
    {
      "dog_id": 0,
      "bbox": [100.5, 150.2, 300.8, 400.6],
      "detection_confidence": 0.95,
      "emotion": "happy",
      "emotion_confidence": 0.87,
      "emotion_probabilities": {
        "angry": 0.02,
        "happy": 0.87,
        "relaxed": 0.05,
        "frown": 0.03,
        "alert": 0.03
      }
    }
  ],
  "message": "Detected 1 dog(s)"
}
```

### GET /health

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "device": "CPU"
}
```

### Interactive API Docs

Visit http://localhost:8000/docs for Swagger UI with:
- All available endpoints
- Request/response schemas
- Try-it-out functionality
- Authentication options (if added later)

---

## 15. Web App Troubleshooting

### Backend Issues

**Problem**: Models not loading
```
Solution: Verify model files exist in best_models/
- detection_YOLOv8_baseline.pt (~50 MB)
- emotion_ResNet50_baseline.pth (~98 MB)
```

**Problem**: Port 8000 already in use
```bash
# Find and kill process using port 8000
lsof -ti:8000 | xargs kill
```

**Problem**: Import errors
```bash
cd api_service
pip install -r requirements.txt
```

### Frontend Issues

**Problem**: Cannot connect to API
```
Solution: 
1. Check if backend is running: curl http://localhost:8000/health
2. Verify CORS settings in api_service/main.py
3. Check browser console for error messages
```

**Problem**: npm install fails
```bash
# Clear npm cache
npm cache clean --force
# Remove node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Performance Issues

**Problem**: Slow inference on CPU

Expected inference time on CPU: ~500ms - 1s per image

Solutions:
1. Use smaller images (< 2MB recommended)
2. Close other applications to free CPU resources
3. Consider using GPU for production deployment

### Testing

Run the automated test suite:

```bash
python test_web_app.py
```

This checks:
- Model files existence
- Backend API health
- Frontend accessibility
- API documentation availability

---

## 16. Performance & Deployment

### Performance Metrics

| Metric | CPU (Mac M1) | GPU (T4) |
|--------|--------------|----------|
| Single Image Inference | ~500ms | ~100ms |
| Memory Usage (Backend) | ~2GB | ~4GB |
| Memory Usage (Frontend) | ~100MB | ~100MB |
| Max Concurrent Users | ~5 | ~50 |

### Configuration Options

#### Backend Configuration

Edit `api_service/main.py`:

```python
# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Add your frontend URL
    ...
)

# Inference parameters
results = pipeline.predict(temp_path, conf=0.5, iou=0.45)
# conf: Detection confidence threshold (0.0-1.0)
# iou: NMS IoU threshold (0.0-1.0)
```

#### Frontend Configuration

Edit `web_intf/src/services/api.js`:

```javascript
const API_BASE_URL = 'http://localhost:8000';  // Backend URL
```

### Future Enhancements

- [x] ~~Real-time webcam support~~ ✅ **COMPLETED in v2.0.0** - Live camera with frame capture
- [x] ~~Video upload and analysis~~ ✅ **COMPLETED in v3.0.0** - Video file upload with periodic frame processing
- [ ] Real-time WebSocket streaming for sub-second latency
- [ ] Batch processing for multiple images
- [ ] Save detection history to database
- [ ] User authentication and accounts
- [ ] Export results as CSV/JSON
- [ ] Mobile-responsive improvements
- [ ] Model performance monitoring
- [ ] Docker containerization

### Production Deployment Recommendations

For production deployment, consider:

1. **GPU Server**: Deploy on GPU-enabled instance for better performance
2. **Docker**: Containerize both services for easy deployment
3. **Database**: Add PostgreSQL for result history
4. **Authentication**: Implement JWT-based user auth
5. **Rate Limiting**: Prevent API abuse
6. **Logging**: Structured logging with ELK stack
7. **Monitoring**: Prometheus + Grafana dashboards
8. **CDN**: Serve static assets via CDN
9. **HTTPS**: SSL/TLS certificates
10. **Load Balancing**: Nginx reverse proxy

---

## 17. Version History & Changelog

### v3.1.0 (2026-04-25) - Smooth Video Annotations with Preprocessing 🎬

#### ✨ New Features

**Preprocessing-Based Video Analysis**
- 🎯 **Batch Frame Analysis**: Backend analyzes all frames upfront (5fps, every 200ms)
- ⏱️ **20-Second Limit**: Videos limited to 20 seconds for optimal performance
- 📊 **Progress Indicator**: Real-time progress display during analysis phase
- 🎬 **100 Frames Max**: Maximum 100 frames analyzed (20s × 5fps)

**Smooth Boundary Box Animation**
- 🔄 **Linear Interpolation**: Boundary boxes smoothly follow dog movement between frames
- 🎨 **Fluent Tracking**: No more jumping boxes - smooth 60fps animation
- 📐 **Real-time Sync**: Annotations perfectly synced with video playback time
- 🎮 **Play/Pause Support**: Works seamlessly with video controls

**Enhanced User Experience**
- ✅ **No Video Blocking**: Analysis happens before playback, no interruptions
- ⚡ **Instant Playback**: Once analysis is complete, video plays smoothly
- 📈 **Analysis Summary**: Shows frame count and duration after processing
- ⚠️ **Error Handling**: Clear error messages for invalid videos or failed analysis

#### 🔧 Technical Implementation

**Backend Changes (`api_service/main.py`):**
- Added `/api/analyze-video` endpoint for full video preprocessing
- Uses OpenCV to extract frames at 5fps sampling rate
- Validates video duration (max 20 seconds)
- Returns all frame detections in single JSON response
- Proper temporary file cleanup for memory efficiency

**Frontend Changes (`VideoResultsDisplay.jsx`):**
- Complete rewrite with preprocessing workflow
- Implements linear interpolation algorithm for smooth bbox animation
- Uses `timeupdate` event to sync annotations with video playback
- Calculates interpolation factor between adjacent frames
- Handles edge cases (missing detections, video boundaries)

**Interpolation Algorithm:**
```javascript
// Find frames before and after current time
const frameBefore = frames[currentFrameIndex];
const frameAfter = frames[currentFrameIndex + 1];

// Calculate interpolation factor (0.0 to 1.0)
const t = (currentTime - frameTimeBefore) / (frameTimeAfter - frameTimeBefore);

// Interpolate bounding box coordinates
const interpolatedBbox = lerpBbox(bboxBefore, bboxAfter, t);
```

**API Changes (`api.js`):**
- Added `analyzeVideo()` function for video preprocessing
- 5-minute timeout for long analysis operations
- Proper error handling and progress reporting

#### 🐛 Bug Fixes

- **Removed Choppy Updates**: Eliminated 3-second periodic update approach
- **Fixed Jumping Boxes**: Boundary boxes no longer jump between positions
- **Improved Autoplay Handling**: Graceful handling of browser autoplay policies
- **Better Error Messages**: Clear feedback for video duration and format issues

#### 📚 Documentation Updates

- Updated Web App Overview section with smooth animation feature
- Enhanced Mode 2 description with preprocessing details
- Added technical implementation notes for interpolation algorithm
- Documented 20-second video limit and 5fps sampling rate

#### ⚡ Performance Characteristics

**CPU Environment:**
- Processing time: ~1-2 minutes for 20-second video
- Memory usage: ~500MB peak (temporary frame files)
- Network transfer: ~50-100KB JSON response

**User Experience:**
- Initial wait: 1-2 minutes (one-time preprocessing)
- Playback: Instant, smooth 60fps annotations
- CPU load: Backend only during preprocessing, minimal during playback

---

### v3.0.1 (2026-04-25) - Video Annotation Overlay Enhancement 🎨

####  New Features

**Live Video Annotations**
- 🎯 **On-Video Bounding Boxes**: Colored bounding boxes drawn directly on playing video
- 🏷️ **Emotion Labels Overlay**: Real-time emotion labels with emoji and confidence scores
- 🆔 **Dog ID Tags**: Dynamic dog identification labels on video
- 🎨 **Smart Positioning**: Labels automatically adjust to avoid video edges
- 🔄 **Auto-Refresh**: Annotations update every 3 seconds as new frames are analyzed

**Enhanced Video Experience**
- 📹 **Overlay Canvas**: Transparent canvas layer on top of video for annotations
- 🎬 **Responsive Sizing**: Annotations scale correctly with video player size
- 🖱️ **Non-Interactive**: Overlay doesn't block video controls (pointer-events: none)
- 📐 **Dynamic Resize**: Annotations redraw on window resize for correct positioning

#### 🔧 Technical Implementation

**Annotation Rendering:**
- Uses HTML5 Canvas API for high-performance 2D drawing
- Calculates scaling factors between video resolution and display size
- Handles coordinate transformation for accurate bounding box placement
- Implements smart label positioning to avoid canvas edges

**Color Coding:**
- 😊 Happy: Green (#4CAF50)
- 😠 Angry: Red (#f44336)
- 😌 Relaxed: Blue (#2196F3)
- 😟 Frown: Orange (#FF9800)
- 👀 Alert: Purple (#9C27B0)

**Performance Optimizations:**
- Efficient canvas clearing and redrawing
- Debounced resize handler
- Conditional rendering (only draws when results exist)
- Minimal DOM manipulation

#### 🐛 Bug Fixes

- **Autoplay Error**: Fixed `AbortError: The play() request was interrupted` by handling browser autoplay policy gracefully
- **Canvas Sizing**: Fixed annotation overlay to match video display dimensions
- **Coordinate Scaling**: Corrected bounding box positioning for videos with different display vs actual resolution

#### 📚 Documentation Updates

- Updated Web App Overview section with video annotation feature
- Enhanced Mode 2 description with live annotation details
- Added technical details about overlay canvas implementation

---

### v3.0.0 (2026-04-25) - Three-Mode Interface Enhancement 🎉

#### ✨ New Features

**Three-Mode Architecture**
- 📷 **Upload Image Mode**: Traditional single image analysis with immediate detection
- 🎬 **Upload Video Mode**: NEW! Upload video files, plays video and processes frames periodically
- 📹 **Live Stream Mode**: Real-time camera feed with periodic frame capture

**Video Upload & Analysis**
- 🎥 New `VideoUploader` component for video file selection
- 🎬 New `VideoResultsDisplay` component for video playback and results
- ⏱️ Automatic frame extraction every 3 seconds during video playback
- 🎮 Video controls (play/pause) with processing status indicator
- 📊 Real-time emotion detection results displayed alongside video with on-video overlay

**Enhanced UI/UX**
- 🔘 Three mode buttons in header for easy switching
- 🎨 Active mode highlighting with visual feedback
- 🔄 Clean state management when switching between modes
- 📱 Responsive design for all screen sizes

#### 🎨 Component Updates

**New Components Created:**
- `web_intf/src/components/VideoUploader.jsx` - Video file upload interface
- `web_intf/src/components/VideoUploader.css` - Video uploader styles
- `web_intf/src/components/VideoResultsDisplay.jsx` - Video playback and results display
- `web_intf/src/components/VideoResultsDisplay.css` - Video results styles
- `web_intf/src/components/LiveStream.jsx` - Live camera stream component
- `web_intf/src/components/LiveStream.css` - Live stream styles

**Modified Components:**
- `web_intf/src/App.jsx` - Refactored to support three-mode architecture
- `web_intf/src/App.css` - Updated styles for mode buttons and layout
- `web_intf/src/components/ResultsDisplay.jsx` - Removed legacy live stream code (now in dedicated component)

**Backend Updates:**
- `api_service/main.py` - Added `/api/detect-base64` endpoint for base64 image processing
- Added `Base64ImageRequest` Pydantic model
- Implemented `detect_emotion_base64` function for video frame analysis

**API Service Updates:**
- `web_intf/src/services/api.js` - Added `detectEmotionFromBase64` function

#### 📚 Technical Implementation

**Video Processing Strategy:**
- HTML5 video element for native playback
- Canvas API for frame extraction
- Base64 encoding for backend transmission
- Periodic processing interval (3 seconds) balances performance and real-time feedback

**State Management:**
- Centralized mode state in App component
- Independent state for each mode (image results, video file, live stream)
- Automatic cleanup when switching modes

**Backend Integration:**
- Image mode: Uses `/api/detect` endpoint (multipart form data)
- Video mode: Uses `/api/detect-base64` endpoint (JSON with base64 image)
- Both endpoints return identical response structure

#### 🐛 Bug Fixes

- Fixed mode switching state conflicts
- Improved error handling for camera access
- Better video file validation (type and size checks)
- Enhanced responsive layout for mobile devices
- Graceful handling of browser autoplay policy

#### 🔧 Migration Guide

**For Users:**
- Simply refresh the page after updating
- All three modes are accessible via header buttons
- Video upload supports MP4, WebM, AVI formats (max 50MB)
- Video annotations appear directly on playing video

**For Developers:**
- Check new component structure in `web_intf/src/components/`
- Review mode switching logic in `App.jsx`
- Video frame processing connects to `/api/detect-base64` backend endpoint
- Backend must be running for video analysis to work
---

### v2.0.0 (Previous Release)

- Basic web application with static image upload
- Dog face detection using YOLOv8
- Emotion classification using ResNet50
- FastAPI backend with REST endpoints
- React frontend with Vite
- Bounding box visualization on canvas
- Multi-dog support
- CPU-compatible inference

---

## 18. Development Notes

### Adding New Features

1. **Backend**: Add endpoints in `api_service/main.py`
2. **Frontend**: Create components in `web_intf/src/components/`
3. **API Client**: Update `web_intf/src/services/api.js`

### Code Style

- **Python**: Follow PEP 8, use type hints
- **JavaScript**: ES6+, functional components with hooks
- **CSS**: Modular CSS with component-scoped styles

---

## Credits

Built with:
- YOLOv8 by Ultralytics
- ResNet50 from torchvision
- FastAPI framework
- React ecosystem

---

**Happy Coding! 🐕✨**
