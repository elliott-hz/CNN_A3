# Visual Dog Emotion Recognition System

A complete two-stage deep learning pipeline for automatic dog emotion recognition from images and videos.

## 🎯 Project Overview

This system implements a **two-stage computer vision pipeline**:

```
Input Image/Video
       ↓
┌─────────────────────┐
│  Dog Face Detection  │ ← YOLOv8 (bounding box localization)
└─────────────────────┘
       ↓ (crop face)
┌─────────────────────┐
│ Emotion Classification│ ← ResNet50/AlexNet/GoogLeNet
│  (5 emotions)        │
└─────────────────────┘
       ↓
Output: BBox + Emotion Label
```

### Supported Emotions
- 😊 **Happy**: Joyful, playful expression
- 😠 **Angry**: Aggressive, threatening posture  
- 😌 **Relaxed**: Calm, peaceful state
- 😟 **Frown**: Sad, concerned look
- 👀 **Alert**: Attentive, watchful stance

### Key Features
- ✅ **Multi-model support**: Compare YOLOv8 variants and CNN architectures
- ✅ **Web interface**: React + FastAPI for real-time inference
- ✅ **Three interaction modes**: Image upload, video analysis, live stream
- ✅ **CPU/GPU compatible**: Works on both CPU (testing) and GPU (production)
- ✅ **Modular design**: Easy to extend with new models or experiments

---

## 📚 Documentation Structure

This project is documented across 4 files for clarity:

| Document | Purpose | Content |
|----------|---------|---------|
| **[README.md](README.md)** | Core overview | Architecture, tech stack, quick start |
| **[DATA_PREPROCESSING.md](DATA_PREPROCESSING.md)** | Data preparation | Dataset sources, preprocessing workflow, data format |
| **[MODEL_TRAINING.md](MODEL_TRAINING.md)** | Model training | Experiment configs, training strategies, evaluation metrics |
| **[MODEL_APPLICATION.md](MODEL_APPLICATION.md)** | Deployment & inference | Web app architecture, API docs, deployment guide |

**For detailed information on any topic, please refer to the corresponding document.**

---

## 🏗️ System Architecture

### Technical Stack

**Core Framework:**
- PyTorch 2.0+ with torchvision
- Ultralytics YOLOv8 (detection)
- OpenCV & PIL (image processing)

**Classification Models:**
- ResNet50 (~25.6M params) - Modern residual network
- AlexNet (~60M params) - Classic CNN architecture
- GoogLeNet (~7M params) - Efficient inception modules

**Web Application:**
- Frontend: React 18 + Vite + Axios
- Backend: FastAPI (async web framework)
- Communication: REST API with JSON responses

**Hardware Support:**
- Development: CPU environment for logic validation
- Production: NVIDIA GPU (T4 16GB VRAM tested)

### Directory Structure

```
CNN_A3/
├── README.md                    # This file (core overview)
├── DATA_PREPROCESSING.md        # Data preparation guide
├── MODEL_TRAINING.md            # Training configuration guide
├── MODEL_APPLICATION.md         # Web app & deployment guide
├── config.yaml                  # Global configuration
├── requirements.txt             # Python dependencies
│
├── data/                        # Datasets (auto-created)
│   ├── raw/                     # Original downloaded datasets
│   └── processed/               # Preprocessed splits (JSON metadata)
│
├── src/                         # Source code package
│   ├── data_processing/         # Dataset parsing utilities
│   ├── models/                  # Model definitions (YOLOv8, ResNet50, etc.)
│   ├── training/                # Training frameworks
│   ├── evaluation/              # Metrics calculation
│   ├── inference/               # Inference pipeline
│   └── utils/                   # Helper functions
│
├── experiments/                 # 6 experiment scripts
│   ├── exp01_detection_*.py     # YOLOv8 detection variants (3)
│   ├── exp04_classification_ResNet50_baseline.py # ResNet50 baseline
│   ├── exp05_classification_AlexNet.py
│   └── exp06_classification_GoogLeNet.py
│
├── outputs/                     # Experiment results (timestamped)
│   └── <experiment_name>/
│       └── run_TIMESTAMP/
│           ├── model/           # Saved weights
│           ├── logs/            # Training metrics
│           └── figures/         # Visualizations
│
├── api_service/                 # FastAPI backend
│   └── main.py                  # API endpoints
│
├── web_intf/                    # React frontend
│   └── src/
│       ├── components/          # UI components
│       └── services/            # API client
│
└── scripts/                     # Automation scripts
    ├── run_data_preprocessing.sh
    ├── run_all_experiments.sh
    └── inference_demo.sh
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 - 3.11 (⚠️ Python 3.12+ not supported)
- Git
- Node.js 16+ and npm (for web app)

### Installation

#### Step 1: Clone Repository
```bash
git clone <repository-url>
cd CNN_A3
```

#### Step 2: Install Dependencies

**Option A: CPU Setup (Local Testing)**
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch (CPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt

# Fix NumPy compatibility
pip install 'numpy>=1.24.0,<2.0.0' --force-reinstall
```

**Option B: GPU Setup (Production Training)**
```bash
# Install PyTorch (CUDA 11.8)
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
  --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
pip install 'numpy>=1.24.0,<2.0.0' --force-reinstall
```

#### Step 3: Verify Installation
```bash
python test_setup.py
```
Expected output: `🎉 ALL TESTS PASSED!`

### Dataset Preparation

For detailed dataset setup instructions, see **[DATA_PREPROCESSING.md](DATA_PREPROCESSING.md)**.

**Quick summary:**
1. Download datasets manually from Kaggle
2. Place in `data/raw/` directory
3. Run preprocessing script:
```bash
bash scripts/run_data_preprocessing.sh
```

### Running Experiments

For detailed training configurations, see **[MODEL_TRAINING.md](MODEL_TRAINING.md)**.

**Quick examples:**
```bash
# Classification baseline (simplest)
python experiments/exp04_classification_ResNet50_baseline.py

# Detection baseline
python experiments/exp01_detection_YOLOv8_baseline.py

# Run all experiments
bash scripts/run_all_experiments.sh
```

### Launching Web Application

For complete web app documentation, see **[MODEL_APPLICATION.md](MODEL_APPLICATION.md)**.

**Quick start:**
```bash
# One-command startup
chmod +x start_web_app.sh
./start_web_app.sh

# Access points:
# Frontend: http://localhost:5173
# API Docs: http://localhost:8000/docs
```

---

## 📊 Model Performance Summary

### Detection Models (YOLOv8)

| Experiment | Backbone | Input Size | Parameters | Best For |
|------------|----------|------------|------------|----------|
| Baseline | Medium (m) | 640px | ~25M | Balanced speed/accuracy |
| V1 | Large (l) | 1280px | ~43M | Maximum accuracy |
| V2 | Small (s) | 640px | ~11M | Fast inference |

### Classification Models

| Experiment | Architecture | Parameters | Era | Key Feature |
|------------|--------------|------------|-----|-------------|
| Exp04: ResNet50 | ResNet50 | ~25.6M | 2015 | Modern residual network (configurable) |
| Exp05: AlexNet | AlexNet | ~60M | 2012 | Classic CNN architecture |
| Exp06: GoogLeNet | GoogLeNet | ~7M | 2014 | Efficient inception modules |

**Note**: ResNet50 baseline can be configured with different parameters for various training strategies. See [MODEL_TRAINING.md](MODEL_TRAINING.md) for detailed comparison.

---

## 🔗 Related Documentation

- **Data Processing**: [DATA_PREPROCESSING.md](DATA_PREPROCESSING.md)
  - Dataset sources and structure
  - Preprocessing workflow
  - Data format specifications
  
- **Model Training**: [MODEL_TRAINING.md](MODEL_TRAINING.md)
  - Experiment configurations
  - Training strategies
  - Evaluation metrics
  
- **Web Application**: [MODEL_APPLICATION.md](MODEL_APPLICATION.md)
  - Web app architecture
  - API documentation
  - Deployment guide

---

## 💡 Key Design Principles

1. **Simplified Architecture**: One base model class per task with configurable parameters
2. **Reproducible Research**: Fixed random seeds, detailed logging, automatic config saving
3. **Resource Efficiency**: Mixed precision training, optimized batch sizes, two-stage training
4. **Scalability**: Modular design, consistent API, separation of concerns
5. **Best Practices**: Early stopping, comprehensive metrics, visualization, proper data splits

---

## 📝 License & Credits

Built with:
- YOLOv8 by Ultralytics
- ResNet50, AlexNet, GoogLeNet from torchvision
- FastAPI framework
- React ecosystem

**Happy Coding! 🐕✨**