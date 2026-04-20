# Visual Dog Emotion Recognition - CNN Pipeline

A two-stage deep learning pipeline for detecting dog faces and classifying their emotions using YOLOv8 and ResNet50.

## 🚀 Getting Started

**New to this project? Start here!** 👇

➡️ **[QUICKSTART.md](QUICKSTART.md)** - Complete setup guide with CPU & GPU instructions (10 minutes)

This is your **single source of truth** for:
- ✅ Installation (CPU vs GPU)
- ✅ Dataset setup
- ✅ Running experiments
- ✅ Troubleshooting common issues

---

## 📋 Project Overview

This project implements a complete computer vision pipeline that:
1. **Detects** dog faces in images using YOLOv8
2. **Classifies** emotions (angry, happy, relaxed, frown, alert) using ResNet50

### Architecture
```
Input Image → [YOLOv8 Detection] → Crop Faces → [ResNet50 Classification] → Emotion Labels
```

## 🗂️ Project Structure

```
CNN_A3/
├── data/                    # Datasets (auto-created)
│   ├── raw/                # Original downloaded data
│   └── processed/          # Preprocessed splits
├── src/                    # Source code
│   ├── data_processing/    # Data download & preprocessing
│   ├── models/             # Model definitions
│   ├── training/           # Training frameworks
│   ├── evaluation/         # Evaluation metrics
│   ├── inference/          # Inference pipelines
│   └── utils/              # Utility functions
├── experiments/            # 6 experiment scripts
│   ├── exp01_detection_baseline.py
│   ├── exp02_detection_modified_v1.py
│   ├── exp03_detection_modified_v2.py
│   ├── exp04_classification_baseline.py
│   ├── exp05_classification_modified_v1.py
│   └── exp06_classification_modified_v2.py
├── outputs/                # Experiment results (auto-created)
├── config.yaml             # Global configuration
└── requirements.txt        # Dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd CNN_A3

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Kaggle API

To download datasets, you need Kaggle API credentials:

```bash
# Install kaggle CLI
pip install kaggle

# Create .kaggle directory
mkdir -p ~/.kaggle

# Place your kaggle.json file in ~/.kaggle/
# Download from: https://www.kaggle.com/<username>/account

# Set permissions
chmod 600 ~/.kaggle/kaggle.json
```

### 3. Download and Preprocess Data

Data is automatically downloaded and preprocessed when you run experiments. You can also do it manually:

```bash
# Download datasets
python src/data_processing/download_datasets.py

# Preprocess detection data
python -c "from src.data_processing.detection_preprocessor import DetectionPreprocessor; DetectionPreprocessor().process()"

# Preprocess emotion data
python -c "from src.data_processing.emotion_preprocessor import EmotionPreprocessor; EmotionPreprocessor().process()"
```

### 4. Run Experiments

#### Run a Single Experiment

```bash
# Detection baseline
python experiments/exp01_detection_baseline.py

# Classification baseline
python experiments/exp04_classification_baseline.py
```

#### Run All Experiments

```bash
bash scripts/run_all_experiments.sh
```

### 5. View Results

After running experiments, check the outputs:

```bash
# List experiment results
ls outputs/exp01_detection_baseline/

# View latest run
ls outputs/exp01_detection_baseline/run_*/

# Read experiment report
cat outputs/exp01_detection_baseline/run_*/logs/experiment_report.md
```

## 🧪 Experiments

### Detection Experiments (YOLOv8)

| Experiment | Model | Backbone | Input Size | Key Features |
|------------|-------|----------|------------|--------------|
| Exp01 | Baseline | Medium (m) | 640×640 | Standard configuration |
| Exp02 | Modified V1 | Large (l) | 1280×1280 | Higher accuracy, slower |
| Exp03 | Modified V2 | Small (s) | 640×640 | Faster inference |

### Classification Experiments (ResNet50)

| Experiment | Model | Dropout | FC Layers | Freeze Strategy |
|------------|-------|---------|-----------|-----------------|
| Exp04 | Baseline | 0.5 | No | Partial freeze |
| Exp05 | Modified V1 | 0.7 | Yes (2 layers) | Partial freeze |
| Exp06 | Modified V2 | 0.3 | No | No freeze |

## 🔧 Configuration

Edit `config.yaml` to customize:
- Dataset paths and parameters
- Model architectures
- Training hyperparameters
- Hardware settings

## 📊 Output Structure

Each experiment creates a timestamped output folder:

```
outputs/exp01_detection_baseline/run_20260420_193045/
├── model/
│   ├── best_model.pt
│   └── model_config.json
├── logs/
│   ├── training_log.csv
│   └── experiment_report.md
└── figures/
    ├── precision_recall_curve.png
    └── sample_detections.png
```

## 🎯 Inference

### Detection Only

```python
from src.inference.detection_inference import DetectionInference

detector = DetectionInference('outputs/exp01_detection_baseline/run_*/model/best_model.pt')
results = detector.predict('test_image.jpg')
```

### Classification Only

```python
from src.inference.classification_inference import ClassificationInference

classifier = ClassificationInference('outputs/exp04_classification_baseline/run_*/model/best_model.pth')
result = classifier.predict('cropped_face.jpg')
```

### End-to-End Pipeline

```python
from src.inference.pipeline_inference import PipelineInference

pipeline = PipelineInference(
    detection_model_path='outputs/exp01_detection_baseline/run_*/model/best_model.pt',
    classification_model_path='outputs/exp04_classification_baseline/run_*/model/best_model.pth'
)

results = pipeline.predict('test_image.jpg')
pipeline.visualize('test_image.jpg', 'output.jpg')
```

## 💡 Key Features

- ✅ **Modular Design**: Separate modules for data, models, training, evaluation
- ✅ **Configuration-Driven**: Easy to modify hyperparameters via config files
- ✅ **Reproducible**: Timestamped runs with complete logging
- ✅ **Memory Efficient**: Mixed precision training, gradient accumulation
- ✅ **Experiment Tracking**: Automatic metric logging and report generation

## 🛠️ Troubleshooting

### Out of Memory (OOM)

Reduce batch size or enable gradient accumulation in experiment scripts:

```python
training_config = {
    'batch_size': 8,  # Reduce from 16 or 32
    'gradient_accumulation_steps': 2,  # Accumulate gradients
    'use_amp': True,  # Enable mixed precision
}
```

### Slow Training

- Ensure GPU is available: `torch.cuda.is_available()`
- Use smaller model variants (backbone='s' instead of 'm')
- Reduce number of epochs for testing

### Missing Dependencies

```bash
pip install -r requirements.txt
```

## 📝 License

This project is for educational purposes as part of Assignment 3.

## 👥 Authors

CNN_A3 Project Team

## 📚 References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Dog Face Detection Dataset](https://www.kaggle.com/datasets/jessicali9530/dog-face-detection)
- [Dog Emotion Dataset](https://www.kaggle.com/datasets/tongpython/dog-emotions-5-classes)
