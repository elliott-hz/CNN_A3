# Visual Dog Emotion Recognition - CNN Pipeline

A two-stage deep learning pipeline for detecting dog faces and classifying their emotions using YOLOv8 and ResNet50.

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
├── data/                    # Datasets
│   ├── raw/                # Raw datasets (must be downloaded manually)
│   │   ├── detection_dataset/   # Dog face detection images & labels
│   │   └── emotion_dataset/     # Dog emotion classification images
│   └── processed/          # Preprocessed numpy arrays (auto-created)
├── src/                    # Source code
│   ├── data_processing/    # Data preprocessing & verification
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
├── scripts/                # Automation scripts
│   ├── run_data_preprocessing.sh  # Run data preprocessing
│   ├── run_all_experiments.sh     # Run all experiments
│   └── inference_demo.sh          # Inference demonstration
├── config.yaml             # Global configuration
└── requirements.txt        # Dependencies
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

## 🎯 Inference API

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
- ✅ **Simplified Data Workflow**: Manual download + automated preprocessing pipeline

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install PyTorch first (choose CPU or GPU version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu  # CPU
# OR
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # GPU

# Install other dependencies
pip install -r requirements.txt
```

### 2. Prepare Datasets

Download datasets from Kaggle and extract to `data/raw/`:
- Detection dataset → `data/raw/detection_dataset/`
- Emotion dataset → `data/raw/emotion_dataset/`

Then run preprocessing:
```bash
bash scripts/run_data_preprocessing.sh
```

### 3. Run Experiments

```bash
# Run single experiment
python experiments/exp04_classification_baseline.py

# Or run all experiments
bash scripts/run_all_experiments.sh
```

For detailed instructions, see [QUICKSTART.md](QUICKSTART.md).

## 💡 Key Features

- ✅ **Modular Design**: Separate modules for data, models, training, evaluation
- ✅ **Configuration-Driven**: Easy to modify hyperparameters via config files
- ✅ **Reproducible**: Timestamped runs with complete logging
- ✅ **Memory Efficient**: Mixed precision training, gradient accumulation
- ✅ **Experiment Tracking**: Automatic metric logging and report generation

## 📝 License

This project is for educational purposes as part of Assignment 3.

## 👥 Authors

CNN_A3 Project Team

## 📚 References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Dog Face Detection Dataset](https://www.kaggle.com/datasets/jessicali9530/dog-face-detection)
- [Dog Emotion Dataset](https://www.kaggle.com/datasets/tongpython/dog-emotions-5-classes)
