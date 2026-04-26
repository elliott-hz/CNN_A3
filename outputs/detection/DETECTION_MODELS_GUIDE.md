# Detection Model Comparison - Complete Guide

This document provides a comprehensive guide to the detection model comparison experiments, including design rationale, dataset configuration, and evaluation metrics.

---

## 📋 Experiment Overview

### Architecture Comparison

Three diverse detection architectures for comprehensive comparison:

| Experiment | Model | Architecture Type | Parameters | Key Feature |
|------------|-------|-------------------|------------|-------------|
| **Exp01** | YOLOv8 (Medium) | Single-stage (Anchor-based) | ~25.9M | Fast inference, balanced accuracy |
| **Exp02** | Faster R-CNN | Two-stage (Region Proposal) | ~41M | Higher accuracy, slower inference |
| **Exp03** | SSD (VGG16) | Single-stage (Multi-scale) | ~26M | Moderate speed, good small object detection |

### Why This Approach?

**Academic Rigor**: Comparing different detection paradigms provides insights into:
- Speed vs accuracy trade-offs
- Single-stage vs two-stage detection strategies
- Different feature extraction approaches

**Fair Comparison Requirements**:
- ✅ Same train/val/test split across all models
- ✅ Only annotation format changes (YOLO → COCO/VOC)
- ✅ All models trained ≥100 epochs
- ✅ Model-specific hyperparameter tuning allowed

---

## 🔄 Data Format Configuration

### Why Different Formats?

Different detection frameworks require different annotation formats:

| Framework | Format | File Type | Structure |
|-----------|--------|-----------|-----------|
| **YOLOv8** (Ultralytics) | YOLO | `.txt` | `class x_center y_center width height` (normalized) |
| **Faster R-CNN** (Torchvision) | COCO | `.json` | JSON with images, annotations, categories |
| **SSD** (Torchvision) | VOC | `.xml` | XML with bounding box coordinates |

### Conversion Script

**File**: [`src/data_processing/convert_detection_format.py`](src/data_processing/convert_detection_format.py)

```bash
# Convert to both COCO and VOC formats
python src/data_processing/convert_detection_format.py --format both
```

**Output Structure**:
```
data/processed/
├── detection/                    # Original YOLO format
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   └── dataset.yaml
│
├── detection_coco/               # COCO format (Faster R-CNN)
│   ├── images/{train,val,test}/
│   ├── annotations/
│   │   ├── instances_train.json
│   │   ├── instances_val.json
│   │   └── instances_test.json
│   └── dataset.yaml
│
└── detection_voc/                # VOC format (SSD)
    ├── images/{train,val,test}/
    ├── annotations/{train,val,test}/
    └── dataset.yaml
```

### Path Configuration

All three formats use **absolute paths** (relative to project root):

```yaml
# All dataset.yaml files use this format
path: data/processed/detection      # or detection_coco, detection_voc
train: images/train
val: images/val
test: images/test
```

**Benefits**:
- ✅ Consistency across all formats
- ✅ No need to change working directory
- ✅ Works from any location

### Important Note

**YOLOv8** requires `dataset.yaml` for training (Ultralytics API).  
**Faster R-CNN & SSD** use hardcoded paths in experiment scripts (don't read `dataset.yaml`).

The `dataset.yaml` files for COCO/VOC formats serve as documentation and reference only.

---

## 📊 Evaluation Metrics

### Core Metrics

#### 1. mAP (Mean Average Precision)

**mAP@0.5**: Average Precision at IoU threshold = 0.5
- Standard metric for most detection tasks
- Interpretation: >0.85 Excellent, 0.75-0.85 Good, 0.50-0.75 Moderate

**mAP@0.5:0.95**: Average of AP at IoU thresholds from 0.5 to 0.95 (step 0.05)
- More stringent metric (COCO standard)
- Rewards precise localization
- Typically 10-20% lower than mAP@0.5

#### 2. Precision, Recall, F1-Score

- **Precision**: TP / (TP + FP) - "How many detected objects are correct?"
- **Recall**: TP / (TP + FN) - "How many actual objects were detected?"
- **F1-Score**: Harmonic mean of precision and recall

#### 3. IoU (Intersection over Union)

**Formula**: `Area of Intersection / Area of Union`

**Statistics tracked**:
- Mean IoU: Average localization quality
- Median IoU: Robust central tendency
- Standard Deviation: Consistency of detections
- Distribution histogram: Full range visualization

### Visualization Outputs

The evaluator generates:
1. **IoU_distribution.png**: Histogram with mean/median markers
2. **mAP_vs_IoU.png**: mAP degradation across IoU thresholds
3. **per_class_metrics.png**: Bar chart of precision/recall/F1 per class

### Benchmark Expectations

| Metric | Poor | Moderate | Good | Excellent |
|--------|------|----------|------|-----------|
| **mAP@0.5** | < 0.60 | 0.60-0.75 | 0.75-0.85 | > 0.85 |
| **mAP@0.5:0.95** | < 0.35 | 0.35-0.50 | 0.50-0.65 | > 0.65 |
| **Precision** | < 0.70 | 0.70-0.80 | 0.80-0.90 | > 0.90 |
| **Recall** | < 0.65 | 0.65-0.75 | 0.75-0.85 | > 0.85 |

---

## 🔧 Implementation Details

### Model Configurations

#### Exp01: YOLOv8 Medium
```python
model_config = {
    'backbone': 'm',
    'input_size': 640,
    'confidence_threshold': 0.5,
    'nms_iou_threshold': 0.45,
}

training_config = {
    'learning_rate': 0.001,
    'batch_size': 24,
    'epochs': 120,
    'optimizer': 'adam',
    'weight_decay': 1e-4,
}
```

#### Exp02: Faster R-CNN
```python
model_config = {
    'architecture': 'faster_rcnn',
    'backbone': 'resnet50_fpn',
    'num_classes': 2,
    'pretrained': True
}

training_config = {
    'learning_rate': 0.005,           # SGD needs higher LR
    'batch_size': 4,                  # Small due to memory
    'epochs': 150,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'gradient_accumulation_steps': 4, # Effective batch = 16
}
```

#### Exp03: SSD
```python
model_config = {
    'architecture': 'ssd',
    'backbone': 'vgg16',
    'num_classes': 2,
    'pretrained': True
}

training_config = {
    'learning_rate': 0.001,
    'batch_size': 16,                 # Larger batch possible
    'epochs': 150,
    'optimizer': 'sgd',
    'momentum': 0.9,
    'weight_decay': 5e-4,
}
```

### Key Design Decisions

1. **Separate APIs**: Keep Ultralytics for YOLO, custom implementation for Torchvision
2. **Memory Optimization**: T4 GPU (10GB) constraints handled via batch size adjustment
3. **No Small Subset**: Removed `--use-small-subset` for simplicity
4. **Format Flexibility**: Demonstrates support for YOLO, COCO, and VOC formats

### Critical Bug Fixes

**Issue 1: Image Loading Error**
- Fixed: Changed from non-existent `F.pil_image_loader` to standard `Image.open()`

**Issue 2: Loss Format Compatibility**
- Problem: SSD returns multi-element tensors (800 elements), not scalars
- Solution: Use `.mean()` to reduce multi-element tensors before summing

**Issue 3: Torchvision Models Don't Return Loss in Eval Mode** ⚠️
- Discovery: Torchvision detection models return predictions (not losses) in eval mode
- Solution: Validation loop only checks for errors, returns dummy loss=0.0
- Early stopping still works based on training loss trend

**Issue 4: Invalid Bounding Boxes**
- Problem: COCO dataset contains zero-width/height boxes
- Solution: Filter invalid boxes during data loading

---

## 📈 Expected Performance

| Metric | YOLOv8 | Faster R-CNN | SSD |
|--------|--------|--------------|-----|
| **mAP@0.5** | 0.75-0.80 | 0.80-0.85 | 0.72-0.78 |
| **mAP@0.5:0.95** | 0.55-0.65 | 0.60-0.70 | 0.50-0.60 |
| **Inference Speed** | 40-60 FPS | 10-15 FPS | 25-35 FPS |
| **Training Time (T4)** | 2-3 hours | 4-5 hours | 3-4 hours |
| **Memory Usage** | Medium | High | Low |

---

## 🚀 Quick Start

### Run Experiments

```bash
# Exp01: YOLOv8
python experiments/exp01_detection_YOLOv8_baseline.py

# Exp02: Faster R-CNN
python experiments/exp02_detection_Faster-RCNN.py

# Exp03: SSD
python experiments/exp03_detection_SSD.py

# Resume training (YOLOv8 only)
python experiments/exp01_detection_YOLOv8_baseline.py --resume
```

### Output Organization

Each experiment saves results to timestamped directories:

```
outputs/exp01_detection_YOLOv8_baseline/
└── run_YYYYMMDD_HHMMSS/
    ├── model/          # Model weights and config
    ├── logs/           # Training logs and reports
    └── figures/        # Visualization plots
```

---

## 📚 Related Documentation

- **[DETECTION_TRAINING.md](../../DETECTION_TRAINING.md)** - Detailed training configurations and troubleshooting
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - Complete implementation details and bug fixes
- **[README.md](../../README.md)** - Project overview and quick start guide

---

**Last Updated**: 2026-04-26  
**Status**: ✅ All experiments ready to run
