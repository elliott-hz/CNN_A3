# Detection Model Training Guide

This document provides comprehensive information about detection model architecture, experiment configurations, training strategies, and evaluation metrics for the Visual Dog Emotion Recognition system.

---

## 📊 Experiment Overview

The detection experiment uses YOLOv8 to detect dog faces in images.

| Experiment | Script | Backbone | Input Size | Key Feature |
|------------|--------|----------|------------|-------------|
| Exp01: Detection Baseline | `exp01_detection_YOLOv8_baseline.py` | Medium (m) | 640px | Balanced config |

**Dataset**: Dog Face Detection Dataset  
**Preprocessing**: Images resized to 640×640, labels in YOLO format (class x y w h)  
**Splits**: Train/Val/Test with stratification  
**Training**: Mixed precision (AMP), early stopping

---

## 🏗️ Detection Model Architecture

### YOLOv8Detector

**File**: [`src/models/detection_model.py`](src/models/detection_model.py)

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
```

**Key Features:**
- Unified wrapper around Ultralytics YOLOv8
- Configurable backbone depth for speed/accuracy trade-off
- Customizable inference thresholds
- Supports multi-dog detection

### Available Backbone Depths

| Depth | Parameters | Speed | Accuracy | Best For |
|-------|-----------|-------|----------|----------|
| **nano (n)** | ~3.2M | Fastest | Lower | Edge devices, real-time |
| **small (s)** | ~11.2M | Fast | Good | Mobile deployment |
| **medium (m)** | ~25.9M | Moderate | Better | Balanced performance |
| **large (l)** | ~43.7M | Slow | High | Server deployment |
| **xlarge (x)** | ~68.2M | Slowest | Highest | Maximum accuracy |

---

## ⚙️ Experiment Configuration

### Exp01: Detection Baseline

**Script**: [`experiments/exp01_detection_YOLOv8_baseline.py`](experiments/exp01_detection_YOLOv8_baseline.py)

**Model Configuration:**
```python
model_config = {
    'backbone': 'm',              # Medium backbone
    'input_size': 640,            # Standard resolution
    'confidence_threshold': 0.5,  # Moderate confidence
    'nms_iou_threshold': 0.45,    # Standard NMS
}
```

**Training Configuration:**
```python
training_config = {
    'epochs': 120,
    'optimizer': 'Adam',
    'learning_rate': 0.001,
    'batch_size': 16,
    'use_amp': True,              # Mixed precision
    'early_stopping_patience': 15,
}
```

**Purpose**: Establish baseline performance for dog face detection  
**Expected**: Balanced speed and accuracy

### Configuration Flexibility

You can easily modify the detection experiment by adjusting these parameters:

**Backbone Selection:**
- `'n'` (nano): Fastest, lowest accuracy (~3.2M params)
- `'s'` (small): Fast, good accuracy (~11.2M params)
- `'m'` (medium): Balanced (~25.9M params) ← **Current default**
- `'l'` (large): Slower, higher accuracy (~43.7M params)
- `'x'` (xlarge): Slowest, highest accuracy (~68.2M params)

**Input Size:**
- `640`: Standard resolution, balanced speed/accuracy
- `1280`: Higher resolution, better for small objects
- `320`: Lower resolution, faster inference

**Confidence Threshold:**
- `0.3`: Low threshold, more detections (higher recall)
- `0.5`: Moderate threshold, balanced ← **Current default**
- `0.7`: High threshold, fewer false positives (higher precision)

**NMS IoU Threshold:**
- `0.3`: Aggressive suppression, fewer overlapping boxes
- `0.45`: Moderate suppression ← **Current default**
- `0.6`: Lenient suppression, more overlapping boxes allowed

---

## 📈 Detection Training Strategies

### Built-in YOLOv8 Augmentations

YOLOv8 includes powerful built-in data augmentations:

1. **Mosaic Augmentation**: Combines 4 images into one, improves context understanding
2. **Mixup**: Blends two images and their labels
3. **Random Horizontal Flip**: Mirrors images horizontally
4. **HSV Color Space Augmentation**: Randomly adjusts hue, saturation, value
5. **Perspective Transform**: Applies random perspective warping
6. **Scale Augmentation**: Randomly scales images
7. **Translation**: Randomly shifts images

**Note**: These augmentations are automatically applied during training and don't require manual configuration in our code.

### Learning Rate Strategy

**Default Configuration:**
- Initial LR: 0.001
- Optimizer: Adam (adaptive learning rate)
- No explicit scheduler (YOLOv8 uses internal LR scheduling)

**Customization Options:**
```python
# For faster convergence
training_config = {
    'learning_rate': 0.01,        # Higher initial LR
    'optimizer': 'SGD',           # Traditional optimizer
    'momentum': 0.937,            # YOLOv8 default momentum
}

# For more stable training
training_config = {
    'learning_rate': 0.0001,      # Lower initial LR
    'optimizer': 'AdamW',         # Better weight decay handling
}
```

### Regularization Techniques

1. **Weight Decay**: Applied through optimizer (default 5e-4)
2. **Dropout**: Built into YOLOv8 architecture
3. **Data Augmentation**: Extensive built-in augmentations
4. **Early Stopping**: Patience 15 epochs to prevent overfitting
5. **Mixed Precision**: AMP for numerical stability

---

## 📊 Detection Evaluation Metrics

### Mean Average Precision (mAP)

**mAP@0.5**: Average Precision at IoU threshold = 0.5
- Measures detection accuracy with moderate overlap requirement
- Primary metric for most detection tasks

**mAP@0.5:0.95**: Average mAP across IoU thresholds from 0.5 to 0.95
- More stringent metric
- Evaluates localization precision

### Precision-Recall Curve

Shows the trade-off between:
- **Precision**: TP / (TP + FP) - How many detected dogs are correct?
- **Recall**: TP / (TP + FN) - How many actual dogs were detected?

### IoU Distribution

Histogram of Intersection-over-Union values for all predicted boxes:
- Shows how well predicted boxes align with ground truth
- Higher IoU indicates better localization

### Key Files:
- [`src/evaluation/detection_evaluator.py`](src/evaluation/detection_evaluator.py)

---

## 📂 Detection Output Organization

Each detection experiment saves outputs to timestamped directories:

```
outputs/exp01_detection_YOLOv8_baseline/
└── run_20260420_193045/
    ├── model/
    │   ├── best_model.pt        ← Best model weights
    │   └── model_config.json   ← Model configuration used
    │
    ├── logs/
    │   ├── training_log.csv    ← Epoch-by-epoch metrics
    │   └── experiment_report.md ← Full markdown report
    │
    └── figures/
        ├── precision_recall_curve.png
        ├── IoU_distribution.png
        └── sample_detections.png
```

### Output Contents

#### Model Folder
- `best_model.pt`: Trained model weights (best validation mAP)
- `model_config.json`: JSON file with all hyperparameters used

#### Logs Folder
- `training_log.csv`: CSV with columns:
  ```csv
  epoch, train_loss, val_loss, train_map, val_map, learning_rate
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
- `evaluation_metrics.json`: Test set performance metrics

#### Figures Folder
- `precision_recall_curve.png`: PR curve for each class
- `IoU_distribution.png`: Histogram of IoU values
- `sample_detections.png`: Example predictions on test images
- `confusion_matrix.png`: If multi-class detection

---

## 🚀 Running Detection Experiments

### Single Experiment

```bash
# Detection baseline
python experiments/exp01_detection_YOLOv8_baseline.py

# With custom parameters
python experiments/exp01_detection_YOLOv8_baseline.py --lr 0.001 --batch_size 16 --epochs 120
```

### Quick Testing with Small Subset

```bash
# Run with subset flag for quick validation
python experiments/exp01_detection_YOLOv8_baseline.py --use-small-subset
```

### Backbone Comparison

To compare different backbones:

```python
# In exp01_detection_YOLOv8_baseline.py
model_config = {
    'backbone': 's',  # Try 'n', 's', 'm', 'l', 'x'
    'input_size': 640,
    'confidence_threshold': 0.5,
    'nms_iou_threshold': 0.45,
}
```

---

## 💡 Detection Best Practices

### 1. Reproducibility

- Fixed random seeds in all experiments:
  ```python
  torch.manual_seed(42)
  np.random.seed(42)
  random.seed(42)
  ```
- Save complete configuration to `model_config.json`
- Log all hyperparameters and environmental info

### 2. Resource Efficiency

- Use mixed precision training (`use_amp=True`)
- Adjust batch size based on available GPU memory
- Choose appropriate backbone for your hardware:
  - CPU/Low-end GPU: Use 'n' or 's'
  - Mid-range GPU: Use 'm' ← **Recommended**
  - High-end GPU: Use 'l' or 'x'
- Monitor GPU memory usage: `nvidia-smi`

### 3. Debugging Tips

**Problem**: Out of Memory (OOM)
```
Solution:
- Reduce batch_size (e.g., 16 → 8)
- Use smaller backbone ('m' → 's')
- Reduce input_size (640 → 320)
- Enable AMP if not already enabled
```

**Problem**: Low mAP (< 0.5)
```
Solution:
- Check label quality (verify YOLO format)
- Increase training epochs
- Try larger backbone ('m' → 'l')
- Increase input_size (640 → 1280)
- Verify data augmentation is working
```

**Problem**: Too many false positives
```
Solution:
- Increase confidence_threshold (0.5 → 0.7)
- Decrease nms_iou_threshold (0.45 → 0.3)
- Train longer with more epochs
- Add more diverse training data
```

**Problem**: Missed detections (low recall)
```
Solution:
- Decrease confidence_threshold (0.5 → 0.3)
- Increase input_size for better small object detection
- Use larger backbone for better feature extraction
- Check if training data has sufficient examples
```

### 4. Monitoring Training

**Real-time monitoring:**
```bash
# Watch training log
tail -f outputs/<experiment>/run_<timestamp>/logs/training_log.csv

# Monitor GPU usage
watch -n 1 nvidia-smi
```

**Visualization:**
- Open generated figures in `figures/` directory
- Read comprehensive report in `experiment_report.md`

---

## 🔧 Advanced Configuration

### Custom Anchor Boxes

If detecting dogs at unusual scales, you can customize anchor boxes:

```python
model_config = {
    'backbone': 'm',
    'input_size': 640,
    'anchor_settings': {
        'small': [10, 13, 16, 30, 33, 23],
        'medium': [30, 61, 62, 45, 59, 119],
        'large': [116, 90, 156, 198, 373, 326]
    }
}
```

### Multi-Scale Training

Train on multiple input sizes for robustness:

```python
training_config = {
    'epochs': 120,
    'multi_scale': True,          # Enable multi-scale training
    'scale_range': [320, 640],    # Range of input sizes
}
```

### Transfer Learning

Fine-tune on domain-specific data:

```python
# Load pretrained COCO weights
model = YOLOv8Detector(backbone='m', pretrained=True)

# Freeze backbone initially
for param in model.backbone.parameters():
    param.requires_grad = False

# Train head only for 10 epochs
# Then unfreeze and fine-tune with lower LR
```

---

## 📈 Performance Expectations

Based on backbone selection:

| Backbone | Expected mAP@0.5 | Inference Speed (FPS) | Model Size | Best Use Case |
|----------|------------------|----------------------|------------|---------------|
| **nano (n)** | 0.60-0.70 | 100+ FPS | ~6 MB | Real-time mobile apps |
| **small (s)** | 0.70-0.80 | 50-80 FPS | ~22 MB | Mobile deployment |
| **medium (m)** | 0.80-0.85 | 20-40 FPS | ~52 MB | Balanced performance ← **Recommended** |
| **large (l)** | 0.85-0.90 | 10-20 FPS | ~88 MB | Server deployment |
| **xlarge (x)** | 0.90-0.95 | 5-10 FPS | ~140 MB | Maximum accuracy |

**Note**: Actual performance depends on dataset quality, training duration, and hardware capabilities.

---

## 🐛 Troubleshooting

### Common Issues

**Issue 1: Label Format Errors**

**Symptom**: Training fails with label parsing errors

**Solution**:
- Verify labels are in YOLO format: `class x_center y_center width height`
- All values should be normalized to [0, 1]
- Check for empty label files: `find . -name "*.txt" -empty`
- Validate coordinates don't exceed image boundaries

**Issue 2: Poor Convergence**

**Symptom**: Loss doesn't decrease after 20+ epochs

**Solution**:
- Check learning rate (try 0.001 → 0.01 or 0.0001)
- Verify data loading is correct
- Increase batch size if possible
- Check for label noise or incorrect annotations

**Issue 3: Overfitting**

**Symptom**: Train mAP >> Val mAP (gap > 0.15)

**Solution**:
- Enable stronger augmentations
- Increase early stopping patience
- Add dropout or weight decay
- Collect more diverse training data

---

**Last Updated**: 2026-04-26  
**Implemented by**: AI Assistant  
**Status**: ✅ Documentation created
