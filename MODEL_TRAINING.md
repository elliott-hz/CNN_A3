# Model Training Guide

This document provides comprehensive information about model architectures, experiment configurations, training strategies, and evaluation metrics for the Visual Dog Emotion Recognition system.

---

## 🎯 Training Philosophy

The project follows a **configurable model design** philosophy:
- **One base model class per task** with configurable parameters
- **Multiple experiment scripts** to test different configurations
- **Avoid code duplication** by using configuration dictionaries
- **Reproducible research** with fixed random seeds and detailed logging

---

## 📊 Experiment Overview

The project includes **8 experiments** divided into two categories:

### Detection Experiments (Exp01)

The detection experiment uses YOLOv8 with a balanced configuration.

| Experiment | Script | Backbone | Input Size | Key Feature |
|------------|--------|----------|------------|-------------|
| Baseline | `exp01_detection_YOLOv8_baseline.py` | Medium (m) | 640px | Balanced config |

### Classification Experiments (Exp04-06)

Three classification experiments comparing different CNN architectures and training strategies.

| Experiment | Script | Architecture | Parameters | Era | Key Feature |
|------------|--------|--------------|------------|-----|-------------|
| Exp04: ResNet50 | `exp04_classification_ResNet50_baseline.py` | ResNet50 | ~25.6M | 2015 | Modern residual network with skip connections |
| Exp05: AlexNet | `exp05_classification_AlexNet.py` | AlexNet | ~60M | 2012 | Classic CNN with large FC layers |
| Exp06: GoogLeNet | `exp06_classification_GoogLeNet.py` | GoogLeNet | ~7M | 2014 | Efficient inception modules with auxiliary classifiers |

**Note**: The ResNet50 baseline experiment can be configured with different parameters (dropout rate, freeze strategy, optimizer, etc.) to explore various training approaches. See the configuration details below for recommended settings.

---

## 📊 Detailed Comparison of Classification Experiments

The following table provides a comprehensive side-by-side comparison of all 3 classification experiments, highlighting key differences in model architecture, training configuration, and optimization strategies.

### Complete Experiment Configuration Matrix

| **Configuration** | **Exp04: ResNet50** | **Exp05: AlexNet** | **Exp06: GoogLeNet** |
|-------------------|---------------------|--------------------|----------------------|
| **Model Architecture** | ResNet50 | AlexNet | GoogLeNet/Inception v1 |
| **Architecture Era** | 2015 (Modern) | 2012 (Classic) | 2014 (Efficient) |
| **Parameter Count** | ~25.6M | ~60M | ~7M |
| **Key Architectural Feature** | Skip connections | Large FC layers (4096 units) | Inception modules + Auxiliary classifiers |
| **Pretrained Weights** | ✅ ImageNet | ✅ ImageNet | ✅ ImageNet |
| **Freeze Backbone** | ✅ True | ✅ True | ✅ True |
| **Additional FC Layers** | ❌ False | N/A (3 FC built-in) | N/A (GAP instead) |
| **Auxiliary Classifiers** | N/A | N/A | ✅ Enabled (weight=0.3) |
| **Dropout Rate** | 0.5 | 0.5 | 0.5 |
| **Batch Normalization** | ✅ Yes | N/A | N/A |
| **Training Epochs** | 120 | 200 (Longer) | 120 |
| **Optimizer** | Adam | SGD + Momentum | Adam |
| **Learning Rate** | 0.0005 | 0.01 (High) | 0.001 |
| **Batch Size** | 32 | 64 (Larger) | 32 |
| **Weight Decay** | 5e-4 | 5e-4 (Stronger) | 1e-4 |
| **Gradient Accumulation** | 1 | 1 | 1 |
| **Label Smoothing** | 0.1 | 0.1 | 0.1 |
| **Class Weighting** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Mixed Precision (AMP)** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Early Stopping Patience** | 15 | 30 (Longer) | 15 |
| **LR Scheduler** | ReduceLROnPlateau | StepLR (decay=0.5, interval=40) | None specified |
| **Training Strategy** | Transfer learning (frozen) | Classic SGD approach | Multi-loss with auxiliary |
| **Best For** | Establishing strong baseline | Comparing classic vs modern | Efficiency-focused applications |
| **Memory Requirements** | Medium | High (large FC layers) | Low (most efficient) |
| **Expected Training Speed** | Fast | Slow (SGD, large batch) | Fast (lightweight) |
| **Expected Inference Speed** | Fast | Moderate | Very Fast |
| **Risk of Overfitting** | Low | Low (strong regularization) | Low (efficient design) |

### Key Design Decisions Explained

#### Exp04: ResNet50 (Modern Architecture with Transfer Learning)

**Design Philosophy:**
- Use modern residual network with skip connections to prevent vanishing gradients
- Freeze backbone to leverage pretrained ImageNet features (transfer learning)
- Train only classifier head with moderate learning rate for stable convergence
- Balanced dropout (0.5) and weight decay (5e-4) for regularization
- Adam optimizer for adaptive learning rates
- ReduceLROnPlateau scheduler to automatically adjust LR based on validation performance

**Configuration Flexibility:**
The ResNet50 baseline can be easily modified by adjusting these parameters in the experiment script:
- **Dropout rate**: Try 0.3-0.7 depending on overfitting observed
- **Freeze strategy**: Set `freeze_backbone=False` for full fine-tuning on domain-specific data
- **Learning rate**: Use lower LR (0.0001) if unfreezing backbone
- **Optimizer**: Switch to SGD with momentum for traditional training approach
- **Batch size**: Adjust based on available GPU memory (16-64 range)

**When to Modify:**
- If validation accuracy plateaus early → try unfreezing backbone with lower LR
- If overfitting occurs → increase dropout or add more augmentation
- If underfitting → decrease dropout, train longer, or unfreeze more layers

#### Exp04: ResNet50 (Modern Architecture with Transfer Learning)

**Design Philosophy:**
- Use modern residual network with skip connections to prevent vanishing gradients
- Freeze backbone to leverage pretrained ImageNet features (transfer learning)
- Train only classifier head with moderate learning rate for stable convergence
- Balanced dropout (0.5) and weight decay (5e-4) for regularization
- Adam optimizer for adaptive learning rates
- ReduceLROnPlateau scheduler to automatically adjust LR based on validation performance

**Configuration Flexibility:**
The ResNet50 baseline can be easily modified by adjusting these parameters in the experiment script:
- **Dropout rate**: Try 0.3-0.7 depending on overfitting observed
- **Freeze strategy**: Set `freeze_backbone=False` for full fine-tuning on domain-specific data
- **Learning rate**: Use lower LR (0.0001) if unfreezing backbone
- **Optimizer**: Switch to SGD with momentum for traditional training approach
- **Batch size**: Adjust based on available GPU memory (16-64 range)

**When to Modify:**
- If validation accuracy plateaus early → try unfreezing backbone with lower LR
- If overfitting occurs → increase dropout or add more augmentation
- If underfitting → decrease dropout, train longer, or unfreeze more layers

#### Exp05: AlexNet (Classic Architecture)

**Design Philosophy:**
- Test historical architecture against modern alternatives
- Use traditional SGD with momentum (as in original paper)
- Higher initial LR (0.01) typical for SGD training
- Larger batch size (64) possible due to simpler conv layers
- Stronger weight decay (5e-4) to regularize large FC layers
- Extended training (200 epochs) with step decay schedule
- Longer early stopping patience (30) to avoid premature termination

**Trade-offs:**
- ✅ Simple, well-understood architecture
- ❌ Large parameter count (~60M) due to dense FC layers
- ❌ No skip connections or batch normalization
- ❌ Slower inference compared to modern architectures

#### Exp06: GoogLeNet (Efficiency-Focused)

**Design Philosophy:**
- Maximize accuracy-to-parameter ratio
- Leverage Inception modules for multi-scale feature extraction
- Enable auxiliary classifiers for better gradient flow during training
- Use Adam optimizer (works well with complex architectures)
- Global average pooling eliminates massive FC layers
- Most parameter-efficient design (~7M vs ~25M for ResNet50)

**Unique Features:**
- Multi-loss training: main loss + 2× auxiliary losses (weight=0.3 each)
- Parallel convolutions (1×1, 3×3, 5×5) capture diverse patterns
- Dimensionality reduction via 1×1 convolutions before expensive operations
- Best choice for resource-constrained deployment scenarios

### When to Use Each Experiment

| **Scenario** | **Recommended Experiment** | **Rationale** |
|--------------|---------------------------|---------------|
| Quick baseline establishment | Exp04 ResNet50 | Proven modern architecture, balanced config |
| Need maximum accuracy | Exp04 ResNet50 (with fine-tuning) | Unfreeze backbone for domain adaptation |
| Limited GPU memory | Exp06 GoogLeNet | Only 7M parameters, very efficient |
| Fast inference required | Exp06 GoogLeNet | Lightweight, no heavy FC layers |
| Comparing old vs new architectures | Exp05 AlexNet | Historical reference point from 2012 |
| Domain shift present | Exp04 ResNet50 (unfrozen) | Full fine-tuning adapts better to new domain |
| Production deployment | Exp06 GoogLeNet | Best speed/accuracy/memory trade-off |
| Research/academic study | All 3 experiments | Comprehensive comparison across eras |
| Educational purposes | Start with Exp04, then Exp05, then Exp06 | Learn evolution of CNN architectures |

### Performance Expectations Summary

Based on architectural characteristics:

| **Metric** | **Likely Ranking (Best → Worst)** |
|------------|----------------------------------|
| **Accuracy** | Exp04 ResNet50 > Exp06 GoogLeNet > Exp05 AlexNet |
| **Training Speed** | Exp06 > Exp04 > Exp05 |
| **Inference Speed** | Exp06 > Exp04 > Exp05 |
| **Memory Efficiency** | Exp06 >> Exp04 > Exp05 |
| **Convergence Stability** | Exp04 > Exp06 > Exp05 |
| **Parameter Efficiency** | Exp06 (7M) >> Exp04 (25.6M) > Exp05 (60M) |

**Note**: Actual performance may vary based on dataset characteristics, training duration, and hyperparameter tuning. The ResNet50 model can achieve higher accuracy with full fine-tuning but requires more careful hyperparameter selection.

---

## 🏗️ Model Architectures

### Detection Model: YOLOv8Detector

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

### Classification Models

#### 1. ResNet50Classifier

**File**: [`src/models/classification_model.py`](src/models/classification_model.py)

```python
class ResNet50Classifier(nn.Module):
    """
    ResNet50 classifier with configurable parameters.
    
    Architecture: 50-layer residual network with skip connections
    Parameters: ~25.6M total
    
    Configuration options:
    - dropout_rate: 0.3, 0.5, 0.7
    - additional_fc_layers: True/False (adds extra FC layers)
    - freeze_strategy: 'all', 'partial', 'none'
    - num_classes: 5 (fixed for this task)
    - use_batch_norm: True/False
    - pretrained: Use ImageNet weights (default True)
    """
```

**Key Features:**
- Skip connections prevent vanishing gradient problem
- Deep architecture (50 layers) captures complex features
- Batch normalization for stable training
- Configurable freezing strategy for transfer learning

#### 2. AlexNetClassifier

```python
class AlexNetClassifier(nn.Module):
    """
    AlexNet classifier with configurable parameters.
    
    Architecture: 5 conv layers + 3 FC layers (classic CNN from 2012)
    Parameters: ~60M total (larger due to FC layers)
    
    Configuration options:
    - dropout_rate: 0.5 (default)
    - pretrained: Use ImageNet weights (default True)
    - freeze_backbone: Freeze feature extractor initially (default True)
    - num_classes: 5 (fixed for this task)
    """
```

**Key Features:**
- Simple, straightforward architecture
- Large fully connected layers (4096 units each)
- Historical significance (breakthrough in 2012 ImageNet competition)
- Larger parameter count due to dense FC layers

#### 3. GoogLeNetClassifier

```python
class GoogLeNetClassifier(nn.Module):
    """
    GoogLeNet (Inception v1) classifier with configurable parameters.
    
    Architecture: Inception modules with parallel convolutions
    Parameters: ~7M total (most efficient!)
    
    Configuration options:
    - dropout_rate: 0.5 (default)
    - pretrained: Use ImageNet weights (default True)
    - freeze_backbone: Freeze backbone initially (default True)
    - use_auxiliary: Use auxiliary classifiers during training (default True)
    - num_classes: 5 (fixed for this task)
    """
```

**Key Features:**
- Inception modules perform parallel convolutions (1×1, 3×3, 5×5)
- Auxiliary classifiers at intermediate layers improve gradient flow
- Global average pooling reduces parameters significantly
- Most parameter-efficient architecture (~7M vs ~25M for ResNet50)
- Multi-loss training with auxiliary classifier weights (0.3 each)

**Implementation Note:** Manually extracts features before final FC layer to avoid dimension mismatch (1024-dim features → custom classifier)

---

## ⚙️ Experiment Configurations

### Detection Experiments

#### Exp01: Detection Baseline

**Script**: [`experiments/exp01_detection_YOLOv8_baseline.py`](experiments/exp01_detection_YOLOv8_baseline.py)

**Configuration:**
```python
model_config = {
    'backbone': 'm',              # Medium backbone
    'input_size': 640,            # Standard resolution
    'confidence_threshold': 0.5,  # Moderate confidence
    'nms_iou_threshold': 0.45,    # Standard NMS
}

training_config = {
    'epochs': 120,
    'optimizer': 'Adam',
    'learning_rate': 0.001,
    'batch_size': 16,
    'use_amp': True,              # Mixed precision
}
```

**Purpose**: Establish baseline performance for dog face detection  
**Expected**: Balanced speed and accuracy

### Classification Experiments

All classification experiments use the **same dataset and preprocessing pipeline** for fair comparison:
- **Dataset**: Dog Emotion Dataset (~9,325 images, 5 classes)
- **Preprocessing**: Images resized to 224×224, normalized to [0,1]
- **Splits**: Train/Val/Test (70/20/10) with stratification
- **Training**: Mixed precision (AMP), early stopping patience varies

#### Exp04: ResNet50 Baseline

**Script**: [`experiments/exp04_classification_ResNet50_baseline.py`](experiments/exp04_classification_ResNet50_baseline.py)

**Configuration:**
```python
model_config = {
    'dropout_rate': 0.5,
    'additional_fc_layers': False,
    'freeze_backbone': True,      # Partial freezing
    'pretrained': True,
    'num_classes': 5,
}

training_config = {
    'epochs': 120,
    'learning_rate': 0.0005,
    'optimizer': 'Adam',
    'weight_decay': 5e-4,
    'batch_size': 32,
    'use_amp': True,
    'early_stopping_patience': 15,
}
```

**Purpose**: Establish strong baseline with modern architecture

**Configuration Flexibility**: This experiment can be easily modified to explore different training strategies:
- **Full fine-tuning**: Set `freeze_backbone=False` and use lower learning rate (0.0001) with SGD optimizer
- **Increased capacity**: Add extra FC layers by setting `additional_fc_layers=True` and increase dropout to 0.7
- **Different optimizers**: Try AdamW for better weight decay handling, or SGD with momentum for traditional approach
- **Extended training**: Increase epochs to 200+ if validation metrics haven't plateaued

#### Exp05: AlexNet

**Script**: [`experiments/exp05_classification_AlexNet.py`](experiments/exp05_classification_AlexNet.py)

**Configuration:**
```python
model_config = {
    'dropout_rate': 0.5,
    'freeze_backbone': True,
    'pretrained': True,
}

training_config = {
    'epochs': 200,                # More epochs
    'learning_rate': 0.01,        # Higher initial LR
    'lr_scheduler': 'step',       # Step decay
    'lr_decay_factor': 0.5,
    'lr_decay_epochs': 40,
    'optimizer': 'SGD',
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'batch_size': 64,             # Larger batch (lighter backbone)
    'use_amp': True,
    'early_stopping_patience': 25,
}
```

**Purpose**: Compare classic architecture vs modern ResNet/GoogLeNet

#### Exp06: GoogLeNet/Inception v1

**Script**: [`experiments/exp06_classification_GoogLeNet.py`](experiments/exp06_classification_GoogLeNet.py)

**Configuration:**
```python
model_config = {
    'dropout_rate': 0.5,
    'freeze_backbone': True,
    'pretrained': True,
    'use_auxiliary': True,        # Enable auxiliary classifiers
}

training_config = {
    'epochs': 120,
    'learning_rate': 0.001,
    'auxiliary_weight': 0.3,      # Weight for auxiliary losses
    'optimizer': 'Adam',
    'weight_decay': 1e-4,
    'batch_size': 32,
    'use_amp': True,
    'early_stopping_patience': 15,
}
```

**Purpose**: Test efficiency-focused architecture with multi-loss training

---

## 📈 Training Strategies

### Common Training Practices

#### 1. Two-Stage Training (Classification)

**Stage 1: Frozen Backbone**
- Freeze pretrained convolutional layers
- Train only the classifier head
- Higher learning rate (e.g., 0.001)
- Faster convergence on new task

**Stage 2: Fine-Tuning**
- Unfreeze backbone (partially or fully)
- Lower learning rate (e.g., 0.0001)
- Adjust pretrained features to specific task

#### 2. Mixed Precision Training (AMP)

All experiments use Automatic Mixed Precision:
```python
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Benefits:**
- ~2x faster training on GPU
- Reduced memory usage (allows larger batch sizes)
- No accuracy degradation

#### 3. Early Stopping

Monitor validation loss/metric:
```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    patience_counter = 0
    save_checkpoint(model, 'best_model.pth')
else:
    patience_counter += 1
    if patience_counter >= patience_limit:
        print("Early stopping triggered")
        break
```

#### 4. Learning Rate Scheduling

**Step Decay** (AlexNet):
```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=40, gamma=0.5
)
# LR: 0.01 → 0.005 → 0.0025 → ...
```

**ReduceLROnPlateau** (ResNet50):
```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
# Reduces LR when validation loss plateaus
```

### Data Augmentation

**Detection (YOLOv8 Built-in):**
- Mosaic augmentation
- Mixup
- Random horizontal flip
- HSV color space augmentation
- Perspective transform

**Classification (Custom Transforms):**
```python
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])
```

---

## 📊 Evaluation Metrics

### Detection Metrics

**Mean Average Precision (mAP):**
- mAP@0.5: IoU threshold = 0.5
- mAP@0.5:0.95: Average over IoU thresholds 0.5 to 0.95

**Precision-Recall Curve:**
- Shows trade-off between precision and recall at different confidence thresholds

**IoU Distribution:**
- Histogram of Intersection-over-Union values for predicted boxes

**Key Files:**
- [`src/evaluation/detection_evaluator.py`](src/evaluation/detection_evaluator.py)

### Classification Metrics

**Overall Metrics:**
- Accuracy: Overall correct predictions / total predictions
- Loss: Cross-entropy loss on test set

**Per-Class Metrics:**
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1-Score: Harmonic mean of precision and recall

**Confusion Matrix:**
- 5×5 matrix showing prediction distribution across classes

**ROC Curves:**
- One-vs-rest ROC curves for each emotion class
- AUC (Area Under Curve) calculation

**Key Files:**
- [`src/evaluation/classification_evaluator.py`](src/evaluation/classification_evaluator.py)

---

## 📂 Output Organization

Each experiment saves outputs to timestamped directories:

```
outputs/
├── exp01_detection_YOLOv8_baseline/
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
├── exp04_classification_ResNet50_baseline/
│   └── run_timestamp/
│       ├── model/
│       ├── logs/
│       └── figures/
│
└── ... (other experiments)
```

### Output Contents

#### Model Folder
- `best_model.pt` / `best_model.pth`: Trained model weights
- `model_config.json`: JSON file with all hyperparameters used

#### Logs Folder
- `training_log.csv`: CSV with columns:
  ```csv
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
- `evaluation_metrics.json`: Test set performance metrics

#### Figures Folder

**Detection experiments:**
- `precision_recall_curve.png`: PR curve for each class
- `IoU_distribution.png`: Histogram of IoU values
- `sample_detections.png`: Example predictions on test images
- `confusion_matrix.png`: If multi-class detection

**Classification experiments:**
- `confusion_matrix.png`: 5×5 confusion matrix
- `roc_curve.png`: ROC curves (one-vs-rest for each class)
- `per_class_metrics.png`: Bar chart of precision/recall/F1 per class
- `training_curves.png`: Loss and accuracy over epochs

---

## 🚀 Running Experiments

### Single Experiment

```bash
# Detection baseline
python experiments/exp01_detection_YOLOv8_baseline.py

# Classification baseline
python experiments/exp04_classification_ResNet50_baseline.py

# With custom parameters
python experiments/exp01_detection_YOLOv8_baseline.py --lr 0.001 --batch_size 32 --epochs 100
```

### All Experiments

```bash
bash scripts/run_all_experiments.sh
```

This runs all 8 experiments sequentially with proper error handling.

### Quick Testing with Small Subset

```bash
# Create small subset first
bash scripts/run_data_preprocessing.sh --create-subset

# Run with subset flag
python experiments/exp01_detection_YOLOv8_baseline.py --use-small-subset
python experiments/exp04_classification_ResNet50_baseline.py --use_small_subset
```

---

## 💡 Best Practices

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
- Enable gradient accumulation for very large batches
- Monitor GPU memory usage: `nvidia-smi`

### 3. Debugging Tips

**Problem**: Out of Memory (OOM)
```
Solution:
- Reduce batch_size
- Enable gradient_accumulation_steps
- Use smaller input_size (for detection)
- Enable AMP if not already enabled
```

**Problem**: Training loss not decreasing
```
Solution:
- Check learning rate (try lower/higher)
- Verify data loading is correct
- Check for label errors in dataset
- Ensure model is in training mode: model.train()
```

**Problem**: Overfitting (val loss increasing)
```
Solution:
- Increase dropout rate
- Add more data augmentation
- Enable early stopping
- Reduce model complexity
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

## 📚 Related Documentation

- **Data Preprocessing**: See [DATA_PREPROCESSING.md](DATA_PREPROCESSING.md) for dataset preparation
- **Web Application**: See [MODEL_APPLICATION.md](MODEL_APPLICATION.md) for inference deployment
- **Project Overview**: See [README.md](README.md) for architecture summary
