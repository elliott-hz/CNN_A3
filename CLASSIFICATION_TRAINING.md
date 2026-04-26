# Classification Model Training Guide

This document provides comprehensive information about classification model architectures, experiment configurations, training strategies, and evaluation metrics for the Visual Dog Emotion Recognition system.

---

## 📊 Experiment Overview

Three classification experiments comparing different CNN architectures and training strategies.

| Experiment | Script | Architecture | Parameters | Era | Key Feature |
|------------|--------|--------------|------------|-----|-------------|
| Exp04: ResNet50 | `exp04_classification_ResNet50_baseline.py` | ResNet50 | ~25.6M | 2015 | Modern residual network with skip connections |
| Exp05: AlexNet | `exp05_classification_AlexNet.py` | AlexNet | ~60M | 2012 | Classic CNN with large FC layers |
| Exp06: GoogLeNet | `exp06_classification_GoogLeNet.py` | GoogLeNet | ~7M | 2014 | Efficient inception modules with auxiliary classifiers |

**Note**: All classification experiments use the **same dataset and preprocessing pipeline** for fair comparison:
- **Dataset**: Dog Emotion Dataset (~9,325 images, 5 classes)
- **Preprocessing**: Images resized to 224×224, normalized to [0,1]
- **Splits**: Train/Val/Test (70/20/10) with stratification
- **Training**: Mixed precision (AMP), early stopping

---

## 📊 Detailed Comparison of Classification Experiments

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
| **Training Epochs** | 180 | 200 (Longer) | 180 |
| **Optimizer** | AdamW | SGD + Momentum | AdamW |
| **Learning Rate** | 0.0002 | 0.0005 (High) | 0.001 |
| **Batch Size** | 32 | 32 | 32 |
| **Weight Decay** | 5e-3 | 1e-2 (Stronger) | 5e-3 |
| **Gradient Accumulation** | 1 | 1 | 1 |
| **Label Smoothing** | 0.1 | 0.1 | 0.1 |
| **Class Weighting** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Mixed Precision (AMP)** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Early Stopping Patience** | 25 | 30 (Longer) | 25 |
| **LR Scheduler** | Cosine Annealing Warm Restarts | StepLR (decay=0.5, interval=50) | Cosine Annealing Warm Restarts |
| **Training Strategy** | Transfer learning (frozen) | Classic SGD approach | Multi-loss with auxiliary |
| **Best For** | Establishing strong baseline | Comparing classic vs modern | Efficiency-focused applications |
| **Memory Requirements** | Medium | High (large FC layers) | Low (most efficient) |
| **Expected Training Speed** | Fast | Moderate (SGD) | Fast (lightweight) |
| **Expected Inference Speed** | Fast | Moderate | Very Fast |
| **Risk of Overfitting** | Low | Low (strong regularization) | Low (efficient design) |

### Key Design Decisions Explained

#### Exp04: ResNet50 (Modern Architecture with Transfer Learning)

**Design Philosophy:**
- Use modern residual network with skip connections to prevent vanishing gradients
- Freeze backbone to leverage pretrained ImageNet features (transfer learning)
- Train only classifier head with moderate learning rate for stable convergence
- Balanced dropout (0.5) and weight decay (5e-3) for regularization
- AdamW optimizer for adaptive learning rates with better weight decay handling
- Cosine Annealing Warm Restarts scheduler for periodic LR resets

**Current Configuration (Optimized):**
```python
model_config = {
    'dropout_rate': 0.5,
    'additional_fc_layers': True,
    'freeze_backbone': True,
    'pretrained': True,
    'use_batch_norm': True,
    'num_classes': 5,
}

training_config = {
    'learning_rate': 0.0002,      # Increased from 0.0001
    'epochs': 180,                # Extended from 150
    'optimizer': 'adamw',
    'weight_decay': 5e-3,         # Reduced from 1e-2
    'batch_size': 32,
    'early_stopping_patience': 25,
    'lr_scheduler': 'cosine_annealing_warm_restarts',
    'T_0': 30,                    # Extended restart cycle
    'T_mult': 2,
    'eta_min': 1e-6
}
```

**Configuration Flexibility:**
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
- Higher initial LR (0.0005) suitable for SGD training
- Standard batch size (32) for stability
- Stronger weight decay (1e-2) to regularize large FC layers
- Extended training (200 epochs) with step decay schedule
- Longer early stopping patience (30) to avoid premature termination
- Added BatchNorm layer in classifier for improved gradient flow

**Current Configuration (Optimized):**
```python
model_config = {
    'dropout_rate': 0.5,
    'freeze_backbone': True,
    'pretrained': True,
    'num_classes': 5,
}

training_config = {
    'learning_rate': 0.0005,      # Significantly increased from 0.0001
    'epochs': 200,                # Significantly extended from 150
    'optimizer': 'sgd',           # Switched from AdamW
    'momentum': 0.9,
    'weight_decay': 1e-2,
    'batch_size': 32,
    'early_stopping_patience': 30,
    'lr_scheduler': 'step',       # StepLR更适合SGD
    'lr_decay_factor': 0.5,
    'lr_decay_interval': 50
}
```

**Model Architecture Modification:**
```python
self.classifier = nn.Sequential(
    nn.Dropout(self.dropout_rate),
    nn.Linear(9216, 512),
    nn.BatchNorm1d(512),  # Added BN layer for stability
    nn.ReLU(inplace=True),
    nn.Dropout(self.dropout_rate),
    nn.Linear(512, self.num_classes)
)
```

**Trade-offs:**
- ✅ Simple, well-understood architecture
- ✅ Traditional SGD approach proven effective
- ❌ Large parameter count (~60M) due to dense FC layers
- ❌ No skip connections or batch normalization in backbone
- ❌ Slower inference compared to modern architectures

#### Exp06: GoogLeNet (Efficiency-Focused)

**Design Philosophy:**
- Maximize accuracy-to-parameter ratio
- Leverage Inception modules for multi-scale feature extraction
- Disable auxiliary classifiers to reduce noise (optimized configuration)
- Use AdamW optimizer (works well with complex architectures)
- Global average pooling eliminates massive FC layers
- Most parameter-efficient design (~7M vs ~25M for ResNet50)
- Simplified classifier head for better gradient flow

**Current Configuration (Optimized):**
```python
model_config = {
    'dropout_rate': 0.5,
    'freeze_backbone': True,
    'pretrained': True,
    'use_auxiliary': False,       # Disabled to reduce noise
    'num_classes': 5,
}

training_config = {
    'learning_rate': 0.001,       # Significantly increased from 0.0001
    'epochs': 180,                # Extended from 150
    'optimizer': 'adamw',
    'weight_decay': 5e-3,         # Reduced from 1e-2
    'batch_size': 32,
    'early_stopping_patience': 25,
    'lr_scheduler': 'cosine_annealing_warm_restarts',
    'T_0': 25,
    'T_mult': 2,
    'eta_min': 1e-6
}
```

**Model Architecture Modification:**
```python
self.classifier = nn.Sequential(
    nn.Dropout(self.dropout_rate),
    nn.Linear(1024, self.num_classes)  # Simplified: removed intermediate FC layer
)
```

**Unique Features:**
- Inception modules perform parallel convolutions (1×1, 3×3, 5×5)
- Auxiliary classifiers can be enabled/disabled based on training needs
- Dimensionality reduction via 1×1 convolutions before expensive operations
- Best choice for resource-constrained deployment scenarios

**Implementation Note:** Manually extracts features before final FC layer to avoid dimension mismatch (1024-dim features → custom classifier)

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

## 🏗️ Classification Model Architectures

### 1. ResNet50Classifier

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

### 2. AlexNetClassifier

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
- Added BatchNorm in classifier for improved training stability

### 3. GoogLeNetClassifier

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
- Can enable/disable auxiliary classifiers based on training needs

---

## 📈 Classification Training Strategies

### Two-Stage Training Strategy

All classification experiments use a two-stage training approach:

**Stage 1: Frozen Backbone (15 epochs)**
- Freeze pretrained convolutional layers
- Train only the classifier head
- Higher learning rate (e.g., 0.0002-0.001)
- Faster convergence on new task
- Allows classifier to adapt to task-specific features

**Stage 2: Fine-Tuning (remaining epochs)**
- Unfreeze backbone (partially or fully based on architecture)
- Lower learning rate (LR × 0.1)
- Adjust pretrained features to specific task
- GoogLeNet uses full unfreeze, others use partial unfreeze

### Data Augmentation Pipeline

Classification experiments use enhanced data augmentation applied on-the-fly:

**PIL-level transforms** (before tensor conversion):
- Random horizontal flip (p=0.5)
- Random rotation (±15 degrees)
- Color jitter (brightness, contrast, saturation, hue)
- Random affine transforms (translation, scaling)

**Tensor-level transforms** (after tensor conversion):
- Random erasing (p=0.2) - improves robustness
- Gaussian blur (kernel_size=3, sigma=(0.1, 2.0)) - improves invariance

**Implementation**: [`src/training/classification_trainer.py`](src/training/classification_trainer.py) - `AugmentedDataset` class

### Learning Rate Scheduling

**Cosine Annealing Warm Restarts** (ResNet50 & GoogLeNet):
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=30, T_mult=2, eta_min=1e-6
)
# Periodically restarts LR with increasing cycle length
```

**StepLR** (AlexNet):
```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=50, gamma=0.5
)
# LR: 0.0005 → 0.00025 → 0.000125 → ...
```

### Regularization Techniques

1. **Dropout**: 0.5 rate across all models
2. **Weight Decay**: Model-specific (5e-3 to 1e-2)
3. **Label Smoothing**: 0.1 to prevent overconfidence
4. **Class Weighting**: Balanced weights for imbalanced data
5. **Data Augmentation**: Enhanced pipeline with RandomErasing and GaussianBlur
6. **Early Stopping**: Patience 25-30 epochs

---

## 📊 Classification Evaluation Metrics

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

## 📂 Classification Output Organization

Each classification experiment saves outputs to timestamped directories:

```
outputs/exp04_classification_ResNet50_baseline/
└── run_20260426_071228/
    ├── model/
    │   ├── best_model.pth      ← Best model weights
    │   └── model_config.json  ← Model configuration used
    │
    ├── logs/
    │   ├── training_log.csv   ← Epoch-by-epoch metrics
    │   └── experiment_report.md ← Full markdown report
    │
    └── figures/
        ├── confusion_matrix.png
        ├── roc_curve.png
        ├── per_class_metrics.png
        └── training_curves.png
```

### Output Contents

#### Model Folder
- `best_model.pth`: Trained model weights (best validation accuracy)
- `final_model.pth`: Final model weights after training completion
- `model_config.json`: JSON file with all hyperparameters used

#### Logs Folder
- `training_log.csv`: CSV with columns:
  ```csv
  epoch, train_loss, val_loss, train_acc, val_acc, learning_rate
  1, 2.345, 2.123, 0.45, 0.48, 0.0002
  2, 1.987, 1.876, 0.52, 0.55, 0.0002
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
- `confusion_matrix.png`: 5×5 confusion matrix
- `roc_curve.png`: ROC curves (one-vs-rest for each class)
- `per_class_metrics.png`: Bar chart of precision/recall/F1 per class
- `training_curves.png`: Loss and accuracy over epochs

---

## 🚀 Running Classification Experiments

### Single Experiment

```bash
# ResNet50 baseline
python experiments/exp04_classification_ResNet50_baseline.py

# AlexNet
python experiments/exp05_classification_AlexNet.py

# GoogLeNet
python experiments/exp06_classification_GoogLeNet.py

# With custom parameters
python experiments/exp04_classification_ResNet50_baseline.py --lr 0.0002 --batch_size 32 --epochs 180
```

### Quick Testing with Small Subset

```bash
# Run with subset flag for quick validation
python experiments/exp04_classification_ResNet50_baseline.py --use_small_subset
python experiments/exp05_classification_AlexNet.py --use_small_subset
python experiments/exp06_classification_GoogLeNet.py --use_small_subset
```

---

## 💡 Classification Best Practices

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

**Problem**: Overfitting (val loss increasing, gap > 10%)
```
Solution:
- Increase dropout rate
- Add more data augmentation
- Enable early stopping
- Increase weight_decay
- Reduce model complexity
```

**Problem**: Underfitting (train acc < 70%)
```
Solution:
- Increase learning rate
- Train longer (more epochs)
- Unfreeze backbone layers
- Decrease dropout rate
- Check if model capacity is sufficient
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

## 🚀 Optimization Plan & Implementation Record

### 📊 Optimization Background

Based on the first round of experimental results (ResNet50: 64.2%, AlexNet: 47.16%, GoogLeNet: 36.33%), the following issues were identified:
- **ResNet50**: Entered a plateau (Val Acc ~68%), needs to break through the bottleneck.
- **AlexNet**: Severe underfitting (Train Acc only 70%), learning rate too low.
- **GoogLeNet**: Severe underfitting + overfitting (Train/Val Acc both ~60%), auxiliary classifiers introducing noise.

### 🎯 Optimization Goals

Improve the accuracy of all three models while maintaining **dataset consistency**, **training epochs ≥ 100**, and **comparability**.

---

### ✅ Implemented Modifications

#### 1. ResNet50 Optimization (exp04)

**Configuration Adjustments**:
```python
training_config = {
    'learning_rate': 0.0002,      # ↑ Increased from 0.0001
    'epochs': 180,                # ↑ Extended from 150
    'weight_decay': 5e-3,         # ↓ Reduced from 1e-2
    'early_stopping_patience': 25, # ↑ Increased from 20
    'T_0': 30,                    # ↑ Extended restart cycle
}
```

**Expected Outcome**: Val Acc increased to 72-75%, Test Acc increased to 68-70%.

---

#### 2. AlexNet Optimization (exp05)

**Key Changes**:
1. **Switch Optimizer**: AdamW → SGD + Momentum (aligns with original AlexNet design).
2. **Significantly Increase Learning Rate**: 0.0001 → 0.0005.
3. **Significantly Extend Training**: 150 → 200 epochs.
4. **Add BatchNorm Layers**: Added BN in classifier to improve gradient flow.

**Configuration Adjustments**:
```python
training_config = {
    'learning_rate': 0.0005,      # ↑↑ Significant increase
    'epochs': 200,                # ↑↑ Significant extension
    'optimizer': 'sgd',           # ← Switched back to SGD
    'momentum': 0.9,
    'early_stopping_patience': 30, # ↑↑ Significantly increased patience
    'lr_scheduler': 'step',       # ← StepLR is more suitable for SGD
    'lr_decay_factor': 0.5,
    'lr_decay_interval': 50
}
```

**Model Architecture Modification** (`src/models/classification_model.py`):
```python
self.classifier = nn.Sequential(
    nn.Dropout(self.dropout_rate),
    nn.Linear(9216, 512),
    nn.BatchNorm1d(512),  # ← New BN layer
    nn.ReLU(inplace=True),
    nn.Dropout(self.dropout_rate),
    nn.Linear(512, self.num_classes)
)
```

**Expected Outcome**: Val Acc increased to 58-62%, Test Acc increased to 55-58%.

---

#### 3. GoogLeNet Optimization (exp06)

**Key Changes**:
1. **Disable Auxiliary Classifiers**: `use_auxiliary=False` (reduce noise).
2. **Simplify Classifier Head**: Remove intermediate FC layer (1024→num_classes).
3. **Significantly Increase Learning Rate**: 0.0001 → 0.001.
4. **Full Unfreeze in Phase 2**: Unfreeze all backbone layers.

**Configuration Adjustments**:
```python
model_config['use_auxiliary'] = False  # ← Disable auxiliary classifiers

training_config = {
    'learning_rate': 0.001,       # ↑↑ Significant increase
    'epochs': 180,                # ↑ Extended from 150
    'weight_decay': 5e-3,         # ↓ Reduced from 1e-2
    'early_stopping_patience': 25, # ↑ Increased from 20
    'T_0': 25,                    # Restart cycle
}
```

**Model Architecture Modification** (`src/models/classification_model.py`):
```python
self.classifier = nn.Sequential(
    nn.Dropout(self.dropout_rate),
    nn.Linear(1024, self.num_classes)  # ← Simplified: removed intermediate FC
)
```

**Training Logic Modification** (`src/training/classification_trainer.py`):
```python
# Phase 2: Full unfreeze for GoogLeNet
unfreeze_all = self.model_config.get('architecture', '').lower() == 'googlenet'
model.unfreeze_backbone(unfreeze_all=unfreeze_all)
```

**Expected Outcome**: Val Acc increased to 55-60%, Test Acc increased to 52-56%.

---

#### 4. Common Enhancements

**Enhanced Data Augmentation** (`src/training/classification_trainer.py`):
```python
# PIL-level transforms
transforms.RandomHorizontalFlip(p=0.5),
transforms.RandomRotation(degrees=15),
transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),

# Tensor-level transforms (NEW)
transforms.RandomErasing(p=0.2),  # Improves robustness
transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Improves invariance
```

**Extended Phase 1 Training** (`src/training/classification_trainer.py`):
- Phase 1 (frozen backbone): 10 epochs → **15 epochs**
- Purpose: Better classifier adaptation before fine-tuning

---

### 📈 Expected Performance Improvements

| Model | Original Test Acc | Target Test Acc | Improvement | Key Changes |
|-------|------------------|-----------------|-------------|-------------|
| **ResNet50** | 64.20% | **68-70%** | +4-6% | Higher LR, longer training, stronger augmentation |
| **AlexNet** | 47.16% | **55-58%** | +8-11% | SGD optimizer, BN layer, 200 epochs, higher LR |
| **GoogLeNet** | 36.33% | **52-56%** | +16-20% | Disable aux, higher LR, simplified head, full unfreeze |

---

## 🐛 Bug Fixes & Troubleshooting

### Issue 1: GaussianBlur and RandomErasing Compatibility Error

**Date**: 2026-04-26  
**Error Message**: 
```
AttributeError: 'Image' object has no attribute 'shape'. Did you mean: 'save'?
```

**Root Cause**:
- `transforms.GaussianBlur` and `transforms.RandomErasing` in newer torchvision versions require **Tensor input** (with `.shape` attribute)
- Our code was passing **PIL Image** objects to these transforms
- PIL Image objects have `.size` but not `.shape`, causing the AttributeError

**Solution Implemented**:
Split the augmentation pipeline into two stages in [`src/training/classification_trainer.py`](src/training/classification_trainer.py):

1. **PIL-level transforms** (applied before ToTensor):
   - RandomHorizontalFlip
   - RandomRotation
   - ColorJitter
   - RandomAffine

2. **Tensor-level transforms** (applied after ToTensor):
   - RandomErasing
   - GaussianBlur

**Code Changes**:
```python
# In AugmentedDataset.__init__
self.transform_pil = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(...),
    transforms.RandomAffine(...),
])

self.transform_tensor = transforms.Compose([
    transforms.RandomErasing(p=0.2),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
])

# In AugmentedDataset.__getitem__
pil_img = self.transform_pil(pil_img) if self.transform_pil else pil_img
tensor_img = transforms.ToTensor()(pil_img)
tensor_img = self.transform_tensor(tensor_img) if self.transform_tensor else tensor_img
```

**Impact**: All three classification experiments (exp04, exp05, exp06) now run without errors.

---

### Issue 2: GoogLeNet Auxiliary Classifiers TypeError

**Date**: 2026-04-26  
**Error Message**:
```
TypeError: 'NoneType' object is not callable
```

**Affected Experiment**: Exp06 GoogLeNet only (ResNet50 and AlexNet work fine)

**Root Cause**:
- Configuration set `use_auxiliary=False` to disable auxiliary classifiers
- However, [GoogLeNetClassifier.forward](file:///Users/elliott/vscode_workplace/CNN_A3/src/models/classification_model.py#L520-L567) method still attempted to call `self.backbone.aux1(x)` and `self.backbone.aux2(x)`
- When `use_auxiliary=False`, these auxiliary classifiers are not replaced and remain as original PyTorch implementations or None
- The check `hasattr(self.backbone, 'aux1')` returns True even when aux is None, leading to calling None as a function

**Why Only GoogLeNet?**:
- ResNet50 and AlexNet don't have auxiliary classifier mechanisms
- Only GoogLeNet architecture includes auxiliary classifiers for multi-loss training

**Solution Implemented**:
Modified [`src/models/classification_model.py`](src/models/classification_model.py) - [GoogLeNetClassifier.forward](src/models/classification_model.py) method:

1. **Enhanced auxiliary classifier checks**:
```python
# Before (incorrect):
if hasattr(self.backbone, 'aux1') and self.training:
    aux1_out = self.backbone.aux1(x)

# After (correct):
aux1_out = None
if self.use_auxiliary and self.training and hasattr(self.backbone, 'aux1') and self.backbone.aux1 is not None:
    aux1_out = self.backbone.aux1(x)
```

2. **Fixed return value logic**:
```python
# Before:
if self.training and hasattr(self.backbone, 'aux1') and aux1_out is not None:
    return main_logits, aux1_out, aux2_out

# After:
if self.training and self.use_auxiliary and aux1_out is not None:
    return main_logits, aux1_out, aux2_out
else:
    return main_logits
```

**Key Improvements**:
- Added explicit `self.use_auxiliary` configuration check
- Added `is not None` validation before calling auxiliary classifiers
- Ensured return values match the training mode and configuration

**Impact**: GoogLeNet experiment can now run successfully with `use_auxiliary=False` configuration.

---

**Last Updated**: 2026-04-26  
**Implemented by**: AI Assistant  
**Status**: ✅ Code modifications completed, ready for GPU environment verification
