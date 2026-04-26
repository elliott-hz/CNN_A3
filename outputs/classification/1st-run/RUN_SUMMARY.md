# Classification Training Run Summary - 1st Run

**Date**: 2026-04-26  
**Purpose**: Evaluate optimized training configurations after implementing CLASSIFICATION_TRAINING.md recommendations

---

## Model Configurations

### ResNet50 (exp04)
- **Architecture**: ResNet50 with additional FC layers + BatchNorm
- **Optimizer**: AdamW, LR=0.0002, Weight Decay=5e-3
- **Scheduler**: Cosine Annealing Warm Restarts (T_0=30, T_mult=2)
- **Epochs**: 180 (Phase 1: 15 frozen, Phase 2: 165 fine-tune)
- **Batch Size**: 32
- **Early Stopping Patience**: 25

### AlexNet (exp05)
- **Architecture**: AlexNet with BatchNorm in classifier
- **Optimizer**: SGD + Momentum (0.9), LR=0.0005, Weight Decay=1e-2
- **Scheduler**: StepLR (decay=0.5 every 50 epochs)
- **Epochs**: 200 (Phase 1: 15 frozen, Phase 2: 185 fine-tune)
- **Batch Size**: 32
- **Early Stopping Patience**: 30

### GoogLeNet (exp06)
- **Architecture**: GoogLeNet with auxiliary classifiers **DISABLED**, simplified head (1024→5)
- **Optimizer**: AdamW, LR=0.001, Weight Decay=5e-3
- **Scheduler**: Cosine Annealing Warm Restarts (T_0=25, T_mult=2)
- **Epochs**: 180 (Phase 1: 15 frozen, Phase 2: 165 full unfreeze)
- **Batch Size**: 32
- **Early Stopping Patience**: 25

---

## Test Results Summary

| Model | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Best Val Acc | Epochs Trained |
|-------|---------------|----------------|-------------|---------------|--------------|----------------|
| **ResNet50** | **64.20%** | 0.6420 | 0.6420 | 0.6374 | ~67.8% | 106 (early stop) |
| **AlexNet** | **47.16%** | 0.5180 | 0.4716 | 0.4554 | ~60.5% | 115 (early stop) |
| **GoogLeNet** | **36.33%** | 0.4395 | 0.3633 | 0.3175 | ~60.6% | 140 (early stop) |

---

## Per-Class Performance Analysis

### ResNet50 - Best Performing Model
- **Strongest class**: happy (F1=0.8212, Precision=0.8596, Recall=0.7861)
- **Weakest class**: angry (F1=0.5116, Precision=0.5570, Recall=0.4731)
- **Issue**: alert class has low precision (0.5354), indicating false positives
- **Observation**: relax class has high recall (0.8710) but moderate precision (0.6612)

### AlexNet - Moderate Performance
- **Strongest class**: happy (Precision=0.8068, but low Recall=0.3797 → biased predictions)
- **Weakest class**: alert (F1=0.3105, very low Recall=0.2299)
- **Issue**: Severe class imbalance in predictions - over-predicts relax (Recall=0.8710) but under-predicts happy
- **Observation**: High variance in per-class metrics indicates unstable training

### GoogLeNet - Worst Performance
- **Strongest class**: relax (F1=0.5080, Recall=0.9355 - extremely biased toward this class)
- **Weakest class**: happy (F1=0.1878, Recall=0.1070 - severely under-detected)
- **Critical Issue**: Model collapsed to predicting mostly "relax" class
- **Observation**: Training loss remained high (~0.89-1.90), indicating model failed to converge properly

---

## Training Dynamics Analysis

### ResNet50
✅ **Strengths**:
- Stable convergence throughout training
- Good separation between train/val accuracy (Train: ~84.6%, Val: ~67.8%)
- Consistent improvement in both phases

⚠️ **Issues**:
- Validation accuracy plateaued around epoch 95-106
- Gap between train and val accuracy suggests some overfitting
- Early stopping triggered at epoch 106 (patience=25)

### AlexNet
❌ **Critical Issues**:
- Extremely unstable training with oscillating validation accuracy
- SGD optimizer with high LR (0.0005) causing gradient instability
- Training accuracy only reached ~70%, indicating severe underfitting
- Early stopping triggered prematurely at epoch 115 despite patience=30

⚠️ **Root Causes**:
- SGD may not be suitable for this dataset/task combination
- BatchNorm layer in classifier might be interfering with gradient flow
- Weight decay (1e-2) too aggressive for AlexNet's large FC layers

### GoogLeNet
❌ **Critical Issues**:
- Highest training loss among all models (~0.89-1.90 vs ~0.73-0.77 for ResNet)
- Disabled auxiliary classifiers removed helpful multi-task learning signal
- Full unfreeze strategy may have destabilized pretrained features
- Model converged to trivial solution (predicting majority class)

⚠️ **Root Causes**:
- Learning rate (0.001) might still be too low for effective fine-tuning
- Simplified classifier head (1024→5) lacks capacity for complex decision boundary
- GoogLeNet architecture may require auxiliary losses for stable training on small datasets

---

## Comparison with Optimization Goals

| Model | Original Test Acc | Target Test Acc | Actual Test Acc | Status |
|-------|------------------|-----------------|-----------------|--------|
| ResNet50 | 64.20% | 68-70% | **64.20%** | ❌ No improvement |
| AlexNet | 47.16% | 55-58% | **47.16%** | ❌ No improvement |
| GoogLeNet | 36.33% | 52-56% | **36.33%** | ❌ No improvement |

**Conclusion**: The optimizations did NOT achieve the expected improvements. Models performed similarly to or worse than baseline.

---

## Recommended Improvements for Next Iteration

### ResNet50
1. **Unfreeze more layers**: Try unfreezing layer2 in addition to layer3/layer4
2. **Adjust scheduler**: Increase T_0 from 30 to 50 for smoother LR decay
3. **Reduce regularization**: Lower weight_decay from 5e-3 to 1e-3
4. **Increase augmentation**: Add CutMix or MixUp for better generalization
5. **Longer training**: Extend to 250 epochs with patience=40

### AlexNet
1. **Switch optimizer**: Change from SGD back to AdamW (more stable for this task)
2. **Reduce LR**: Decrease from 0.0005 to 0.0002
3. **Remove BatchNorm**: Eliminate BN layer from classifier (may cause instability)
4. **Lower weight decay**: Reduce from 1e-2 to 5e-3
5. **Alternative**: Consider using Adam optimizer instead of SGD/AdamW

### GoogLeNet
1. **Re-enable auxiliary classifiers**: Set `use_auxiliary=True` with proper loss weighting
2. **Increase LR further**: Try 0.002 or 0.005 for faster convergence
3. **Add intermediate FC layer**: Restore 1024→512→5 structure for better feature transformation
4. **Partial unfreeze**: Instead of full unfreeze, only unfreeze inception4/5 modules
5. **Reduce dropout**: Lower from 0.5 to 0.3 to prevent underfitting

### General Recommendations
1. **Data augmentation consistency**: Verify all three experiments use identical augmentation pipeline
2. **Random seed control**: Ensure reproducibility across runs
3. **Learning rate warmup**: Add 5-epoch warmup period for all models
4. **Gradient clipping**: Add gradient clipping (max_norm=1.0) to stabilize training
5. **Ensemble approach**: Consider averaging predictions from multiple checkpoints

---

## Action Items for 2nd Run

- [ ] Implement AlexNet optimizer change (SGD → AdamW)
- [ ] Re-enable GoogLeNet auxiliary classifiers
- [ ] Adjust learning rates based on observations
- [ ] Monitor training curves more closely for early signs of instability
- [ ] Save intermediate checkpoints every 10 epochs for analysis
- [ ] Compare per-class confusion matrices to identify systematic errors

---

**Next Steps**: Implement recommended changes and run 2nd iteration. Track improvements in test accuracy, training stability, and per-class performance balance.
