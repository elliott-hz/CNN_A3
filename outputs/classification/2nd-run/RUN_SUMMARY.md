# Classification Training Run Summary - 2nd Run

**Date**: 2026-04-26  
**Purpose**: Evaluate model performance consistency and identify training instability patterns

---

## Model Configurations

*Same as 1st Run - no configuration changes between runs*

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
| **ResNet50** | **63.99%** | 0.6401 | 0.6399 | 0.6350 | ~67.0% | 103 (early stop) |
| **AlexNet** | **43.84%** | 0.5003 | 0.4384 | 0.4171 | ~58.0% | 88 (early stop) |
| **GoogLeNet** | **40.84%** | 0.4737 | 0.4084 | 0.3785 | ~63.8% | 101 (early stop) |

---

## Comparison: 1st Run vs 2nd Run

| Model | 1st Run Test Acc | 2nd Run Test Acc | Change | 1st Run Val Acc | 2nd Run Val Acc | Val Change |
|-------|------------------|------------------|--------|-----------------|-----------------|------------|
| **ResNet50** | 64.20% | 63.99% | **-0.21%** | ~67.8% | ~67.0% | -0.8% |
| **AlexNet** | 47.16% | 43.84% | **-3.32%** ⚠️ | ~60.5% | ~58.0% | -2.5% |
| **GoogLeNet** | 36.33% | 40.84% | **+4.51%** ✅ | ~60.6% | ~63.8% | +3.2% |

### Key Observations:
1. **ResNet50**: Highly stable across runs (<0.3% variation). Consistent performance indicates robust training.
2. **AlexNet**: Significant degradation (-3.32%). Training stopped earlier (88 vs 115 epochs), suggesting increased instability.
3. **GoogLeNet**: Notable improvement (+4.51%). Better convergence in 2nd run, though still poor absolute performance.

---

## Per-Class Performance Analysis (2nd Run)

### ResNet50 - Consistent Performance
- **Strongest class**: happy (F1=0.8274, Precision=0.8483, Recall=0.8075)
- **Weakest class**: angry (F1=0.5260, Precision=0.6099, Recall=0.4624)
- **Change from 1st run**: 
  - angry: Improved precision (0.5570→0.6099) but worse recall (0.4731→0.4624)
  - alert: Slightly improved (F1: 0.5506→0.5530)
  - Overall: Very consistent, minor fluctuations

### AlexNet - Degraded Performance
- **Strongest class**: relax (F1=0.5314, Recall=0.9086 - still biased toward this class)
- **Weakest class**: alert (F1=0.3026, Recall=0.2193)
- **Change from 1st run**:
  - All classes degraded significantly
  - happy: Severe drop (F1: 0.5164→0.4335, Recall: 0.3797→0.3048)
  - frown: Moderate drop (F1: 0.4715→0.4588)
  - Critical: Training instability led to worse generalization

### GoogLeNet - Slight Improvement
- **Strongest class**: relax (F1=0.5514, Recall=0.9086 - extreme bias continues)
- **Weakest class**: alert (F1=0.2734, Recall=0.2032)
- **Change from 1st run**:
  - angry: Improved (F1: 0.3203→0.3360)
  - happy: Improved (F1: 0.1878→0.3460, Recall: 0.1070→0.2193) ✅
  - frown: Improved (F1: 0.3215→0.3866)
  - Still severely biased toward "relax" class

---

## Training Dynamics Analysis

### ResNet50
✅ **Consistency**: Nearly identical training curves to 1st run
- Final train accuracy: ~84.6% (both runs)
- Final val accuracy: ~67% (slightly lower than 1st run's 67.8%)
- Early stopping triggered at epoch 103 (vs 106 in 1st run)

⚠️ **Persistent Issues**:
- Validation plateau remains around 67-68%
- Train-val gap persists (~17%), indicating overfitting limit
- No significant improvement despite extended training

### AlexNet
❌ **Increased Instability**:
- Training stopped much earlier (88 epochs vs 115 in 1st run)
- Final train accuracy only ~56% (severe underfitting)
- Validation accuracy dropped to ~58% (from ~60.5%)
- SGD optimizer showing high variance between runs

⚠️ **Critical Problems**:
- StepLR scheduler reduced LR too aggressively (final LR=2.5e-5)
- Model failed to learn meaningful features
- BatchNorm layer may be causing gradient issues with SGD

### GoogLeNet
✅ **Improved Convergence**:
- Training progressed further (101 epochs vs 140, but better final performance)
- Lower final training loss (~0.89 vs ~1.89 in 1st run)
- Better validation accuracy (~63.8% vs ~60.6%)
- More stable training trajectory

⚠️ **Remaining Issues**:
- Still severe class imbalance in predictions
- Training loss higher than ResNet50 (~0.89 vs ~0.73)
- Disabled auxiliary classifiers limiting learning capacity

---

## Root Cause Analysis

### Why Did AlexNet Degrade?
1. **SGD Sensitivity**: SGD with momentum is highly sensitive to initialization and data ordering
2. **Learning Rate Schedule**: StepLR decayed LR too quickly, preventing adequate exploration
3. **Random Seed Impact**: Different random seed led to different mini-batch sequences, amplifying SGD instability
4. **BatchNorm + SGD Interaction**: BN statistics may not stabilize well with SGD on small batches

### Why Did GoogLeNet Improve?
1. **Better Initialization**: Random seed favored more effective weight initialization
2. **Full Unfreeze Benefit**: Complete backbone unfreezing allowed better adaptation
3. **Higher LR Advantage**: LR=0.001 provided sufficient gradient signal for convergence
4. **Cosine Scheduler**: Smoother LR decay compared to AlexNet's StepLR

### Why Is ResNet50 Stable?
1. **AdamW Robustness**: Adaptive optimizer less sensitive to hyperparameter variations
2. **Skip Connections**: Residual connections provide stable gradient flow
3. **Proven Architecture**: Extensive pretraining on ImageNet transfers well
4. **Moderate Hyperparameters**: Balanced LR and regularization prevent extremes

---

## Comparison with Optimization Goals

| Model | Original Test Acc | Target Test Acc | 1st Run Acc | 2nd Run Acc | Status |
|-------|------------------|-----------------|-------------|-------------|--------|
| ResNet50 | 64.20% | 68-70% | 64.20% | 63.99% | ❌ No improvement, stable |
| AlexNet | 47.16% | 55-58% | 47.16% | 43.84% | ❌ Degraded, unstable |
| GoogLeNet | 36.33% | 52-56% | 36.33% | 40.84% | ⚠️ Slight improvement, still poor |

**Conclusion**: Optimizations failed to achieve targets. Only GoogLeNet showed modest improvement, but all models remain far from goals.

---

## Critical Insights from Two Runs

### 1. Optimizer Choice Matters Most
- **AdamW** (ResNet50, GoogLeNet): Stable, reproducible results
- **SGD** (AlexNet): High variance, unreliable convergence
- **Recommendation**: Use AdamW for all models unless specific reason for SGD

### 2. Auxiliary Classifiers Are Important for GoogLeNet
- Disabling them removed multi-task learning benefit
- GoogLeNet was designed with auxiliary losses for a reason
- **Recommendation**: Re-enable with proper weighting (0.3 each)

### 3. Learning Rate Scheduling Strategy
- **Cosine Annealing** (ResNet50, GoogLeNet): Smooth, effective
- **StepLR** (AlexNet): Too aggressive, caused premature convergence
- **Recommendation**: Use cosine scheduling for all models

### 4. Architecture-Dataset Mismatch
- AlexNet's large FC layers may be too complex for ~9K image dataset
- GoogLeNet's efficiency comes at cost of training complexity
- ResNet50 provides best balance for this task size

---

## Recommended Improvements for 3rd Run

### Immediate Changes (High Priority)

#### AlexNet - Critical Fixes
1. **Switch to AdamW**: Replace SGD with AdamW optimizer
2. **Reduce LR**: Change from 0.0005 to 0.0002
3. **Remove BatchNorm**: Eliminate BN layer from classifier
4. **Lower weight decay**: Reduce from 1e-2 to 5e-3
5. **Use cosine scheduler**: Replace StepLR with CosineAnnealingWarmRestarts

#### GoogLeNet - Enable Auxiliary Losses
1. **Re-enable auxiliary**: Set `use_auxiliary=True` in model config
2. **Verify loss weighting**: Ensure 0.3 weight for aux1 and aux2
3. **Increase LR**: Try 0.002 instead of 0.001
4. **Partial unfreeze**: Only unfreeze inception4b onwards (not full unfreeze)

#### ResNet50 - Push Beyond Plateau
1. **Unfreeze layer2**: Add layer2 to unfrozen layers in Phase 2
2. **Increase T_0**: Change cosine scheduler T_0 from 30 to 50
3. **Add CutMix augmentation**: Implement CutMix or MixUp for better generalization
4. **Extend training**: Increase epochs to 250, patience to 40

### Experimental Changes (Medium Priority)

1. **Learning Rate Warmup**: Add 5-epoch linear warmup for all models
2. **Gradient Clipping**: Add `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
3. **Label Smoothing Adjustment**: Try 0.05 instead of 0.1 for ResNet50
4. **Class Weight Tuning**: Compute weights based on validation set, not training set
5. **Mixed Precision Verification**: Ensure AMP is working correctly (check loss scaling)

### Data-Level Improvements (Low Priority)

1. **Verify Augmentation Consistency**: Log augmentation parameters for each experiment
2. **Check Data Leakage**: Ensure no test images in training/validation splits
3. **Analyze Class Distribution**: Plot histogram of samples per class
4. **Hard Example Mining**: Identify consistently misclassified samples

---

## Action Items for 3rd Run

### Pre-Training Checklist
- [ ] Update AlexNet config: optimizer=adamw, lr=0.0002, remove BN, weight_decay=5e-3
- [ ] Update GoogLeNet config: use_auxiliary=True, lr=0.002, partial unfreeze
- [ ] Update ResNet50 config: unfreeze layer2, T_0=50, epochs=250
- [ ] Verify all three experiments use identical data splits
- [ ] Add logging for augmentation statistics
- [ ] Implement gradient clipping in trainer

### During Training
- [ ] Monitor training curves in real-time (tensorboard or wandb)
- [ ] Save checkpoints every 10 epochs for post-hoc analysis
- [ ] Track per-class metrics during validation (not just overall accuracy)
- [ ] Log learning rate at each epoch
- [ ] Record GPU memory usage and training speed

### Post-Training Analysis
- [ ] Compare confusion matrices across all three models
- [ ] Analyze misclassified samples for common patterns
- [ ] Calculate inference time for each model
- [ ] Measure model size and parameter count
- [ ] Create ensemble predictions (average probabilities from all 3 models)

---

## Expected Outcomes for 3rd Run

| Model | Current Best Acc | Target Acc | Confidence |
|-------|------------------|------------|------------|
| ResNet50 | 64.20% | **68-70%** | Medium (needs architectural changes) |
| AlexNet | 47.16% | **55-58%** | High (optimizer fix should help) |
| GoogLeNet | 40.84% | **52-56%** | Medium-High (auxiliary losses critical) |

**Overall Goal**: Achieve average test accuracy >60% across all three models with stable, reproducible training.

---

**Next Steps**: Implement recommended changes, run 3rd iteration with enhanced monitoring, and perform detailed error analysis on misclassified samples.
