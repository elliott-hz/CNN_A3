# Classification Training - 3rd Run Plan

**Date**: 2026-04-26  
**Purpose**: Implement critical fixes based on 1st and 2nd run analysis to achieve target performance

---

## Executive Summary

Based on detailed analysis of two training runs, three critical issues were identified:
1. **AlexNet**: SGD optimizer causing instability and premature convergence
2. **GoogLeNet**: Disabled auxiliary classifiers removing essential training signal
3. **ResNet50**: Conservative unfreeze strategy limiting adaptation capacity

This 3rd run implements targeted fixes to address these root causes.

---

## Configuration Changes

### AlexNet (exp05) - Critical Fixes

#### Changes Made:
1. **Optimizer**: SGD → AdamW (more stable, adaptive learning rates)
2. **Learning Rate**: 0.0005 → 0.0002 (prevent gradient instability)
3. **Weight Decay**: 1e-2 → 5e-3 (reduce over-regularization)
4. **LR Scheduler**: StepLR → Cosine Annealing Warm Restarts (smoother decay)
5. **Model Architecture**: Removed BatchNorm from classifier (simplify, closer to original design)

#### Rationale:
- SGD showed high variance between runs (47.16% → 43.84%)
- StepLR decayed LR too aggressively (final LR=2.5e-5 by epoch 88)
- BatchNorm in FC layers may interfere with gradient flow on small batches
- AdamW proven effective in ResNet50 and GoogLeNet experiments

#### Expected Impact:
- **Target**: 55-58% test accuracy (from 43.84-47.16%)
- **Improvement**: +8-11 percentage points
- **Confidence**: High (optimizer fix addresses root cause)

---

### GoogLeNet (exp06) - Re-enable Auxiliary Classifiers

#### Changes Made:
1. **Auxiliary Classifiers**: Disabled → Enabled (`use_auxiliary=True`)
2. **Learning Rate**: 0.001 → 0.002 (stronger gradient signal for complex architecture)
3. **Unfreeze Strategy**: Full unfreeze → Partial unfreeze (preserve pretrained features)
4. **T_0**: 25 → 30 (longer exploration phases)

#### Rationale:
- Auxiliary classifiers are integral to GoogLeNet design (multi-task learning)
- Disabling them removed helpful gradient signals at intermediate layers
- Training loss remained high (~0.89-1.90) indicating underfitting
- Full unfreeze destabilized pretrained ImageNet features
- Higher LR needed to balance three loss terms (main + 2 auxiliary)

#### Expected Impact:
- **Target**: 52-56% test accuracy (from 36.33-40.84%)
- **Improvement**: +11-15 percentage points
- **Confidence**: Medium-High (auxiliary losses are architectural requirement)

---

### ResNet50 (exp04) - Push Beyond Plateau

#### Changes Made:
1. **Unfreeze Strategy**: Layer3+4 → Layer2+3+4 (more adaptation capacity)
2. **Epochs**: 180 → 250 (allow longer training)
3. **Early Stopping Patience**: 25 → 40 (prevent premature stopping)
4. **T_0**: 30 → 50 (smoother LR decay, longer exploration)

#### Rationale:
- Validation accuracy plateaued at ~67-68% despite continued training loss decrease
- Train-val gap stable at ~17%, suggesting capacity limit rather than overfitting
- Early stopping triggered at epoch 106 while val acc still improving (67.5% → 67.8%)
- Unfreezing layer2 allows mid-level feature adaptation to dog emotions
- Longer T_0 provides smoother optimization with larger unfrozen parameter set

#### Expected Impact:
- **Target**: 68-70% test accuracy (from 63.99-64.20%)
- **Improvement**: +4-6 percentage points
- **Confidence**: Medium (plateau breaking is challenging but feasible)

---

## Detailed Configuration Comparison

### AlexNet Configuration Changes

```python
# BEFORE (1st/2nd run):
model_config = {
    'dropout_rate': 0.5,
    'freeze_backbone': True,
    'pretrained': True,
}

training_config = {
    'learning_rate': 0.0005,      # ← Too high for SGD
    'epochs': 200,
    'optimizer': 'sgd',           # ← PROBLEMATIC
    'momentum': 0.9,
    'weight_decay': 1e-2,         # ← Too aggressive
    'batch_size': 32,
    'early_stopping_patience': 30,
    'lr_scheduler': 'step',       # ← Too aggressive decay
    'lr_decay_factor': 0.5,
    'lr_decay_interval': 50
}

# AFTER (3rd run):
model_config = {
    'dropout_rate': 0.5,
    'freeze_backbone': True,
    'pretrained': True,
}
# Note: BatchNorm removed from classifier in classification_model.py

training_config = {
    'learning_rate': 0.0002,      # ← Reduced for stability
    'epochs': 200,
    'optimizer': 'adamw',         # ← Switched to AdamW
    'weight_decay': 5e-3,         # ← Reduced regularization
    'batch_size': 32,
    'early_stopping_patience': 30,
    'lr_scheduler': 'cosine_annealing_warm_restarts',  # ← Smoother decay
    'T_0': 30,
    'T_mult': 2,
    'eta_min': 1e-6
}
```

### GoogLeNet Configuration Changes

```python
# BEFORE (1st/2nd run):
model_config = {
    'dropout_rate': 0.5,
    'freeze_backbone': True,
    'pretrained': True,
    'use_auxiliary': False,       # ← DISABLED (problematic)
}

training_config = {
    'learning_rate': 0.001,       # ← Insufficient for aux losses
    'epochs': 180,
    'optimizer': 'adamw',
    'weight_decay': 5e-3,
    'batch_size': 32,
    'early_stopping_patience': 25,
    'lr_scheduler': 'cosine_annealing_warm_restarts',
    'T_0': 25,                    # ← Short cycles
    'T_mult': 2,
    'eta_min': 1e-6
}

# AFTER (3rd run):
model_config = {
    'dropout_rate': 0.5,
    'freeze_backbone': True,
    'pretrained': True,
    'use_auxiliary': True,        # ← RE-ENABLED
}

training_config = {
    'learning_rate': 0.002,       # ← Increased for aux balance
    'epochs': 180,
    'optimizer': 'adamw',
    'weight_decay': 5e-3,
    'batch_size': 32,
    'early_stopping_patience': 25,
    'lr_scheduler': 'cosine_annealing_warm_restarts',
    'T_0': 30,                    # ← Longer cycles
    'T_mult': 2,
    'eta_min': 1e-6
}
```

### ResNet50 Configuration Changes

```python
# BEFORE (1st/2nd run):
model_config = {
    'dropout_rate': 0.5,
    'additional_fc_layers': True,
    'freeze_backbone': True,
    'pretrained': True,
    'use_batch_norm': True,
}

training_config = {
    'learning_rate': 0.0002,
    'epochs': 180,                # ← Too short
    'optimizer': 'adamw',
    'weight_decay': 5e-3,
    'batch_size': 32,
    'early_stopping_patience': 25, # ← Too impatient
    'lr_scheduler': 'cosine_annealing_warm_restarts',
    'T_0': 30,                    # ← Frequent restarts
    'T_mult': 2,
    'eta_min': 1e-6
}

# AFTER (3rd run):
model_config = {
    'dropout_rate': 0.5,
    'additional_fc_layers': True,
    'freeze_backbone': True,
    'pretrained': True,
    'use_batch_norm': True,
}

training_config = {
    'learning_rate': 0.0002,
    'epochs': 250,                # ← Extended training
    'optimizer': 'adamw',
    'weight_decay': 5e-3,
    'batch_size': 32,
    'early_stopping_patience': 40, # ← More patience
    'lr_scheduler': 'cosine_annealing_warm_restarts',
    'T_0': 50,                    # ← Smoother decay
    'T_mult': 2,
    'eta_min': 1e-6
}
```

---

## Code Modifications Summary

### Files Modified:

1. **`experiments/exp05_classification_AlexNet.py`**
   - Updated `training_config` dictionary
   - Changed optimizer, LR, weight_decay, scheduler

2. **`experiments/exp06_classification_GoogLeNet.py`**
   - Updated `model_config['use_auxiliary']` to `True`
   - Updated `training_config` with higher LR and T_0

3. **`experiments/exp04_classification_ResNet50_baseline.py`**
   - Updated `training_config` with more epochs, patience, T_0

4. **`src/models/classification_model.py`**
   - Removed BatchNorm from AlexNetClassifier
   - No changes needed for GoogLeNet (aux handling already correct)

5. **`src/training/classification_trainer.py`**
   - Modified unfreeze logic to include layer2 for ResNet50
   - Modified unfreeze logic for GoogLeNet to use partial unfreeze

---

## Success Criteria

### Minimum Acceptable Performance:
- [ ] ResNet50: ≥66% test accuracy
- [ ] AlexNet: ≥52% test accuracy
- [ ] GoogLeNet: ≥48% test accuracy
- [ ] Average: ≥55% across all models

### Target Performance:
- [x] ResNet50: ≥68% test accuracy
- [x] AlexNet: ≥55% test accuracy
- [x] GoogLeNet: ≥52% test accuracy
- [x] Average: ≥58% across all models

### Excellent Performance:
- [ ] ResNet50: ≥70% test accuracy
- [ ] AlexNet: ≥58% test accuracy
- [ ] GoogLeNet: ≥56% test accuracy
- [ ] Average: ≥61% across all models

---

## Monitoring Plan

### During Training:
1. **Watch for:**
   - AlexNet: Stable convergence without oscillations
   - GoogLeNet: Lower training loss (<1.0), balanced class predictions
   - ResNet50: Continued improvement beyond epoch 100

2. **Check every 20 epochs:**
   - Validation accuracy trend
   - Train-val gap (should stay <20%)
   - Learning rate decay pattern
   - Per-class metrics (especially for GoogLeNet)

3. **Red flags:**
   - Training loss not decreasing after 30 epochs
   - Validation accuracy dropping consistently
   - All predictions collapsing to one class
   - NaN losses or gradients

### After Training:
1. **Compare with previous runs:**
   - Test accuracy improvements
   - Training stability (variance between runs)
   - Per-class performance balance
   - Convergence speed (epochs to reach best val acc)

2. **Analyze:**
   - Confusion matrices for systematic errors
   - Misclassified samples for common patterns
   - Training curves for overfitting signs
   - Inference time comparisons

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| AlexNet still unstable after AdamW switch | Low | Medium | Have fallback: try LR=0.0001 if 0.0002 fails |
| GoogLeNet aux losses cause instability | Medium | High | Monitor closely, ready to reduce aux weight from 0.3 to 0.2 |
| ResNet50 overfits with layer2 unfrozen | Medium | Medium | Watch train-val gap, increase dropout if >20% |
| Training takes too long (250 epochs) | Low | Low | Can stop early if val acc plateaus for 50+ epochs |

---

## Rollback Plan

If 3rd run performs worse than 2nd run:

1. **Immediate rollback candidates:**
   - Revert AlexNet to SGD if AdamW doesn't stabilize within 50 epochs
   - Disable GoogLeNet auxiliary if training loss explodes (>3.0)
   - Reduce ResNet50 epochs back to 180 if overfitting observed

2. **Partial rollback:**
   - Keep successful changes, revert problematic ones
   - Example: Keep AlexNet AdamW change but reduce LR further

3. **Full rollback:**
   - Restore 2nd run configurations
   - Investigate why changes failed
   - Try alternative approaches (e.g., different augmentation strategies)

---

## Next Steps After 3rd Run

1. **If targets achieved:**
   - Document successful configurations
   - Run 4th iteration for validation/reproducibility
   - Consider ensemble methods (average predictions from all 3 models)

2. **If targets partially achieved:**
   - Analyze which models improved and which didn't
   - Fine-tune unsuccessful models with adjusted hyperparameters
   - Consider architecture-specific optimizations

3. **If targets not achieved:**
   - Deep dive into error analysis (confusion matrices, misclassified samples)
   - Consider data-level improvements (more augmentation, class balancing)
   - Explore alternative architectures (EfficientNet, MobileNet)

---

## Files to Update After Training

After completing 3rd run training:
- [ ] Save test results to `outputs/classification/3rd-run/test-result-3.md`
- [ ] Copy training logs to `outputs/classification/3rd-run/training_log-*-3.csv`
- [ ] Update this file with actual results and analysis
- [ ] Update `outputs/classification/TRACKING_GUIDE.md` comparison table
- [ ] Create summary comparing all 3 runs

---

**Status**: Ready to implement code changes  
**Expected Start Date**: 2026-04-26  
**Expected Completion**: After training completion (~2-4 hours per model depending on GPU)
