# Detection Experiments Code Review (exp01, exp02, exp03)

## 📋 Overview

This document provides a comprehensive code review of the three dog face detection experiments using YOLOv8. All three experiments share the same architecture but differ in model configurations and training hyperparameters.

---

## 🏗️ Architecture Summary

### Experiment Structure

All three experiments follow the same 5-step pipeline:

```
1. Verify processed datasets ✓
2. Load dataset configuration (dataset.yaml) ✓
3. Initialize model and trainer ✓
4. Train model → outputs/ ✗ (Issues found)
5. Evaluate on test set → outputs/ ✗ (Issues found)
```

### Model Variants

| Experiment | Backbone | Input Size | Confidence | NMS IoU | Purpose |
|-----------|----------|------------|------------|---------|---------|
| **exp01** (Baseline) | `m` (Medium) | 640×640 | 0.5 | 0.45 | Standard baseline |
| **exp02** (Modified V1) | `l` (Large) | 1280×1280 | 0.6 | 0.50 | High accuracy |
| **exp03** (Modified V2) | `s` (Small) | 640×640 | 0.4 | 0.40 | Fast inference |

### Training Configurations

| Parameter | exp01 (Baseline) | exp02 (V1 - Large) | exp03 (V2 - Small) |
|-----------|------------------|---------------------|---------------------|
| Learning Rate | 0.001 | 0.0005 | 0.002 |
| Batch Size | 16 | 8 | 32 |
| Epochs | 50 | 60 | 40 |
| Optimizer | Adam | AdamW | SGD |
| Weight Decay | 1e-4 | 1e-4 | 1e-4 |
| Early Stopping | 10 epochs | 12 epochs | 8 epochs |
| Gradient Accumulation | 1 | 2 | 1 |
| Warmup Epochs | 5 | 5 | 3 |
| Scheduler | Cosine | Cosine | Step |
| AMP | ✅ Yes | ✅ Yes | ✅ Yes |

---

## ⚠️ Issues Found

### 🔴 CRITICAL ISSUES

#### 1. **Training Configuration Not Fully Utilized** ❌

**Problem**: The `DetectionTrainer` receives a comprehensive `training_config` dictionary, but most parameters are **ignored** during actual training.

**Location**: 
- [`src/training/detection_trainer.py`](../src/training/detection_trainer.py) lines 97-113
- [`src/models/detection_model.py`](../src/models/detection_model.py) lines 96-112

**Details**:
```python
# In DetectionTrainer.train() - only these params are passed:
results = model.train_model(
    data=train_data,
    epochs=self.epochs,              # ✅ Used
    imgsz=self.model_config.get('input_size', 640),  # ✅ Used
    batch=self.batch_size,           # ✅ Used
    lr0=self.lr,                     # ✅ Used
    weight_decay=self.weight_decay,  # ✅ Used
    patience=self.patience,          # ✅ Used
    amp=self.use_amp,                # ✅ Used
    # ❌ MISSING: optimizer_type ('adam' vs 'adamw' vs 'sgd')
    # ❌ MISSING: scheduler_type ('cosine' vs 'step')
    # ❌ MISSING: warmup_epochs
    # ❌ MISSING: gradient_accumulation_steps
    name="detection_training",
    project=str(output_dir),
    exist_ok=True
)
```

**Impact**:
- exp02 specifies `optimizer='adamw'` but YOLOv8 uses default (likely SGD)
- exp03 specifies `scheduler='step'` but YOLOv8 uses default cosine annealing
- exp02 specifies `gradient_accumulation_steps=2` but it's not implemented
- exp02/exp03 warmup settings are ignored

**Why This Matters**:
- Different optimizers have different convergence behaviors
- AdamW typically works better with larger models (exp02)
- Gradient accumulation allows effective larger batch sizes on limited GPU memory
- Scheduler choice affects final model performance

**Fix Required**: Pass additional parameters to YOLOv8's train method or implement custom training loop.

---

#### 2. **Evaluation Uses Wrong Model Weights** ❌

**Problem**: After training, the evaluation step uses the in-memory `model` object, which may not be the **best performing model**.

**Location**: 
- All three experiment scripts (lines 103-120)

**Current Flow**:
```python
# Step 4: Training
results = trainer.train(model=model, ...)  # Trains and saves best_model.pt

# Step 5: Evaluation  
metrics = evaluator.evaluate(model=model, ...)  # ❌ Uses current model state
```

**Issue**: 
- The trainer saves `best_model.pt` to disk based on validation metrics
- But the `model` object in memory might be from the last epoch, not the best epoch
- YOLOv8's internal model state after training may not automatically revert to best weights

**Impact**: Evaluation results may be worse than actual best model performance.

**Fix Required**: Reload the best model before evaluation:
```python
# After training, reload best weights
best_model_path = Path(output_dir) / "model" / "best_model.pt"
model = YOLOv8Detector(model_config)  # Reinitialize
model.model = YOLO(str(best_model_path))  # Load best weights
```

---

### 🟡 MODERATE ISSUES

#### 3. **Gradient Accumulation Not Implemented** ⚠️

**Problem**: exp02 configures `gradient_accumulation_steps=2`, but this feature is not implemented anywhere in the codebase.

**Location**: 
- [`src/training/detection_trainer.py`](../src/training/detection_trainer.py) - parameter accepted but never used
- [`src/models/detection_model.py`](../src/models/detection_model.py) - `train_model()` doesn't support it

**Why It Matters**:
- exp02 uses batch_size=8 with a large model (YOLOv8-l at 1280px)
- Without gradient accumulation, effective batch size is only 8
- With accumulation=2, effective batch size would be 16 (better for stability)

**Implementation Needed**:
```python
# Would require custom training loop since YOLOv8's built-in train() 
# doesn't expose gradient accumulation control at this level
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / self.grad_accum_steps
    loss.backward()
    
    if (i + 1) % self.grad_accum_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**Current Status**: Parameter is defined but has **zero effect**.

---

#### 4. **Learning Rate Scheduler Configuration Ignored** ⚠️

**Problem**: All three experiments specify different schedulers (`cosine`, `step`), but YOLOv8 uses its own default scheduler.

**Location**: 
- exp01: `'scheduler': 'cosine'` (line 66)
- exp02: `'scheduler': 'cosine'` (line 72)
- exp03: `'scheduler': 'step'` (line 69)

**YOLOv8 Default Behavior**:
- YOLOv8 automatically uses cosine learning rate decay with warmup
- The `scheduler` parameter in training_config is never passed to YOLO
- Cannot easily switch to step scheduler without custom implementation

**Impact**:
- exp03 expects step decay but gets cosine decay
- May affect convergence behavior and final performance

**Note**: This is less critical since YOLOv8's default cosine scheduler is generally good, but it means the configuration is misleading.

---

#### 5. **Logging Step Number Error in exp02** 🐛

**Problem**: Minor typo in logging message.

**Location**: [`experiments/exp02_detection_modified_v1.py`](../experiments/exp02_detection_modified_v1.py) line 58

```python
# Current (WRONG):
logger.info("\n[Step 4/5] Initializing model and trainer...")

# Should be:
logger.info("\n[Step 3/5] Initializing model and trainer...")
```

**Impact**: Cosmetic only, doesn't affect functionality.

---

### 🟢 MINOR ISSUES / IMPROVEMENTS

#### 6. **No Validation During Training Visibility** 💡

**Problem**: While training runs, there's no real-time logging of validation metrics to console.

**Current Behavior**:
- YOLOv8 prints its own progress bar
- But the experiment script doesn't log intermediate validation results
- Users must wait until training completes to see any metrics

**Improvement Suggestion**:
Add callback or periodic logging to show validation mAP every N epochs.

---

#### 7. **Hardcoded Output Directory Names** 💡

**Problem**: The trainer hardcodes subdirectory names.

**Location**: [`src/training/detection_trainer.py`](../src/training/detection_trainer.py) lines 83-88

```python
model_dir = Path(output_dir) / "model"
log_dir = Path(output_dir) / "logs"
figures_dir = Path(output_dir) / "figures"
```

**Issue**: Works fine, but if you want to customize output structure, you'd need to modify the trainer.

**Status**: Not a bug, just a design observation.

---

#### 8. **Exception Handling Could Be More Specific** 💡

**Problem**: Generic exception catching without specific error types.

**Location**: All three experiments (lines 95-100, 115-120)

```python
except Exception as e:
    logger.error(f"Training failed: {e}")
    import traceback
    traceback.print_exc()
    return  # ⚠️ Returns None implicitly
```

**Issue**: 
- Function doesn't return anything on error (implicit `None`)
- Caller doesn't check return value
- Script continues to evaluation even if training failed (though evaluation will also fail)

**Improvement**: Use `sys.exit(1)` instead of `return` to stop execution on critical errors.

---

## ✅ What Works Well

### 1. **Modular Design** ✓
- Clear separation: Model ↔ Trainer ↔ Evaluator
- Easy to swap components
- Configuration-driven approach

### 2. **Dataset Verification** ✓
- [`verify_processed_datasets()`](../src/data_processing/processed_datasets_verify.py) checks all requirements
- Prevents running experiments on unprocessed data
- Clear error messages guide users

### 3. **Relative Path Usage** ✓
- `dataset.yaml` uses relative paths (`path: .`)
- Portable across different machines
- Follows YOLOv8 best practices

### 4. **Timestamped Output Directories** ✓
- Each run creates unique directory: `outputs/exp01_*/run_YYYYMMDD_HHMMSS/`
- No overwriting of previous results
- Easy to compare multiple runs

### 5. **Configuration Management** ✓
- Model configs defined in [`detection_model.py`](../src/models/detection_model.py)
- Easy to modify and extend
- Centralized configuration source

---

## 🔧 Recommended Fixes (Priority Order)

### Priority 1: CRITICAL

#### Fix 1: Reload Best Model Before Evaluation

**Files to modify**: All three experiment scripts

**Change**:
```python
# After training, before evaluation:
logger.info("Reloading best model weights for evaluation...")
best_model_path = output_dir / "model" / "best_model.pt"
if best_model_path.exists():
    # Create new model instance with best weights
    from ultralytics import YOLO
    best_yolo_model = YOLO(str(best_model_path))
    model.model = best_yolo_model  # Replace internal model
    logger.info(f"Loaded best model from: {best_model_path}")
else:
    logger.warning("Best model file not found, using current model state")
```

---

#### Fix 2: Pass Missing Training Parameters to YOLOv8

**File**: [`src/models/detection_model.py`](../src/models/detection_model.py)

**Current** (line 96-112):
```python
def train_model(self, data, epochs=100, imgsz=None, **kwargs):
    train_args = {
        'data': data,
        'epochs': epochs,
        'imgsz': imgsz,
        'conf': self.confidence_threshold,
        'iou': self.nms_iou_threshold,
    }
    train_args.update(kwargs)
    results = self.model.train(**train_args)
    return results
```

**Improved**:
```python
def train_model(self, data, epochs=100, imgsz=None, 
                optimizer=None, scheduler=None, warmup_epochs=None,
                **kwargs):
    train_args = {
        'data': data,
        'epochs': epochs,
        'imgsz': imgsz,
        'conf': self.confidence_threshold,
        'iou': self.nms_iou_threshold,
    }
    
    # Add optional parameters if provided
    if optimizer is not None:
        train_args['optimizer'] = optimizer  # 'SGD', 'Adam', 'AdamW'
    if scheduler is not None:
        # Map our scheduler names to YOLOv8 format
        scheduler_map = {
            'cosine': 'CosineLR',
            'step': 'StepLR',
            'reduce_on_plateau': 'ReduceLROnPlateau'
        }
        train_args['lr_scheduler'] = scheduler_map.get(scheduler, scheduler)
    if warmup_epochs is not None:
        train_args['warmup_epochs'] = warmup_epochs
    
    train_args.update(kwargs)
    results = self.model.train(**train_args)
    return results
```

**Then update trainer** ([`detection_trainer.py`](../src/training/detection_trainer.py) line 97):
```python
results = model.train_model(
    data=train_data,
    epochs=self.epochs,
    imgsz=self.model_config.get('input_size', 640),
    batch=self.batch_size,
    lr0=self.lr,
    weight_decay=self.weight_decay,
    patience=self.patience,
    amp=self.use_amp,
    optimizer=self.optimizer_type.upper(),  # Pass optimizer
    scheduler=self.scheduler_type,          # Pass scheduler
    warmup_epochs=self.warmup_epochs,       # Pass warmup
    name="detection_training",
    project=str(output_dir),
    exist_ok=True
)
```

**Note**: Gradient accumulation still cannot be easily implemented with YOLOv8's built-in training. Would require custom training loop.

---

### Priority 2: MODERATE

#### Fix 3: Correct Logging Step Number

**File**: [`experiments/exp02_detection_modified_v1.py`](../experiments/exp02_detection_modified_v1.py) line 58

**Change**:
```python
# FROM:
logger.info("\n[Step 4/5] Initializing model and trainer...")

# TO:
logger.info("\n[Step 3/5] Initializing model and trainer...")
```

---

#### Fix 4: Improve Error Handling

**Files**: All three experiment scripts

**Change**:
```python
# FROM:
except Exception as e:
    logger.error(f"Training failed: {e}")
    import traceback
    traceback.print_exc()
    return  # Implicit None

# TO:
except Exception as e:
    logger.error(f"Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)  # Explicit exit with error code
```

---

### Priority 3: ENHANCEMENTS (Optional)

#### Enhancement 1: Add Training Progress Callback

Would allow real-time logging of validation metrics during training.

#### Enhancement 2: Document YOLOv8 Parameter Mapping

Create a reference table showing which training_config parameters map to YOLOv8 arguments.

#### Enhancement 3: Add Memory Monitoring

Log GPU memory usage during training to help tune batch sizes.

---

## 📊 Comparison Table: Intended vs Actual Behavior

| Feature | exp01 Intended | exp01 Actual | exp02 Intended | exp02 Actual | exp03 Intended | exp03 Actual |
|---------|---------------|--------------|---------------|--------------|---------------|--------------|
| **Backbone** | YOLOv8-m | ✅ YOLOv8-m | YOLOv8-l | ✅ YOLOv8-l | YOLOv8-s | ✅ YOLOv8-s |
| **Input Size** | 640px | ✅ 640px | 1280px | ✅ 1280px | 640px | ✅ 640px |
| **Optimizer** | Adam | ❌ Default (SGD) | AdamW | ❌ Default (SGD) | SGD | ❌ Default (SGD) |
| **Scheduler** | Cosine | ✅ Cosine (default) | Cosine | ✅ Cosine (default) | Step | ❌ Cosine (default) |
| **Batch Size** | 16 | ✅ 16 | 8 | ✅ 8 | 32 | ✅ 32 |
| **LR** | 0.001 | ✅ 0.001 | 0.0005 | ✅ 0.0005 | 0.002 | ✅ 0.002 |
| **Grad Accum** | 1 | ✅ N/A | 2 | ❌ Not implemented | 1 | ✅ N/A |
| **Warmup** | 5 epochs | ✅ 5 epochs | 5 epochs | ✅ 5 epochs | 3 epochs | ❌ Default (3) |
| **Eval Model** | Best weights | ❌ Last epoch | Best weights | ❌ Last epoch | Best weights | ❌ Last epoch |

---

## 🎯 Summary

### Overall Assessment: **B+ (Good, with Important Issues)**

**Strengths**:
- ✅ Clean modular architecture
- ✅ Good configuration management
- ✅ Proper dataset verification
- ✅ Timestamped outputs prevent overwrites
- ✅ Relative paths for portability

**Critical Issues**:
- ❌ Training configurations partially ignored (optimizer, scheduler, warmup)
- ❌ Evaluation may not use best model weights
- ❌ Gradient accumulation not implemented

**Recommendation**: 
1. **Must fix**: Reload best model before evaluation (affects all experiments)
2. **Should fix**: Pass optimizer/scheduler parameters to YOLOv8
3. **Nice to have**: Implement gradient accumulation or document limitation

### Impact on Results

The issues mean that:
- **exp02** (Large model) is likely underperforming because it's using SGD instead of AdamW, and not using gradient accumulation
- **exp03** (Small model) is using cosine scheduler instead of step decay as intended
- All experiments may report slightly worse metrics than achievable because evaluation might not use the best checkpoint

However, the **relative comparison** between experiments is still valid since they all have the same issues. The absolute performance numbers could be improved with the fixes.

---

## 📝 Action Items

- [ ] **Fix 1**: Reload best model weights before evaluation (all 3 experiments)
- [ ] **Fix 2**: Update `train_model()` to accept and pass optimizer/scheduler/warmup params
- [ ] **Fix 3**: Correct step numbering in exp02
- [ ] **Fix 4**: Improve error handling with `sys.exit(1)`
- [ ] **Document**: Add note about gradient accumulation limitation
- [ ] **Test**: Re-run one experiment after fixes to verify improvements

---

**Last Updated**: 2026-04-22  
**Reviewed By**: AI Code Assistant  
**Status**: Review Complete - Fixes Identified
