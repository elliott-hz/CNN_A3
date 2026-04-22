or# Training Checkpoint & Resume Guide

## 📅 Date: 2026-04-22

This document explains the checkpoint and resume training functionality for detection experiments.

---

## ✅ Feature Status: **ENABLED** ✓

All three detection experiments (exp01, exp02, exp03) now support **automatic checkpoint saving and resume training**.

---

## 🎯 How It Works

### Automatic Checkpoint Saving

During training, YOLOv8 automatically saves checkpoints after each epoch:

```
outputs/exp01_detection_baseline/run_YYYYMMDD_HHMMSS/
├── model/
│   ├── best.pt          # Best performing model (by validation mAP)
│   └── last.pt          # ← CHECKPOINT: Most recent epoch
├── logs/
│   └── training_log.csv
└── figures/
```

**Key Files**:
- **[last.pt](file:///Users/elliott/vscode_workplace/CNN_A3/outputs/exp01_detection_baseline/run_20260422_105702/model/last.pt)**: Contains complete training state (model weights, optimizer state, epoch number, etc.)
- **[best.pt](file:///Users/elliott/vscode_workplace/CNN_A3/outputs/exp01_detection_baseline/run_20260422_105702/model/best_model.pt)**: Best model by validation metrics (used for evaluation)

---

## 🔄 Resume Training Scenarios

### Scenario 1: Training Interrupted (Power Outage, Crash, Manual Stop)

**What happens**:
1. Training stops at epoch N (e.g., epoch 35 of 50)
2. [last.pt](file:///Users/elliott/vscode_workplace/CNN_A3/outputs/exp01_detection_baseline/run_20260422_105702/model/last.pt) checkpoint is saved in the output directory
3. You restart the experiment script

**What the code does**:
```python
# DetectionTrainer automatically checks for last.pt
last_checkpoint = model_dir / "last.pt"
resume_training = last_checkpoint.exists()

if resume_training:
    print("✓ Found previous checkpoint")
    print("  Resuming training from last checkpoint...")
    
# YOLOv8 resumes from where it left off
results = model.train_model(
    ...,
    resume=resume_training  # ← This enables resume
)
```

**Result**:
- Training continues from epoch 36 (not from epoch 1)
- All progress is preserved
- No wasted computation

---

### Scenario 2: Intentional Pause and Resume

**Use case**: You want to pause training overnight and resume tomorrow.

**Steps**:
1. Let training run until you need to stop
2. Press `Ctrl+C` to gracefully stop (YOLOv8 will save [last.pt](file:///Users/elliott/vscode_workplace/CNN_A3/outputs/exp01_detection_baseline/run_20260422_105702/model/last.pt))
3. Next day, run the same experiment script again
4. Training automatically resumes from the last checkpoint

---

### Scenario 3: Fresh Start (No Resume)

**When this happens**:
- First time running an experiment
- Output directory doesn't exist yet
- [last.pt](file:///Users/elliott/vscode_workplace/CNN_A3/outputs/exp01_detection_baseline/run_20260422_105702/model/last.pt) file doesn't exist

**Behavior**:
```
  Starting fresh training...
```

Training starts from epoch 1 as normal.

---

## 📋 Experiment-Specific Details

### exp01 (Baseline - YOLOv8-m)
- **Total epochs**: 50
- **If interrupted at epoch 30**: Resumes from epoch 31
- **Time saved**: ~60% of remaining training time

### exp02 (Modified V1 - YOLOv8-l, 1280px)
- **Total epochs**: 60
- **If interrupted at epoch 45**: Resumes from epoch 46
- **Time saved**: ~75% of remaining training time
- **⚠️ Important**: This is the longest training run - resume is CRITICAL!

### exp03 (Modified V2 - YOLOv8-s)
- **Total epochs**: 40
- **If interrupted at epoch 25**: Resumes from epoch 26
- **Time saved**: ~62.5% of remaining training time

---

## 🔍 How to Verify Resume is Working

### Check Console Output

**Fresh Training**:
```
[Step 4/5] Training model...

  Starting fresh training...
  
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
  1/50      2.5G      1.234      0.567      0.890         45        640
  2/50      2.5G      1.123      0.534      0.845         45        640
  ...
```

**Resume Training**:
```
[Step 4/5] Training model...

✓ Found previous checkpoint: outputs/exp01_*/run_*/model/last.pt
  Resuming training from last checkpoint...
  
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
 31/50      2.5G      0.789      0.345      0.678         45        640
 32/50      2.5G      0.756      0.332      0.665         45        640
  ...
```

Notice the epoch numbers continue from where you left off!

---

## ⚙️ Technical Implementation

### What's Saved in [last.pt](file:///Users/elliott/vscode_workplace/CNN_A3/outputs/exp01_detection_baseline/run_20260422_105702/model/last.pt)?

The checkpoint contains:
- ✅ Model weights (all layers)
- ✅ Optimizer state (momentum, learning rate schedule)
- ✅ Current epoch number
- ✅ Training history (losses, metrics)
- ✅ Random number generator states (for reproducibility)

### YOLOv8 Resume Mechanism

When `resume=True`:
1. YOLOv8 loads [last.pt](file:///Users/elliott/vscode_workplace/CNN_A3/outputs/exp01_detection_baseline/run_20260422_105702/model/last.pt)
2. Restores model weights, optimizer state, and epoch counter
3. Continues training from `current_epoch + 1`
4. Preserves learning rate schedule position

**Reference**: [YOLOv8 Documentation - Resume Training](https://docs.ultralytics.com/modes/train/#resume-training)

---

## 💡 Best Practices

### 1. Don't Delete Output Directories Mid-Training

❌ **Bad**: Deleting `outputs/exp01_*/run_*/` while training
✅ **Good**: Let YOLOv8 manage the directory

### 2. Use Graceful Interruption When Possible

❌ **Bad**: Force kill (`kill -9`)
✅ **Good**: `Ctrl+C` (allows checkpoint save)

### 3. Verify Checkpoint Exists Before Restarting

```bash
# Check if checkpoint exists
ls -lh outputs/exp01_detection_baseline/run_*/model/last.pt

# If file exists, you can safely resume
python experiments/exp01_detection_baseline.py
```

### 4. Monitor Disk Space

Checkpoints can be large:
- exp01 (YOLOv8-m): ~50 MB per checkpoint
- exp02 (YOLOv8-l): ~200 MB per checkpoint
- exp03 (YOLOv8-s): ~20 MB per checkpoint

Ensure you have enough disk space in the `outputs/` directory.

---

## 🆘 Troubleshooting

### Problem: Training Doesn't Resume Automatically

**Check**:
1. Does [last.pt](file:///Users/elliott/vscode_workplace/CNN_A3/outputs/exp01_detection_baseline/run_20260422_105702/model/last.pt) exist in the output directory?
   ```bash
   ls outputs/exp01_*/run_*/model/last.pt
   ```

2. Are you running the **same experiment script**?
   - exp01 resumes only exp01 runs
   - Each experiment has its own output directory

3. Is the output directory structure intact?
   ```
   outputs/exp01_detection_baseline/
   └── run_YYYYMMDD_HHMMSS/  ← Must exist with last.pt inside
       └── model/
           └── last.pt
   ```

**Solution**: If [last.pt](file:///Users/elliott/vscode_workplace/CNN_A3/outputs/exp01_detection_baseline/run_20260422_105702/model/last.pt) exists but resume doesn't work, there may be a code issue. Check the console for error messages.

---

### Problem: Want to Start Fresh (Ignore Checkpoint)

**Solution**: Delete the checkpoint file before running:
```bash
# Remove last.pt to force fresh start
rm outputs/exp01_detection_baseline/run_*/model/last.pt

# Now run experiment - it will start from epoch 1
python experiments/exp01_detection_baseline.py
```

Or delete the entire run directory:
```bash
# Remove entire run
rm -rf outputs/exp01_detection_baseline/run_YYYYMMDD_HHMMSS/

# Run creates new timestamped directory
python experiments/exp01_detection_baseline.py
```

---

### Problem: Corrupted Checkpoint

**Symptoms**: Error loading [last.pt](file:///Users/elliott/vscode_workplace/CNN_A3/outputs/exp01_detection_baseline/run_20260422_105702/model/last.pt), training fails to resume

**Solution**:
1. Delete corrupted checkpoint:
   ```bash
   rm outputs/exp01_*/run_*/model/last.pt
   ```

2. Restart training from scratch (unfortunately, progress is lost)

**Prevention**: Ensure sufficient disk space and avoid interrupting during checkpoint save (rare).

---

## 📊 Time Savings Example

### Scenario: exp02 (Large Model, 60 Epochs)

**Assumptions**:
- Each epoch takes ~5 minutes on T4 GPU
- Total training time: 60 × 5 = 300 minutes (5 hours)

**Case 1: Crash at Epoch 50**
- Without resume: Restart from 0 → 300 minutes total
- With resume: Continue from 50 → 10 × 5 = 50 minutes additional
- **Time saved**: 250 minutes (4.2 hours) ⏱️

**Case 2: Crash at Epoch 30**
- Without resume: Restart from 0 → 300 minutes total
- With resume: Continue from 30 → 30 × 5 = 150 minutes additional
- **Time saved**: 150 minutes (2.5 hours) ⏱️

**ROI**: The resume feature pays for itself after just one interruption!

---

## 🔗 Related Documentation

- [Detection Experiments Changes Applied](./detection_experiments_changes_applied.md)
- [Detection Experiments Code Review](./detection_experiments_review.md)
- [Data Preprocessing Pipeline](./data_preprocessing.md)

---

## 📝 Summary

| Feature | Status | Notes |
|---------|--------|-------|
| **Automatic Checkpoint Save** | ✅ Enabled | After each epoch |
| **Resume Training** | ✅ Enabled | Detects [last.pt](file:///Users/elliott/vscode_workplace/CNN_A3/outputs/exp01_detection_baseline/run_20260422_105702/model/last.pt) automatically |
| **Gradient Accumulation** | ❌ Not Available | Set to 1 for all experiments |
| **Manual Resume Control** | ⚠️ Limited | Delete [last.pt](file:///Users/elliott/vscode_workplace/CNN_A3/outputs/exp01_detection_baseline/run_20260422_105702/model/last.pt) to force fresh start |

**Bottom Line**: Your training is now **resilient to interruptions**! If your computer crashes or you need to stop training, just rerun the experiment script and it will continue from where it left off. 🎉

---

**Last Updated**: 2026-04-22  
**Feature Status**: ✅ Fully Implemented and Tested
