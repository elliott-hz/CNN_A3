# Overfitting Fix for Classification Baseline (Exp04)

## Problem Analysis

The training logs show **severe overfitting**:
- **Training Accuracy**: 99.15% (Epoch 21)
- **Validation Accuracy**: ~67-68% (plateaued)
- **Gap**: >30% difference indicates serious overfitting

### Root Causes Identified

1. **No Data Augmentation**: Images were loaded directly without any augmentation transforms
2. **Early Stopping Disabled**: `early_stopping_patience` was set to `0`, meaning training continued regardless of validation performance
3. **Learning Rate Too High**: Initial LR of 0.001 with fine-tuning at 0.0001 caused rapid overfitting
4. **Insufficient Regularization**: Weight decay of 1e-4 was too weak

## Changes Made

### 1. Updated Training Configuration (`exp04_classification_baseline.py`)

```python
training_config = {
    'learning_rate': 0.0005,        # Reduced from 0.001 (50% reduction)
    'batch_size': 32,
    'epochs': 120,
    'optimizer': 'adam',
    'weight_decay': 5e-4,           # Increased from 1e-4 (5x stronger regularization)
    'early_stopping_patience': 15,  # Changed from 0 to enable early stopping
    'use_amp': True,
    'gradient_accumulation_steps': 1,
    'label_smoothing': 0.1,
    'class_weighting': True
}
```

**Rationale:**
- Lower learning rate allows more gradual learning and better generalization
- Higher weight decay penalizes large weights more strongly
- Early stopping prevents training beyond the point where validation performance degrades

### 2. Added Memory-Efficient Data Augmentation (`classification_trainer.py`)

#### Problem with Initial Implementation
The first version pre-computed all augmented images upfront, causing:
- ❌ **High memory usage**: All augmented images stored in RAM simultaneously
- ❌ **Slow data loading**: Pre-processing all 6527+ training images before training starts
- ❌ **Memory overflow**: Exceeded 16GB RAM on typical systems

#### Solution: On-the-Fly Augmentation
Created a custom `AugmentedDataset` class that applies transforms **only when needed**:

```python
class AugmentedDataset(Dataset):
    """Applies augmentation during __getitem__, not upfront"""
    
    def __getitem__(self, idx):
        img = self.X[idx]
        pil_img = Image.fromarray(img.astype('uint8'))
        
        # Apply random augmentation each time this sample is accessed
        if self.transform:
            pil_img = self.transform(pil_img)
        
        return transforms.ToTensor()(pil_img), self.y[idx]
```

**Benefits:**
- ✅ **Low memory footprint**: Only augments images in the current batch
- ✅ **Fast startup**: No pre-processing delay before training begins
- ✅ **True randomness**: Each epoch sees different augmentations of the same image
- ✅ **Scalable**: Works with datasets of any size without memory issues

**Augmentation Pipeline:**
```python
transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),      # 50% chance of horizontal flip
    transforms.RandomRotation(degrees=15),       # ±15° rotation
    transforms.ColorJitter(                      # Color variation
        brightness=0.2, contrast=0.2, 
        saturation=0.2, hue=0.1
    ),
    transforms.RandomAffine(                     # Spatial transformation
        degrees=0, translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    ),
])
```

## Expected Improvements

With these changes, you should see:

1. **Reduced Overfitting Gap**: Train-val accuracy gap should decrease from 30%+ to <10%
2. **Better Validation Performance**: Val accuracy should improve from ~68% to 75-80%+
3. **Earlier Convergence**: Early stopping will halt training when val performance plateaus
4. **More Robust Model**: Data augmentation creates a model that generalizes better to unseen images
5. **Efficient Memory Usage**: Memory consumption stays constant regardless of dataset size

## How to Re-run Training

Stop the current training process and restart with the updated code:

```bash
cd ~/CNN_A3
python experiments/exp04_classification_baseline.py
```

The training will now:
- Apply data augmentation **on-the-fly** during training (not validation/test)
- Use lower learning rate for more stable training
- Stop automatically if validation accuracy doesn't improve for 15 epochs
- Apply stronger L2 regularization
- **Start immediately** without long data loading delays
- **Use minimal memory** (only stores original images, not augmented versions)

## Monitoring Training

Watch for these indicators of healthy training:

✅ **Good Signs:**
- Train and val accuracy increase together
- Gap between train/val stays <10%
- Val loss decreases steadily
- Early stopping triggers before epoch 120
- Fast startup (no long preprocessing delay)
- Stable memory usage throughout training

❌ **Warning Signs:**
- Train accuracy >> val accuracy (>15% gap)
- Val loss starts increasing while train loss decreases
- No improvement in val accuracy for many epochs
- Memory usage keeps growing (memory leak)

## Additional Recommendations (If Still Overfitting)

If overfitting persists after these changes:

1. **Increase Dropout**: Modify `BASELINE_CLASSIFICATION_CONFIG` to use `dropout_rate: 0.7`
2. **Reduce Model Complexity**: Try using fewer trainable layers during fine-tuning
3. **Add More Augmentation**: Consider `transforms.RandomErasing()` or `transforms.GaussianBlur()`
4. **Use Label Smoothing**: Already enabled at 0.1, could increase to 0.2
5. **Reduce Batch Size**: Smaller batches (16) can act as implicit regularization

## Technical Notes

- **On-the-fly augmentation**: Applied during `__getitem__`, so each epoch sees different augmented versions
- **Validation/test sets**: Use **no augmentation** to get consistent evaluation metrics
- **PIL Image conversion**: Ensures compatibility with torchvision transforms
- **Augmentation scope**: Only applies when `train=True` in dataloader creation
- **Memory efficiency**: Original images stored once; augmented versions created temporarily per batch
- **Worker processes**: Reduced `num_workers` from 4 to 2 to further reduce memory overhead