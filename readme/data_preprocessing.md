# Data Preprocessing Pipeline - Complete Analysis

## 📋 Overview

This document provides a comprehensive analysis of the data preprocessing pipeline for the CNN_A3 project, which handles two distinct computer vision tasks: **Dog Face Detection** and **Dog Emotion Classification**.

---

## 🗂️ Directory Structure

```
data/
├── raw/                          # Original raw datasets (NOT committed to Git)
│   ├── detection_dataset/        # Detection task raw data
│   │   ├── train_img/            # 5,924 training images
│   │   ├── train_label/          # 5,924 YOLO format labels
│   │   ├── val_img/              # 230 validation images
│   │   └── val_label/            # 230 validation labels
│   └── emotion_dataset/          # Classification task raw data
│       ├── alert/                # 1,865 images
│       ├── angry/                # 1,865 images
│       ├── frown/                # 1,865 images
│       ├── happy/                # 1,865 images
│       └── relax/                # 1,865 images
│
├── processed/                    # Preprocessed data (NOT committed to Git)
│   └── detection/                # Processed detection dataset
│       ├── train/                # 5,331 preprocessed images (640x640)
│       ├── valid/                # 230 preprocessed images (640x640)
│       ├── test/                 # 593 preprocessed images (640x640)
│       ├── annotations/          # Adjusted YOLO format labels
│       │   ├── train/
│       │   ├── valid/
│       │   └── test/
│       ├── dataset.yaml          # ⭐ YOLOv8 configuration file
│       └── metadata.json         # Processing metadata
│
└── splitting/                    # Split metadata (committed to Git)
    ├── detection_split/          # Detection split info (reference only)
    │   ├── train_split.json      # Original image paths
    │   ├── val_split.json
    │   ├── test_split.json
    │   └── metadata.json
    └── emotion_split/            # ⭐ Emotion split info (CRITICAL)
        ├── train_split.json      # 5,221 image paths + labels
        ├── val_split.json        # 1,492 image paths + labels
        ├── test_split.json       # 747 image paths + labels
        └── metadata.json
```

---

## 🔍 Dataset Details

### 1. Detection Dataset (`data/raw/detection_dataset/`)

#### **Purpose**
Dog face detection using YOLOv8 algorithm.

#### **Raw Format**
- **Images**: JPEG format in `train_img/` and `val_img/` folders
- **Labels**: YOLO format text files (`.txt`) with structure:
  ```
  <class_id> <x_center> <y_center> <width> <height>
  ```
  - All coordinates normalized to [0, 1]
  - Single class: `dog_face` (class_id = 0)

#### **Original Statistics**
- Training set: 5,924 images
- Validation set: 230 images
- **Total**: 6,154 images

---

### 2. Emotion Dataset (`data/raw/emotion_dataset/`)

#### **Purpose**
Dog emotion classification into 5 categories.

#### **Raw Format**
- **Structure**: Folder-based classification format
- **Classes**: 5 emotion types
  - `alert`: Alert/watchful state
  - `angry`: Angry/aggressive state
  - `frown`: Frowning/concerned state
  - `happy`: Happy/content state
  - `relax`: Relaxed/calm state
- **Images**: JPEG format, varying dimensions

#### **Statistics**
- Each class: 1,865 images
- **Total**: 9,325 images
- **Class Balance**: Perfectly balanced

---

## ⚙️ Preprocessing Pipeline

### Detection Dataset Processing

**Script**: [`src/data_processing/detection_preprocessor.py`](../src/data_processing/detection_preprocessor.py)

#### **Processing Steps**

1. **Data Loading (Streaming Mode)**
   - Loads train and val splits sequentially to minimize memory usage
   - Reads YOLO format annotations from `.txt` files
   - Stores image paths and bounding box lists

2. **Dataset Splitting (Option A Strategy)**
   - **Preserves original validation set**: 230 images kept as-is
   - **Splits original training set**: 90% train / 10% test
     - Train: 5,331 images (from original train_img)
     - Test: 593 images (from original train_img)
   - **Rationale**: Respects professional original train/val split while creating proper test set

3. **Letterbox Resize** (Key Feature)
   - **Target Size**: 640×640 pixels
   - **Method**: Preserves aspect ratio with padding
   - **Process**:
     1. Calculate scale ratio: `min(640/W, 640/H)`
     2. Resize image proportionally
     3. Add gray padding (RGB: 114, 114, 114) to reach 640×640
     4. Adjust bounding box coordinates:
        - Scale by resize ratio
        - Offset by padding amount
        - Normalize to 640×640 canvas
   
   **Why Letterbox?**
   - ✅ Prevents image distortion (no stretching/squashing)
   - ✅ Maintains object proportions for accurate detection
   - ✅ Industry standard for YOLO models

4. **File Storage**
   - Saves processed images as individual JPEG files
   - Generates corresponding YOLO annotation files
   - Uses batch processing (100 images/batch) with garbage collection
   - Memory-efficient: avoids loading all data into RAM simultaneously

5. **YOLOv8 Configuration Generation**
   - Creates `dataset.yaml` with **relative paths** for portability
   - Configuration:
     ```yaml
     path: .              # Current directory (where dataset.yaml is located)
     train: train         # Relative path to train images
     val: valid           # Relative path to validation images
     test: test           # Relative path to test images
     nc: 1                # Number of classes
     names: [dog_face]    # Class names
     ```

#### **Output Location**
`data/processed/detection/`

#### **Memory Management**
- Streaming data loading (process train → clear → process val)
- Batch processing with `batch_size=100`
- Explicit `gc.collect()` calls after each batch
- Temporary file cleanup for failed images

---

### Emotion Dataset Processing

**Script**: [`src/data_processing/emotion_preprocessor.py`](../src/data_processing/emotion_preprocessor.py)

#### **Processing Steps**

1. **Data Loading**
   - Scans emotion class folders (`alert/`, `angry/`, etc.)
   - Collects all image paths (supports `.jpg`, `.jpeg`, `.png`)
   - Assigns class labels based on folder name

2. **Stratified Splitting**
   - **Split Ratio**: 70% train / 20% val / 10% test
   - **Method**: Stratified sampling preserves class distribution
   - **Results**:
     - Train: 5,221 images (~70%)
     - Val: 1,492 images (~20%)
     - Test: 747 images (~10%)
   - **Class Balance**: Maintained across all splits

3. **Metadata Saving**
   - Saves split information as JSON files
   - Each JSON contains:
     - Image paths (absolute or relative)
     - Corresponding class labels
   - No image preprocessing or transformation

#### **Key Difference from Detection**
❌ **NO preprocessing applied**  
❌ **NO image resizing**  
❌ **NO file copying**  

✅ Images remain in original location (`data/raw/emotion_dataset/`)  
✅ Only split metadata is saved to `data/splitting/emotion_split/`  
✅ Images loaded dynamically during training with on-the-fly augmentation

#### **Output Location**
`data/splitting/emotion_split/` (JSON metadata files only)

---

## 📊 Dataset Comparison

| Aspect | Detection Task | Emotion Classification Task |
|--------|---------------|----------------------------|
| **Algorithm** | YOLOv8 (Object Detection) | ResNet50 (Image Classification) |
| **Raw Data Size** | 6,154 images | 9,325 images |
| **Preprocessing** | ✅ Letterbox resize to 640×640 | ❌ No preprocessing |
| **Storage Strategy** | Individual JPEG files in `data/processed/detection/` | Original images in `data/raw/emotion_dataset/` |
| **Annotations** | YOLO format bounding boxes | Class labels from folder structure |
| **Split Method** | Option A: Preserve val, split train | Stratified 70/20/10 split |
| **Training Data Source** | `data/processed/detection/` via `dataset.yaml` | `data/raw/emotion_dataset/` via split JSON files |
| **Critical Metadata** | `dataset.yaml` in processed folder | JSON files in `splitting/emotion_split/` |
| **Memory Usage** | High (preprocessed files stored) | Low (images loaded on-demand) |
| **Augmentation** | Applied during preprocessing | Applied during training (on-the-fly) |

---

## 🎯 Key Design Decisions

### 1. Why Different Strategies?

**Detection Dataset**:
- YOLOv8 requires fixed-size input (640×640)
- Preprocessing once avoids redundant computation during training
- Letterbox resize preserves detection accuracy
- Storing preprocessed files speeds up training iterations

**Emotion Dataset**:
- Classification models handle variable input sizes
- On-the-fly augmentation increases training diversity
- Avoids duplicating 9,325 images (saves disk space)
- Dynamic resizing allows flexible experimentation

### 2. Why Option A Split for Detection?

**Problem**: Original dataset has train/val but no test set.

**Solution**: 
- Keep original validation set intact (professionally curated)
- Split training data to create test set
- Ensures test data never seen during validation

**Alternative Considered**: Mix all data and re-split randomly
- ❌ Rejected: Would destroy professional train/val balance
- ❌ Rejected: May introduce data leakage

### 3. Why Relative Paths in dataset.yaml?

**Portability Requirement**:
- Project must work across different machines (local CPU, AWS GPU, etc.)
- Absolute paths break when moving between environments

**YOLOv8 Behavior**:
- Automatically resolves relative paths from `dataset.yaml` location
- `path: .` means "directory containing dataset.yaml"
- No need to hardcode absolute paths in experiment scripts

---

## 🔧 Usage in Experiments

### Detection Experiments (exp01, exp02, exp03)

```python
# Experiment script loads dataset via YAML config
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(
    data='data/processed/detection/dataset.yaml',  # ← Uses this config
    epochs=50,
    imgsz=640,
    ...
)
```

**No manual data loading required** - YOLOv8 reads `dataset.yaml` directly.

---

### Emotion Experiments (exp04, exp05, exp06)

```python
# Experiment script loads split metadata
import json

with open('data/splitting/emotion_split/train_split.json', 'r') as f:
    train_data = json.load(f)

# Custom Dataset class loads images on-the-fly
class EmotionDataset(Dataset):
    def __init__(self, split_json_path):
        with open(split_json_path, 'r') as f:
            data = json.load(f)
        self.image_paths = data['images']
        self.labels = data['labels']
    
    def __getitem__(self, idx):
        img = load_and_preprocess(self.image_paths[idx])  # ← Dynamic loading
        label = self.labels[idx]
        return img, label
```

**Split JSON files are critical** - they define which images belong to each set.

---

## 📝 Important Notes

### Git Management

According to project specifications:
- ❌ **DO NOT commit** `data/raw/` or `data/processed/` to Git
- ✅ **DO commit** `data/splitting/` to Git (metadata is small and essential)
- 💡 Compressed processed data can be shared separately if needed

### Reprocessing Data

If you need to reprocess:
1. Delete `data/processed/detection/` contents
2. Delete `data/splitting/*/` contents
3. Run preprocessing scripts again:
   ```bash
   python src/data_processing/detection_preprocessor.py
   python src/data_processing/emotion_preprocessor.py
   ```

### Common Pitfalls

1. **NumPy Version Compatibility**: Ensure `numpy<2.0.0` for ultralytics compatibility
2. **Path Issues**: Always use relative paths in configs for portability
3. **Memory Errors**: Use streaming approach for large datasets (already implemented)
4. **Missing Classes**: Verify all 5 emotion folders exist before processing

---

## 🚀 Quick Reference

### File Locations Summary

| Purpose | Path |
|---------|------|
| Detection preprocessing script | `src/data_processing/detection_preprocessor.py` |
| Emotion preprocessing script | `src/data_processing/emotion_preprocessor.py` |
| Processed detection data | `data/processed/detection/` |
| Detection YAML config | `data/processed/detection/dataset.yaml` |
| Emotion split metadata | `data/splitting/emotion_split/*.json` |
| Detection split metadata | `data/splitting/detection_split/*.json` |

### Key Parameters

| Parameter | Value | Location |
|-----------|-------|----------|
| Detection image size | 640×640 | `config.yaml` / `detection_preprocessor.py` |
| Detection padding color | RGB(114, 114, 114) | `_letterbox_resize()` method |
| Emotion split ratio | 70/20/10 | `config.yaml` |
| Detection split strategy | Option A (preserve val) | `_split_dataset()` method |
| Detection batch size | 100 images | `_preprocess_and_save_split()` |

---

## 📚 Related Documentation

- [Project README](./README.md) - Overall project overview
- [Quick Start Guide](./QUICKSTART.md) - Setup and running instructions
- [Part C Project Structure](./PartC-Project%20Structure.md) - Code architecture details

---

**Last Updated**: 2026-04-22  
**Author**: CNN_A3 Development Team
