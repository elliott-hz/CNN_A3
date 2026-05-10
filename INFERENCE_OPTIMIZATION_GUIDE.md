# 🚀 Dog Emotion Recognition - Inference Optimization Guide

## 📊 Current Performance Baseline

### Single Image Inference (YOLOv8m + ResNet50 on MPS)

```
image 1/1 /var/folders/.../temp.jpg: 480x640 1 dog, 148.0ms
Speed: 1.5ms preprocess, 148.0ms inference, 0.4ms postprocess per image at shape (1, 3, 480, 640)
```

| Stage | Time | Description |
|-------|------|-------------|
| **Preprocessing** | 1.5ms | Image resize, normalization, tensor conversion |
| **Detection (YOLOv8m)** | ~100ms | Bounding box prediction |
| **Classification (ResNet50)** | ~48ms | Emotion prediction per detected dog |
| **Post-processing** | 0.4ms | NMS, confidence filtering |
| **Total** | **~150ms** | Complete pipeline per image |

### Live Stream Performance

| Metric | Value |
|--------|-------|
| **Per-frame latency** | ~200ms (detection + classification) |
| **Expected FPS** | 5-7 FPS |
| **Input resolution** | 640×480 |
| **Frame capture interval** | 200ms (5 FPS) |

---

##  Optimization Goals: Reality Check

###  Unrealistic Target: 10ms
```
Current:     148ms
Target:      10ms
Speedup:     14.8x faster
Feasibility: NOT ACHIEVABLE with current architecture
```

**Why 10ms is impossible:**
- YOLOv8m is a medium-sized model with complex architecture
- ResNet50 has 25M+ parameters requiring substantial compute
- MPS (Apple Silicon) is not optimized for ultra-low latency
- Would require complete architecture redesign + quantization + specialized hardware

### ✅ Realistic Targets

| Optimization Level | Expected Time | Speedup | FPS | Effort |
|-------------------|---------------|---------|-----|--------|
| **Current (Baseline)** | 148ms | 1.0x | ~6-7 | ✅ Done |
| **Quick Win (This Guide)** | 80-100ms | 1.5-1.8x | ~10-12 | ⭐ 10-30 min |
| **Model Swap (YOLOv8n)** | 30-50ms | 3-5x | ~20-30 | ⭐⭐ 1-2 hours |
| **CoreML Export** | 20-35ms | 4-7x | ~30-50 | ⭐⭐⭐ 2-4 hours |
| **INT8 Quantization** | 15-25ms | 6-10x | ~40-60 | ⭐⭐⭐⭐ 1-2 days |
| **Custom lightweight** | 5-10ms | 15-30x | ~100+ | ⭐⭐⭐⭐⭐ Weeks |

---

## 🔧 Quick Win Optimizations (Implement Now)

### Option 1: Lower Input Resolution ⭐⭐

**Impact:** 20-30% faster (120-100ms)  
**Effort:** 5 minutes  
**Risk:** Minimal (may slightly reduce accuracy)

#### Implementation:

**File: `/Users/25509225/CNN_A3/src/inference/pipeline_inference.py`**

```python
def predict(self, image_path: str, conf: float = 0.5, iou: float = 0.45, img_size: int = 320) -> List[Dict[str, Any]]:
    """
    Run end-to-end inference on an image.
    
    Args:
        image_path: Path to input image OR numpy array (BGR format)
        conf: Detection confidence threshold
        iou: NMS IoU threshold
        img_size: Input resolution for YOLOv8 (default: 320 for speed)
        
    Returns:
        List of results, one per detected dog
    """
    # Lower resolution: 320 instead of 640
    if isinstance(image_path, np.ndarray):
        original_img_bgr = image_path
        original_img_rgb = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2RGB)
        detections = self.detector.predict_from_array(original_img_bgr, conf=conf, iou=iou, imgsz=img_size)
    else:
        detections = self.detector.predict(image_path, conf=conf, iou=iou, imgsz=img_size)
        original_img_bgr = cv2.imread(image_path)
        original_img_rgb = cv2.cvtColor(original_img_bgr, cv2.COLOR_BGR2RGB)
```

**File: `/Users/25509225/CNN_A3/src/inference/detection_inference.py`**

```python
def predict(self, image_path: str, conf: float = 0.5, iou: float = 0.45, imgsz: int = 320) -> List[Dict[str, Any]]:
    """
    Predict bounding boxes in an image.
    
    Args:
        image_path: Path to input image
        conf: Confidence threshold
        iou: NMS IoU threshold
        imgsz: Input size for YOLOv8 (lower = faster)
    """
    # Run inference with smaller input size
    results = self.model(image_path, conf=conf, iou=iou, imgsz=imgsz)
```

---

### Option 2: Reduce Classification Overhead ⭐⭐

**Impact:** 15-25% faster (110-125ms)  
**Effort:** 10 minutes  
**Risk:** Low (skip low-confidence detections)

#### Implementation:

**File: `/Users/25509225/CNN_A3/src/inference/pipeline_inference.py`**

```python
# Skip classification for low-confidence detections
MIN_CONFIDENCE_FOR_CLASSIFICATION = 0.7

for i, detection in enumerate(detections):
    bbox = detection['bbox']
    detection_conf = detection['confidence']
    
    # Skip classification if detection confidence is low
    if detection_conf < MIN_CONFIDENCE_FOR_CLASSIFICATION:
        results.append({
            'dog_id': i + 1,
            'bbox': bbox,
            'detection_confidence': detection_conf,
            'emotion': 'unknown',  # Skip classification
            'emotion_confidence': 0.0,
            'emotion_probabilities': {}
        })
        continue
    
    # Normal classification pipeline
    x1, y1, x2, y2 = map(int, bbox)
    # ... rest of classification code
```

---

### Option 3: Enable FP16 on MPS (if supported) ⭐

**Impact:** 10-15% faster (130-135ms)  
**Effort:** 15 minutes  
**Risk:** Medium (MPS FP16 support is experimental)

#### Implementation:

**File: `/Users/25509225/CNN_A3/src/inference/classification_inference.py`**

```python
def __init__(self, model_path: str, class_names: list = None, use_fp16: bool = False):
    """
    Initialize classification inference with trained model.
    
    Args:
        model_path: Path to trained model (.pth file)
        class_names: List of class names
        use_fp16: Enable half-precision inference (experimental on MPS)
    """
    self.class_names = class_names or ['angry', 'happy', 'relaxed', 'frown', 'alert']
    self.use_fp16 = use_fp16 and torch.backends.mps.is_available()
    
    checkpoint = torch.load(model_path, map_location='cpu')
    from src.models.classification_model import ResNet50Classifier
    
    self.model = ResNet50Classifier(checkpoint['config'])
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.model.eval()
    
    if torch.cuda.is_available():
        self.device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        self.device = torch.device('mps')
    else:
        self.device = torch.device('cpu')
    
    self.model.to(self.device)
    
    # Enable FP16 if supported
    if self.use_fp16:
        self.model = self.model.half()
        print(f"✅ FP16 enabled for faster inference")
    
    print(f"Loaded classification model from: {model_path}")
    print(f"Using device: {self.device}")

def predict(self, image) -> Dict[str, Any]:
    # ... preprocessing code ...
    
    img_tensor = torch.FloatTensor(img_normalized).permute(2, 0, 1).unsqueeze(0)
    img_tensor = img_tensor.to(self.device)
    
    # Convert to FP16 if enabled
    if self.use_fp16:
        img_tensor = img_tensor.half()
    
    with torch.no_grad():
        outputs = self.model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
```

---

### Option 4: Batch Preprocessing ⭐

**Impact:** 10-20% faster for multiple dogs  
**Effort:** 20 minutes  
**Risk:** Low

#### Implementation:

**File: `/Users/25509225/CNN_A3/src/inference/pipeline_inference.py`**

```python
# Collect all crops first
crops = []
crop_indices = []

for i, detection in enumerate(detections):
    if detection['confidence'] < MIN_CONFIDENCE_FOR_CLASSIFICATION:
        continue
    
    bbox = detection['bbox']
    x1, y1, x2, y2 = map(int, bbox)
    
    # Crop with padding
    x1_pad = max(0, x1 - int((x2-x1) * 0.1))
    y1_pad = max(0, y1 - int((y2-y1) * 0.1))
    x2_pad = min(w, x2 + int((x2-x1) * 0.1))
    y2_pad = min(h, y2 + int((y2-y1) * 0.1))
    
    crop = original_img_rgb[y1_pad:y2_pad, x1_pad:x2_pad]
    crops.append(crop)
    crop_indices.append(i)

# Batch classification if multiple crops
if crops:
    # Process all crops in batch
    batch_results = self.classifier.predict_batch(crops)
    
    # Map results back to detections
    for idx, result in zip(crop_indices, batch_results):
        detection = detections[idx]
        results.append({
            'dog_id': idx + 1,
            'bbox': detection['bbox'],
            'detection_confidence': detection['confidence'],
            'emotion': result['predicted_class'],
            'emotion_confidence': result['confidence'],
            'emotion_probabilities': result['probabilities']
        })
```

---

## 📈 Combined Expected Results

### Before Optimization:
```
Total: 148ms
├─ Preprocessing: 1.5ms
├─ Detection (640×640): ~100ms
├─ Classification (per dog): ~48ms
└─ Post-processing: 0.4ms
```

### After All Quick Wins:
```
Total: 80-100ms (1.5-1.8x faster)
├─ Preprocessing: 1.0ms (optimized)
─ Detection (320×320): ~50-60ms (2x faster)
├─ Classification: ~30-40ms (skip low-confidence + batch)
└─ Post-processing: 0.3ms
```

**Expected Live Stream FPS:** 10-12 FPS (up from 5-7 FPS)

---

## 🚀 Advanced Optimizations (Future Work)

### Option 5: Switch to YOLOv8n (Nano Model)

**Impact:** 3-5x faster (30-50ms)  
**Effort:** 1-2 hours (retraining required)  
**Trade-off:** Slightly lower accuracy

#### Steps:

1. **Download YOLOv8n pretrained model:**
```bash
yolo detect train model=yolov8n.pt data=your_dataset.yaml epochs=100 device=mps
```

2. **Update model path:**
```python
# In pipeline_inference.py
self.detector = DetectionInference("best_models/detection_YOLOv8n.pt")
```

3. **Expected results:**
   - Detection: 15-25ms (vs 100ms)
   - Total pipeline: 30-50ms
   - FPS: 20-30

---

### Option 6: CoreML Export for Apple Silicon

**Impact:** 4-7x faster (20-35ms)  
**Effort:** 2-4 hours  
**Trade-off:** Requires model conversion

#### Steps:

1. **Install coremltools:**
```bash
pip3 install coremltools
```

2. **Export YOLOv8 to CoreML:**
```python
from ultralytics import YOLO

model = YOLO("best_models/detection_YOLOv8_baseline.pt")
model.export(format="coreml")
```

3. **Use CoreML inference:**
```python
import coremltools as ct

# Load CoreML model
coreml_model = ct.models.MLModel("detection_YOLOv8_baseline.mlmodel")

# Inference
prediction = coreml_model.predict({"image": image_array})
```

---

### Option 7: INT8 Quantization

**Impact:** 6-10x faster (15-25ms)  
**Effort:** 1-2 days  
**Trade-off:** Requires quantization-aware training

#### Steps:

1. **Prepare calibration dataset**
2. **Run quantization-aware training**
3. **Export quantized model**
4. **Use TensorRT/CoreML with INT8**

---

##  Performance Summary Table

| Optimization | Time | Speedup | FPS | Complexity |
|-------------|------|---------|-----|------------|
| **Baseline** | 148ms | 1.0x | 6-7 | - |
| **Lower Resolution (320)** | 100-120ms | 1.3-1.5x | 8-10 | ⭐ Easy |
| **Skip Low-Confidence** | 110-125ms | 1.2-1.3x | 8-9 | ⭐ Easy |
| **Batch Classification** | 120-130ms | 1.1-1.2x | 7-8 | ⭐ Medium |
| **FP16 (if supported)** | 130-135ms | 1.1x | 7 | ⭐⭐ Medium |
| **All Quick Wins Combined** | 80-100ms | 1.5-1.8x | 10-12 | ⭐⭐⭐ |
| **YOLOv8n (retrain)** | 30-50ms | 3-5x | 20-30 | ⭐⭐⭐ |
| **CoreML Export** | 20-35ms | 4-7x | 30-50 | ⭐⭐⭐⭐ |
| **INT8 Quantization** | 15-25ms | 6-10x | 40-60 | ⭐⭐⭐⭐ |
| **10ms Target** | ❌ | ❌ |  | Not feasible |

---

##  Recommendation: What to Do Now

### Immediate Action (Next 30 minutes):

1. ✅ **Lower resolution to 320×320** → 100-120ms
2. ✅ **Skip low-confidence classification** → 110-125ms
3. ✅ **Test both changes together** → Expect 80-100ms

**Result:** ~10-12 FPS live stream (good enough for demo!)

### Medium-term (Next week):

- Consider retraining with YOLOv8n for 30-50ms performance
- Evaluate if CoreML export is worth the effort

### Long-term (Future projects):

- Explore INT8 quantization for production deployment
- Consider cloud GPU (NVIDIA T4) for ultra-low latency needs

---

## 📝 Implementation Checklist

- [ ] Lower input resolution to 320×320
- [ ] Add `imgsz` parameter to detection pipeline
- [ ] Implement low-confidence skip logic
- [ ] Test performance improvements
- [ ] Validate accuracy hasn't degraded significantly
- [ ] Update live stream frame capture rate (if faster)
- [ ] Document new performance baseline

---

## 📚 References

- YOLOv8 documentation: https://docs.ultralytics.com/
- PyTorch MPS guide: https://pytorch.org/docs/stable/notes/mps.html
- CoreML export: https://coremltools.readme.io/
- Model quantization: https://pytorch.org/docs/stable/quantization.html

---

**Last Updated:** 2026-05-10  
**Author:** Elliott  
**Hardware:** Apple Silicon (MPS), macOS
