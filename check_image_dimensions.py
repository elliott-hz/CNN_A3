"""
Image Dimension Checker
Checks the dimensions (width, height, channels) of images in both datasets.
This is crucial for determining if resizing is needed during training.
"""

from pathlib import Path
import cv2
from collections import Counter


def check_dataset_dimensions():
    """Check image dimensions for both detection and emotion datasets."""
    
    print("=" * 80)
    print("IMAGE DIMENSION ANALYSIS")
    print("=" * 80)
    
    # ==================== DETECTION DATASET ====================
    print("\n" + "=" * 80)
    print("DETECTION DATASET - Image Dimensions Analysis")
    print("=" * 80)
    
    detection_train_dir = Path("data/raw/detection_dataset/train_img")
    detection_val_dir = Path("data/raw/detection_dataset/val_img")
    
    def check_image_dimensions(img_dir, split_name, max_samples=200):
        """Check image dimensions (height, width, channels)"""
        if not img_dir.exists():
            print(f"\n{split_name}: Directory not found!")
            return None
        
        dimensions = []
        sample_count = 0
        
        for img_file in list(img_dir.glob("*.jpg"))[:max_samples]:
            img = cv2.imread(str(img_file))
            if img is not None:
                h, w, c = img.shape
                dimensions.append((w, h, c))
                sample_count += 1
        
        dim_counter = Counter(dimensions)
        print(f"\n{split_name} (sampled {sample_count} images):")
        print(f"  Unique dimensions (W x H x C): {len(dim_counter)}")
        for dim, count in dim_counter.most_common(10):
            print(f"    {dim[0]}x{dim[1]}x{dim[2]}: {count} images")
        
        # Summary statistics
        if dimensions:
            widths = [d[0] for d in dimensions]
            heights = [d[1] for d in dimensions]
            channels = [d[2] for d in dimensions]
            
            print(f"\n  Statistics:")
            print(f"    Width:  min={min(widths)}, max={max(widths)}, avg={sum(widths)/len(widths):.1f}")
            print(f"    Height: min={min(heights)}, max={max(heights)}, avg={sum(heights)/len(heights):.1f}")
            print(f"    Channels: min={min(channels)}, max={max(channels)}, unique={set(channels)}")
        
        return dim_counter
    
    train_dims = check_image_dimensions(detection_train_dir, "Train Split")
    val_dims = check_image_dimensions(detection_val_dir, "Val Split")
    
    # ==================== EMOTION DATASET ====================
    print("\n" + "=" * 80)
    print("EMOTION DATASET - Image Dimensions Analysis")
    print("=" * 80)
    
    emotion_base_dir = Path("data/raw/emotion_dataset")
    emotion_classes = ["alert", "angry", "frown", "happy", "relax"]
    
    all_emotion_dims = []
    
    for class_name in emotion_classes:
        class_dir = emotion_base_dir / class_name
        if not class_dir.exists():
            print(f"\n{class_name}: Directory not found!")
            continue
        
        dimensions = []
        sample_count = 0
        max_samples = 100
        
        for img_file in list(class_dir.glob("*.jpg"))[:max_samples]:
            img = cv2.imread(str(img_file))
            if img is not None:
                h, w, c = img.shape
                dimensions.append((w, h, c))
                sample_count += 1
        
        all_emotion_dims.extend(dimensions)
        dim_counter = Counter(dimensions)
        
        print(f"\n{class_name} (sampled {sample_count} images):")
        print(f"  Unique dimensions (W x H x C): {len(dim_counter)}")
        for dim, count in dim_counter.most_common(5):
            print(f"    {dim[0]}x{dim[1]}x{dim[2]}: {count} images")
    
    # Overall emotion dataset statistics
    if all_emotion_dims:
        widths = [d[0] for d in all_emotion_dims]
        heights = [d[1] for d in all_emotion_dims]
        channels = [d[2] for d in all_emotion_dims]
        
        print(f"\n{'='*80}")
        print(f"EMOTION DATASET - Overall Statistics (all classes)")
        print(f"{'='*80}")
        print(f"  Total sampled: {len(all_emotion_dims)} images")
        print(f"  Unique dimensions: {len(Counter(all_emotion_dims))}")
        print(f"\n  Statistics:")
        print(f"    Width:  min={min(widths)}, max={max(widths)}, avg={sum(widths)/len(widths):.1f}")
        print(f"    Height: min={min(heights)}, max={max(heights)}, avg={sum(heights)/len(heights):.1f}")
        print(f"    Channels: min={min(channels)}, max={max(channels)}, unique={set(channels)}")
    
    # ==================== CONCLUSION ====================
    print("\n" + "=" * 80)
    print("CONCLUSION & RECOMMENDATIONS")
    print("=" * 80)
    
    # Check if dimensions are uniform
    detection_uniform = len(train_dims) <= 1 if train_dims else False
    emotion_uniform = len(Counter(all_emotion_dims)) <= 1 if all_emotion_dims else False
    
    if detection_uniform and emotion_uniform:
        print("✓ All images have UNIFORM dimensions - No resizing needed!")
    else:
        print("✗ Images have VARYING dimensions - Resizing REQUIRED during training!")
        print("\nFor deep learning models:")
        print("  - All images in a batch MUST have the same dimensions")
        print("  - You MUST implement resize logic in your Dataset class")
        print("\nRecommended target sizes:")
        print("  - Detection: 640x640 (YOLOv8 standard)")
        print("  - Classification: 224x224 (ResNet standard)")
        print("\nImplementation example:")
        print("  def __getitem__(self, idx):")
        print("      img = cv2.imread(self.image_paths[idx])")
        print("      img = cv2.resize(img, (target_size, target_size))")
        print("      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)")
        print("      return img, label")


if __name__ == "__main__":
    check_dataset_dimensions()
