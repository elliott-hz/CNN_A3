"""
SSD Detector Model Definition
SSD300-based detector with configurable parameters
"""

import torch
import torch.nn as nn
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
import torchvision.transforms as T
from PIL import Image
import numpy as np
import yaml
from pathlib import Path
import ssl
from torch.utils.data import Dataset
from typing import Dict, Any, List


class SSDDataset(Dataset):
    """
    Custom dataset for SSD that loads YOLO format annotations
    and converts them to the format expected by torchvision detection models.
    """
    def __init__(self, dataset_path: str, split: str = 'train', transform=None, max_images: int = None):
        super().__init__()
        self.dataset_path = Path(dataset_path)
        
        # Handle dataset.yaml path
        if self.dataset_path.name == "dataset.yaml":
            config_path = self.dataset_path
            self.dataset_path = config_path.parent
        else:
            config_path = self.dataset_path / "dataset.yaml"
        
        self.transform = transform
        self.split = split
        self.max_images = max_images
        
        # Load dataset configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Determine image and label directories for the split
        img_subdir = self.config.get(split, f'images/{split}')
        lbl_subdir = img_subdir.replace('images', 'labels')
        
        self.img_dir = self.dataset_path / img_subdir
        self.label_dir = self.dataset_path / lbl_subdir
        
        # Get list of images
        self.images = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            self.images.extend(list(self.img_dir.glob(f'*{ext}')))
        
        self.images = sorted(list(set(self.images)))
        
        # Limit dataset size if specified (for small local tests)
        if max_images is not None and max_images > 0:
            self.images = self.images[:max_images]
            print(f"[SSD Dataset] Limited to {max_images} images for {split} split")
        
        print(f"[SSD Dataset] Found {len(self.images)} images in {self.img_dir} for split '{split}'")
        
        # Map class names to IDs
        if isinstance(self.config.get('names'), dict):
            self.class_to_idx = self.config['names']
        elif isinstance(self.config.get('names'), list):
            self.class_to_idx = {name: idx for idx, name in enumerate(self.config['names'])}
        else:
            self.class_to_idx = {'dog': 0}
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        
        # Generate label path
        label_filename = img_path.stem + ".txt"
        label_path = self.label_dir / label_filename
        
        # Load image as PIL Image
        image = Image.open(img_path).convert("RGB")
        img_width, img_height = image.size
        
        # Initialize boxes and labels
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(float(parts[0]))
                        center_x, center_y = float(parts[1]), float(parts[2])
                        width, height = float(parts[3]), float(parts[4])
                        
                        # YOLO format (normalized center x,y,w,h) → SSD format (absolute x1,y1,x2,y2)
                        abs_x_center = center_x * img_width
                        abs_y_center = center_y * img_height
                        abs_width = width * img_width
                        abs_height = height * img_height
                        
                        x_min = abs_x_center - abs_width / 2
                        y_min = abs_y_center - abs_height / 2
                        x_max = abs_x_center + abs_width / 2
                        y_max = abs_y_center + abs_height / 2
                        
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(class_id + 1)  # +1 because 0 is background
        
        # Convert to tensors
        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        # Clamp boxes to image boundaries
        if len(boxes) > 0:
            boxes[:, 0] = torch.clamp(boxes[:, 0], min=0, max=img_width)
            boxes[:, 1] = torch.clamp(boxes[:, 1], min=0, max=img_height)
            boxes[:, 2] = torch.clamp(boxes[:, 2], min=0, max=img_width)
            boxes[:, 3] = torch.clamp(boxes[:, 3], min=0, max=img_height)
            
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[keep]
            labels = labels[keep]
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        
        # Convert PIL image to tensor
        image = T.ToTensor()(image)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, target


class SSDDetector:
    """
    SSD300 detector based on VGG16 backbone with configurable parameters.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.nms_iou_threshold = config.get('nms_iou_threshold', 0.5)
        self.input_size = config.get('input_size', 300)
        self.num_classes = config.get('num_classes', 2)  # 1 class (dog) + background
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SSD] Using device: {self.device}")
        
        # Load pretrained SSD300 VGG16
        weights = config.get('weights', 'DEFAULT')
        original_context = ssl._create_default_https_context
        ssl._create_default_https_context = ssl._create_unverified_context
        try:
            if weights == 'DEFAULT':
                self.model = ssd300_vgg16(
                    weights=SSD300_VGG16_Weights.DEFAULT,
                    weights_backbone=None
                )
            else:
                self.model = ssd300_vgg16(weights=None, weights_backbone=None)
        finally:
            ssl._create_default_https_context = original_context
        
        # Modify the classifier head for our number of classes
        # SSD head: num_classes includes background
        from torchvision.models.detection.ssd import SSDClassificationHead
        
        # SSD300 VGG16 uses 6 feature maps with these channel dimensions
        # These are hardcoded for SSD300 architecture
        in_channels = [512, 1024, 512, 256, 256, 256]
        num_anchors = self.model.anchor_generator.num_anchors_per_location()
        
        self.model.head.classification_head = SSDClassificationHead(
            in_channels, num_anchors, self.num_classes
        )
        
        self.model.to(self.device)
        self.model.eval()
        
        # Transforms: SSD expects 300x300 input
        self.transform = T.Compose([
            T.Resize((self.input_size, self.input_size)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess(self, image):
        """Preprocess a single image for the model"""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        image_tensor = T.ToTensor()(image)
        if image_tensor.shape[1] != self.input_size or image_tensor.shape[2] != self.input_size:
            image_tensor = T.Resize((self.input_size, self.input_size))(image_tensor)
        
        return image_tensor
    
    def predict(self, image):
        """Run prediction on a single image"""
        processed_image = self.preprocess(image).to(self.device)
        input_tensor = processed_image.unsqueeze(0)
        
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(input_tensor)
        
        boxes = outputs[0]['boxes'].cpu().numpy()
        scores = outputs[0]['scores'].cpu().numpy()
        labels = outputs[0]['labels'].cpu().numpy()
        
        # Filter by confidence
        mask = scores >= self.confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        return {'boxes': boxes, 'scores': scores, 'labels': labels}
    
    def predict_batch(self, images):
        """Run prediction on a batch of images"""
        processed_images = []
        for img in images:
            processed_img = self.preprocess(img).to(self.device)
            processed_images.append(processed_img)
        
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(processed_images)
        
        results = []
        for output in outputs:
            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()
            
            mask = scores >= self.confidence_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]
            
            results.append({'boxes': boxes, 'scores': scores, 'labels': labels})
        
        return results
    
    def train_mode(self):
        self.model.train()
    
    def eval_mode(self):
        self.model.eval()
    
    def load_weights(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
    
    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)
    
    def get_model_params(self):
        return sum(p.numel() for p in self.model.parameters())
    
    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
    
    @staticmethod
    def collate_fn(batch):
        """Collate function for DataLoader"""
        return tuple(zip(*batch))


# SSD configurations
SSD_BASELINE_CONFIG = {
    'input_size': 300,
    'confidence_threshold': 0.5,
    'nms_iou_threshold': 0.5,
    'num_classes': 2,  # dog + background
    'weights': 'DEFAULT'
}

SSD_MODIFIED_V1_CONFIG = {
    'input_size': 512,
    'confidence_threshold': 0.6,
    'nms_iou_threshold': 0.55,
    'num_classes': 2,
    'weights': 'DEFAULT'
}

SSD_MODIFIED_V2_CONFIG = {
    'input_size': 300,
    'confidence_threshold': 0.4,
    'nms_iou_threshold': 0.45,
    'num_classes': 2,
    'weights': 'DEFAULT'
}

