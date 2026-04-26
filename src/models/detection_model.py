"""
Detection Model Definition
Supports YOLOv8, Faster R-CNN, and SSD models
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, ssd300_vgg16
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import Dict, Any, List, Tuple


class YOLOv8Detector(nn.Module):
    """
    YOLOv8 detector with configurable parameters.
    
    This class wraps the YOLOv8 model and allows for different configurations
    through the config parameter, enabling different experiment variants
    (baseline, modified v1, modified v2) using the same class.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize YOLOv8 detector with configuration.
        
        Args:
            config: Dictionary with model configuration parameters
                - backbone: 'n', 's', 'm', 'l', 'x' (nano, small, medium, large, extra-large)
                - input_size: Input image size (default 640)
                - confidence_threshold: Detection confidence threshold (default 0.5)
                - nms_iou_threshold: NMS IoU threshold (default 0.45)
                - pretrained: Whether to use pretrained weights (default True)
        """
        super(YOLOv8Detector, self).__init__()
        
        # Store configuration
        self.config = config
        self.backbone = config.get('backbone', 'm')  # Default to medium
        self.input_size = config.get('input_size', 640)
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.nms_iou_threshold = config.get('nms_iou_threshold', 0.45)
        self.pretrained = config.get('pretrained', True)
        
        # Construct model name based on backbone
        model_name = f'yolov8{self.backbone}.pt' if self.pretrained else f'yolov8{self.backbone}.yaml'
        
        # Initialize YOLO model
        if self.pretrained:
            # Load pretrained model
            self.model = YOLO(model_name)
        else:
            # Create untrained model from config
            self.model = YOLO(f'yolov8{self.backbone}.yaml')
            # Initialize with random weights
        
        # Update model configuration based on our config
        self.model.model.conf = self.confidence_threshold
        self.model.model.iou = self.nms_iou_threshold
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Model predictions
        """
        # For training, we use the model directly
        # For inference, we might want to adjust confidences
        return self.model(x, verbose=False)
    
    def predict(self, x, conf=None, iou=None):
        """
        Perform prediction with adjustable thresholds.
        
        Args:
            x: Input image or path
            conf: Override confidence threshold (optional)
            iou: Override NMS IoU threshold (optional)
            
        Returns:
            Prediction results
        """
        conf = conf or self.confidence_threshold
        iou = iou or self.nms_iou_threshold
        
        return self.model(x, conf=conf, iou=iou, verbose=False)
    
    def train_model(self, data, epochs=100, imgsz=None, 
                    optimizer=None, warmup_epochs=None,
                    **kwargs):
        """
        Train the model on given data.
        
        Args:
            data: Dataset configuration or path
            epochs: Number of training epochs
            imgsz: Image size (uses default if None)
            optimizer: Optimizer type ('SGD', 'Adam', 'AdamW')
            warmup_epochs: Number of warmup epochs
            **kwargs: Additional training arguments
            
        Returns:
            Training results
        """
        if imgsz is None:
            imgsz = self.input_size
        
        # Set default training args with our config values
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
        if warmup_epochs is not None:
            train_args['warmup_epochs'] = warmup_epochs
        
        # Update with any additional kwargs
        train_args.update(kwargs)
        
        # Perform training
        results = self.model.train(**train_args)
        return results
    
    def save(self, save_path: str):
        """
        Save the model.
        
        Args:
            save_path: Path to save the model
        """
        # Save the underlying model
        self.model.save(save_path)
    
    def update_config(self, **kwargs):
        """
        Update model configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                # Update the model's internal configuration if applicable
                if key == 'confidence_threshold':
                    self.model.model.conf = value
                elif key == 'nms_iou_threshold':
                    self.model.model.iou = value


class FasterRCNNDetector(nn.Module):
    """
    Faster R-CNN detector with ResNet50+FPN backbone.
    
    Two-stage detection model with high accuracy.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Faster R-CNN detector.
        
        Args:
            config: Dictionary with model configuration parameters
                - num_classes: Number of detection classes (including background)
                - pretrained: Whether to use pretrained backbone (default True)
                - min_size: Minimum image size for resizing (default 800)
                - max_size: Maximum image size for resizing (default 1333)
        """
        super(FasterRCNNDetector, self).__init__()
        
        self.config = config
        self.num_classes = config.get('num_classes', 2)  # 1 class + background
        self.pretrained = config.get('pretrained', True)
        self.min_size = config.get('min_size', 800)
        self.max_size = config.get('max_size', 1333)
        
        # Load pretrained Faster R-CNN model
        self.model = fasterrcnn_resnet50_fpn_v2(
            pretrained=self.pretrained,
            progress=True,
            weights_backbone=None if not self.pretrained else "DEFAULT"
        )
        
        # Replace the box predictor for custom number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        
        # Update transform parameters
        self.model.transform.min_size = (self.min_size,)
        self.model.transform.max_size = self.max_size
    
    def forward(self, images: List[torch.Tensor], targets: List[Dict[str, torch.Tensor]] = None):
        """
        Forward pass through the model.
        
        Args:
            images: List of input tensors
            targets: List of target dictionaries (only during training)
            
        Returns:
            During training: dict with losses
            During inference: list of detections
        """
        return self.model(images, targets)
    
    def save(self, save_path: str):
        """
        Save the model.
        
        Args:
            save_path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, save_path)
    
    def load(self, load_path: str):
        """
        Load model weights.
        
        Args:
            load_path: Path to load the model from
        """
        checkpoint = torch.load(load_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])


class SSDDetector(nn.Module):
    """
    SSD (Single Shot Detector) with VGG16 backbone.
    
    Single-stage detection model optimized for speed.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SSD detector.
        
        Args:
            config: Dictionary with model configuration parameters
                - num_classes: Number of detection classes (including background)
                - pretrained: Whether to use pretrained backbone (default True)
                - min_size: Minimum image size (default 300)
                - max_size: Maximum image size (default 300)
        """
        super(SSDDetector, self).__init__()
        
        self.config = config
        self.num_classes = config.get('num_classes', 2)  # 1 class + background
        self.pretrained = config.get('pretrained', True)
        self.min_size = config.get('min_size', 300)
        self.max_size = config.get('max_size', 300)
        
        # Load pretrained SSD model
        self.model = ssd300_vgg16(
            pretrained=self.pretrained,
            progress=True,
            weights_backbone=None if not self.pretrained else "DEFAULT"
        )
        
        # Replace the classification and regression heads for custom number of classes
        # SSD has 6 feature layers with different channel counts
        in_channels = [512, 1024, 512, 256, 256, 256]
        num_anchors = self.model.anchor_generator.num_anchors_per_location()
        
        from torchvision.models.detection.ssd import SSDClassificationHead, SSDRegressionHead
        
        # Create new heads with proper initialization to reduce false positives
        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        
        self.model.head.classification_head = SSDClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=self.num_classes
        )
        
        self.model.head.regression_head = SSDRegressionHead(
            in_channels=in_channels,
            num_anchors=num_anchors
        )
        
        # Initialize output layer biases to favor background class initially
        # This is critical for reducing massive false positives in SSD
        for head in [self.model.head.classification_head, self.model.head.regression_head]:
            for module in head.modules():
                if isinstance(module, torch.nn.Conv2d):
                    torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        # Apply negative bias only to classification head's output layers
                        if (hasattr(head, 'num_classes') and 
                            module.out_channels == self.num_classes * max(num_anchors)):
                            torch.nn.init.constant_(module.bias, bias_value)
                        else:
                            torch.nn.init.constant_(module.bias, 0)
        
        # Update transform parameters
        self.model.transform.min_size = (self.min_size,)
        self.model.transform.max_size = self.max_size
    
    def forward(self, images: List[torch.Tensor], targets: List[Dict[str, torch.Tensor]] = None):
        """
        Forward pass through the model.
        
        Args:
            images: List of input tensors
            targets: List of target dictionaries (only during training)
            
        Returns:
            During training: dict with losses
            During inference: list of detections
        """
        return self.model(images, targets)
    
    def save(self, save_path: str):
        """
        Save the model.
        
        Args:
            save_path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }, save_path)
    
    def load(self, load_path: str):
        """
        Load model weights.
        
        Args:
            load_path: Path to load the model from
        """
        checkpoint = torch.load(load_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])


def create_detection_model(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create detection model instance.
    
    Args:
        config: Model configuration dictionary with 'model_type' key
            - model_type: 'yolov8', 'faster_rcnn', or 'ssd'
            - Other parameters depend on model type
        
    Returns:
        Detection model instance
    """
    model_type = config.get('model_type', 'yolov8')
    
    if model_type == 'yolov8':
        return YOLOv8Detector(config)
    elif model_type == 'faster_rcnn':
        return FasterRCNNDetector(config)
    elif model_type == 'ssd':
        return SSDDetector(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Example configurations for different experiment variants
BASELINE_DETECTION_CONFIG = {
    'model_type': 'yolov8',
    'backbone': 'm',  # Medium model
    'input_size': 640,
    'confidence_threshold': 0.5,
    'nms_iou_threshold': 0.45,
    'pretrained': True
}

FASTER_RCNN_CONFIG = {
    'model_type': 'faster_rcnn',
    'num_classes': 2,  # 1 class + background
    'pretrained': True,
    'min_size': 640,
    'max_size': 640
}

SSD_CONFIG = {
    'model_type': 'ssd',
    'num_classes': 2,  # 1 class + background
    'pretrained': True,
    'min_size': 300,
    'max_size': 300
}

MODIFIED_V1_DETECTION_CONFIG = {
    'backbone': 'l',  # Large model
    'input_size': 1280,  # Larger input
    'confidence_threshold': 0.6,  # Higher confidence
    'nms_iou_threshold': 0.5,
    'pretrained': True
}

MODIFIED_V2_DETECTION_CONFIG = {
    'backbone': 's',  # Small model
    'input_size': 640,
    'confidence_threshold': 0.4,  # Lower confidence
    'nms_iou_threshold': 0.4,
    'pretrained': True
}
