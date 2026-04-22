"""
Detection Model Definition
Single YOLOv8Detector class with configurable parameters for different variants
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from typing import Dict, Any


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
                    optimizer=None, scheduler=None, warmup_epochs=None,
                    **kwargs):
        """
        Train the model on given data.
        
        Args:
            data: Dataset configuration or path
            epochs: Number of training epochs
            imgsz: Image size (uses default if None)
            optimizer: Optimizer type ('SGD', 'Adam', 'AdamW')
            scheduler: Learning rate scheduler ('CosineLR', 'StepLR', etc.)
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


def create_detection_model(config: Dict[str, Any]) -> YOLOv8Detector:
    """
    Factory function to create detection model instance.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        YOLOv8Detector instance
    """
    return YOLOv8Detector(config)


# Example configurations for different experiment variants
BASELINE_DETECTION_CONFIG = {
    'backbone': 'm',  # Medium model
    'input_size': 640,
    'confidence_threshold': 0.5,
    'nms_iou_threshold': 0.45,
    'pretrained': True
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
