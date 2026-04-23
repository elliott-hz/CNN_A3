"""
Models Module
Contains model definitions for detection and classification
"""

from .detection_model import YOLOv8Detector
from .classification_model import ResNet50Classifier
from .alexnet_model import AlexNetClassifier

__all__ = ['YOLOv8Detector', 'ResNet50Classifier', 'AlexNetClassifier']