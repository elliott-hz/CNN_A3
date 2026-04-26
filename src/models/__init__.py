"""
Models Module
Contains model definitions for detection and classification
"""

from .detection_model import YOLOv8Detector, FasterRCNNDetector, SSDDetector
from .classification_model import ResNet50Classifier, AlexNetClassifier, GoogLeNetClassifier

__all__ = [
    'YOLOv8Detector', 
    'FasterRCNNDetector',
    'SSDDetector',
    'ResNet50Classifier', 
    'AlexNetClassifier', 
    'GoogLeNetClassifier'
]
