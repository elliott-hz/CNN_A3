"""
Models Module
Contains model definitions for detection and classification
"""
from .detection_model import YOLOv8Detector
from .classification_model import ResNet50Classifier
from .alexnet_model import AlexNetClassifier
from .googlenet_model import GoogLeNetClassifier
from .googlenet_model_with_auxiliary import GoogLeNetClassifierWithAuxiliary  # New addition

__all__ = ['YOLOv8Detector', 'ResNet50Classifier', 'AlexNetClassifier', 'GoogLeNetClassifier', 'GoogLeNetClassifierWithAuxiliary']
