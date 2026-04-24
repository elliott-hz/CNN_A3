"""
Training Module
Contains training frameworks for detection and classification models
"""

from .detection_trainer import DetectionTrainer
from .classification_trainer import ClassificationTrainer
from .googlenet_trainer import GoogLeNetTrainer  # New addition

__all__ = ['DetectionTrainer', 'ClassificationTrainer', 'GoogLeNetTrainer']
