"""
Training Module
Contains training frameworks for detection and classification models
"""

from .detection_trainer import DetectionTrainer
from .classification_trainer import ClassificationTrainer

__all__ = ['DetectionTrainer', 'ClassificationTrainer']
