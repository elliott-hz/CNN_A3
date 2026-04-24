"""
Evaluation Module
Contains evaluation frameworks for detection and classification models
"""

from .detection_evaluator import DetectionEvaluator
from .classification_evaluator import ClassificationEvaluator

__all__ = ['DetectionEvaluator', 'ClassificationEvaluator']
