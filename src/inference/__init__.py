"""
Inference Module
Contains inference pipelines for detection, classification, and end-to-end processing
"""

from .detection_inference import DetectionInference
from .classification_inference import ClassificationInference
from .pipeline_inference import PipelineInference
from .googlenet_inference import GoogLeNetInference  # New addition

__all__ = ['DetectionInference', 'ClassificationInference', 'PipelineInference', 'GoogLeNetInference']