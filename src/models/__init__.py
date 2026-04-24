# Import all model classes
from .classification_model import ResNet50Classifier, EfficientNetB0Classifier
from .detection_model import YOLOv8Detector
from .alexnet_model import AlexNetClassifier
from .googlenet_model import GoogLeNetClassifier, create_googlenet_model, STANDARD_GOOGLENET_CONFIG
from .googlenet_model_with_auxiliary import GoogLeNetClassifierWithAuxiliary, create_googlenet_with_auxiliary_model, STANDARD_GOOGLENET_WITH_AUX_CONFIG

__all__ = [
    'ResNet50Classifier',
    'EfficientNetB0Classifier',
    'YOLOv8Detector',
    'AlexNetClassifier',
    'GoogLeNetClassifier',
    'GoogLeNetClassifierWithAuxiliary',
    'create_googlenet_model',
    'create_googlenet_with_auxiliary_model',
    'STANDARD_GOOGLENET_CONFIG',
    'STANDARD_GOOGLENET_WITH_AUX_CONFIG'
]