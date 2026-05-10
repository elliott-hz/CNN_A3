"""
Classification Inference
Standalone inference for classification model
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any
import cv2


class ClassificationInference:
    """
    Inference pipeline for emotion classification.
    """
    
    def __init__(self, model_path: str, class_names: list = None):
        """
        Initialize classification inference with trained model.
        
        Args:
            model_path: Path to trained model (.pth file)
            class_names: List of class names
        """
        self.class_names = class_names or ['angry', 'happy', 'relaxed', 'frown', 'alert']
        
        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Import here to avoid circular dependency
        from src.models.classification_model import ResNet50Classifier
        
        # Recreate model with saved config
        self.model = ResNet50Classifier(checkpoint['config'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Device setup - Support CUDA (NVIDIA), MPS (Apple Silicon), or CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)
        
        print(f"Loaded classification model from: {model_path}")
        print(f"Using device: {self.device}")
    
    def predict(self, image) -> Dict[str, Any]:
        """
        Predict emotion from cropped face image.
        
        Args:
            image: Input image (numpy array, path, or tensor)
            
        Returns:
            Dictionary with predicted class and probabilities
        """
        # Preprocess image
        if isinstance(image, str):
            # Load from path
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image, np.ndarray):
            img = image
        else:
            raise ValueError("Unsupported image type")
        
        # Resize and normalize
        img_resized = cv2.resize(img, (224, 224))
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        # Convert to tensor
        img_tensor = torch.FloatTensor(img_normalized).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Build result
        result = {
            'predicted_class': self.class_names[predicted.item()],
            'confidence': float(confidence),
            'probabilities': {
                class_name: float(prob) 
                for class_name, prob in zip(self.class_names, probabilities[0].cpu().numpy())
            }
        }
        
        return result
