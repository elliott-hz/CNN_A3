"""
Custom inference handler for GoogLeNet model with auxiliary classifiers
Handles inference for models with auxiliary outputs but only uses main classifier
"""

import torch
from PIL import Image
from typing import Union, List, Dict, Any
import numpy as np
from .classification_inference import ClassificationInference


class CustomClassificationInference(ClassificationInference):
    """
    Inference class specifically designed to handle models with auxiliary classifiers.
    Extends the base ClassificationInference but ensures that only the main classifier
    output is used during inference while the auxiliary classifiers are only used 
    during training.
    """
    
    def __init__(self, model, transform=None, device=None):
        """
        Initialize the custom inference handler for models with auxiliary classifiers
        
        Args:
            model: Model instance (should support auxiliary classifiers)
            transform: Transformations to apply to input images
            device: Device to run inference on (defaults to CUDA if available)
        """
        super().__init__(model, transform, device)
        
        # Check if the model has auxiliary classifiers
        self.has_auxiliary = (
            hasattr(self.model, 'aux_classifier1') and 
            hasattr(self.model, 'aux_classifier2')
        )
    
    def predict(self, image: Union[Image.Image, torch.Tensor, str]) -> Dict[str, Any]:
        """
        Make a prediction on a single image.
        
        Args:
            image: Input image (PIL Image, tensor, or path to image)
            
        Returns:
            Dictionary containing prediction results
        """
        self.model.eval()
        
        # Preprocess image if needed
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if not isinstance(image, torch.Tensor):
            image = self.transform(image)
        
        image = image.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Ensure only main classifier output is used during inference
            if self.has_auxiliary:
                output = self.model(image, use_auxiliary=False)
            else:
                output = self.model(image)
            
            probabilities = torch.softmax(output, dim=1)
            predicted_class_idx = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted_class_idx].item()
        
        return {
            'class_idx': predicted_class_idx,
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy(),
            'predicted_class': self._idx_to_class(predicted_class_idx)
        }
    
    def predict_batch(self, images: Union[List[Image.Image], List[torch.Tensor], torch.Tensor]) -> List[Dict[str, Any]]:
        """
        Make predictions on a batch of images.
        
        Args:
            images: List of input images or image tensor
            
        Returns:
            List of dictionaries containing prediction results
        """
        self.model.eval()
        
        # Process images if needed
        processed_images = []
        for img in images:
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
            
            if not isinstance(img, torch.Tensor):
                img = self.transform(img)
            
            processed_images.append(img)
        
        # Stack images into a batch
        batch = torch.stack(processed_images).to(self.device)
        
        with torch.no_grad():
            # Ensure only main classifier output is used during inference
            if self.has_auxiliary:
                outputs = self.model(batch, use_auxiliary=False)
            else:
                outputs = self.model(batch)
            
            probabilities = torch.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(outputs, dim=1)
            confidences = torch.max(probabilities, dim=1).values
        
        results = []
        for i in range(len(images)):
            results.append({
                'class_idx': predicted_classes[i].item(),
                'confidence': confidences[i].item(),
                'probabilities': probabilities[i].cpu().numpy(),
                'predicted_class': self._idx_to_class(predicted_classes[i].item())
            })
        
        return results
    
    def _idx_to_class(self, idx: int) -> str:
        """
        Convert class index to class name.
        
        Args:
            idx: Class index
            
        Returns:
            Class name
        """
        class_names = ['Alert', 'Angry', 'Frown', 'Happy', 'Relax']
        if 0 <= idx < len(class_names):
            return class_names[idx]
        else:
            return f'Unknown ({idx})'