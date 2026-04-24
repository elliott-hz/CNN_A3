"""
Inference module for GoogLeNet with auxiliary classifiers.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Union
import torchvision.transforms as transforms


class GoogLeNetInference:
    """
    Inference handler for GoogLeNet models with auxiliary classifiers.
    Ensures that auxiliary classifiers are properly handled during inference.
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        """
        Initialize inference handler.
        
        Args:
            model: Trained GoogLeNet model
            device: Device to run inference on
        """
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Ensure model is in eval mode
        self.model.eval()
        self.model.to(self.device)
        
        # Define image preprocessing transforms
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        Preprocess input image for model inference.
        
        Args:
            image: Path to image file or PIL Image object
            
        Returns:
            Preprocessed image tensor
        """
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Apply preprocessing
        image_tensor = self.preprocess(image)
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def predict(self, image: Union[str, Image.Image], 
                return_probabilities: bool = False) -> Union[int, Dict[str, Any]]:
        """
        Perform inference on a single image.
        
        Args:
            image: Path to image file or PIL Image object
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Predicted class index or dictionary with predictions and probabilities
        """
        # Preprocess the image
        image_tensor = self.preprocess_image(image)
        
        # Perform inference
        with torch.no_grad():
            # Ensure model is in eval mode to get single output
            self.model.eval()
            
            # Get model output
            outputs = self.model(image_tensor)
            
            # Handle both regular and auxiliary classifier models
            if isinstance(outputs, tuple):
                # If model returns tuple (shouldn't happen in eval mode but just in case)
                main_output = outputs[0]
            else:
                # Otherwise use the output directly
                main_output = outputs
            
            # Apply softmax to get probabilities
            probabilities = torch.softmax(main_output, dim=1)
            
            # Get predicted class
            predicted_class = torch.argmax(probabilities, dim=1).item()
        
        if return_probabilities:
            return {
                "predicted_class": predicted_class,
                "probabilities": probabilities.squeeze().cpu().numpy().tolist()
            }
        else:
            return predicted_class
    
    def predict_batch(self, images: List[Union[str, Image.Image]]) -> List[int]:
        """
        Perform inference on a batch of images.
        
        Args:
            images: List of image paths or PIL Image objects
            
        Returns:
            List of predicted class indices
        """
        # Preprocess all images
        image_tensors = []
        for img in images:
            image_tensors.append(self.preprocess_image(img))
        
        # Stack into a batch
        batch_tensor = torch.cat(image_tensors, dim=0)
        
        # Perform inference
        with torch.no_grad():
            # Ensure model is in eval mode to get single output
            self.model.eval()
            
            # Get model outputs
            outputs = self.model(batch_tensor)
            
            # Handle both regular and auxiliary classifier models
            if isinstance(outputs, tuple):
                # If model returns tuple (shouldn't happen in eval mode but just in case)
                main_output = outputs[0]
            else:
                # Otherwise use the output directly
                main_output = outputs
            
            # Apply softmax to get probabilities
            probabilities = torch.softmax(main_output, dim=1)
            
            # Get predicted classes
            predicted_classes = torch.argmax(probabilities, dim=1).cpu().numpy()
        
        return predicted_classes.tolist()