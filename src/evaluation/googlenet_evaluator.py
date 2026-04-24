"""
Custom evaluator for GoogLeNet model with auxiliary classifiers
Handles the evaluation of models with auxiliary outputs during inference
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from .classification_evaluator import ClassificationEvaluator


class CustomClassificationEvaluator(ClassificationEvaluator):
    """
    Evaluator class specifically designed to handle models with auxiliary classifiers.
    Extends the base ClassificationEvaluator but ensures that only the main classifier
    output is used during evaluation/inference.
    """
    
    def __init__(self, model, test_loader, device=None):
        """
        Initialize the custom evaluator for models with auxiliary classifiers
        
        Args:
            model: Model instance (should support auxiliary classifiers)
            test_loader: Test data loader
            device: Device to run evaluation on (defaults to CUDA if available)
        """
        super().__init__(model, test_loader, device)
        
        # Check if the model has auxiliary classifiers
        self.has_auxiliary = (
            hasattr(self.model, 'aux_classifier1') and 
            hasattr(self.model, 'aux_classifier2')
        )
    
    def evaluate(self, dataloader=None):
        """
        Evaluate the model performance.
        
        Args:
            dataloader: Dataloader for evaluation (uses self.test_loader if None)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if dataloader is None:
            dataloader = self.test_loader
            
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Use model in evaluation mode (no auxiliary outputs)
                # This ensures that the forward method returns only the main output
                if self.has_auxiliary:
                    outputs = self.model(inputs, use_auxiliary=False)
                else:
                    outputs = self.model(inputs)
                
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        report = classification_report(
            all_targets, all_predictions, 
            target_names=['Alert', 'Angry', 'Frown', 'Happy', 'Relax'],
            output_dict=True
        )
        conf_matrix = confusion_matrix(all_targets, all_predictions)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def predict_single(self, image_tensor):
        """
        Make a prediction on a single image.
        
        Args:
            image_tensor: Image tensor to predict
            
        Returns:
            Predicted class index and probability
        """
        self.model.eval()
        
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            # Ensure only main classifier output is used
            if self.has_auxiliary:
                output = self.model(image_tensor, use_auxiliary=False)
            else:
                output = self.model(image_tensor)
            
            probabilities = torch.softmax(output, dim=1)
            predicted_class_idx = torch.argmax(output, dim=1).item()
            confidence = probabilities[0][predicted_class_idx].item()
        
        return {
            'class_idx': predicted_class_idx,
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy()
        }