"""
Custom trainer for GoogLeNet model with auxiliary classifiers
Handles the training loop for models with auxiliary outputs
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from .classification_trainer import ClassificationTrainer
from ..utils.googlenet_utils import compute_combined_loss


class CustomClassificationTrainer(ClassificationTrainer):
    """
    Trainer class specifically designed to handle models with auxiliary classifiers.
    Extends the base ClassificationTrainer to support combined loss computation
    when auxiliary classifiers are present.
    """
    
    def __init__(self, model, train_loader: DataLoader, val_loader: DataLoader, 
                 config: Dict[str, Any], device: Optional[torch.device] = None):
        """
        Initialize the custom trainer for models with auxiliary classifiers
        
        Args:
            model: Model instance (should support auxiliary classifiers)
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: Device to run training on (defaults to CUDA if available)
        """
        super().__init__(model, train_loader, val_loader, config, device)
        
        # Extract auxiliary loss weights from config
        self.main_weight = config.get('main_weight', 1.0)
        self.aux1_weight = config.get('aux1_weight', 0.3)
        self.aux2_weight = config.get('aux2_weight', 0.3)
        
        # Check if the model has auxiliary classifiers
        self.has_auxiliary = (
            hasattr(self.model, 'aux_classifier1') and 
            hasattr(self.model, 'aux_classifier2')
        )
    
    def train_step(self, batch):
        """
        Perform a single training step.
        
        Args:
            batch: A batch of training data (inputs, targets)
            
        Returns:
            Loss value for the batch
        """
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        
        self.optimizer.zero_grad()
        
        if self.has_auxiliary:
            # Get outputs from model with auxiliary classifiers
            outputs = self.model(inputs, use_auxiliary=True)
            
            if isinstance(outputs, tuple) and len(outputs) == 3:
                main_out, aux1_out, aux2_out = outputs
                loss, main_loss, aux1_loss, aux2_loss = compute_combined_loss(
                    main_out, aux1_out, aux2_out, targets,
                    self.main_weight, self.aux1_weight, self.aux2_weight
                )
            else:
                # Fallback to regular output if model doesn't return auxiliary outputs
                outputs = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
                loss = self.criterion(outputs, targets)
        else:
            # Regular forward pass if no auxiliary classifiers
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
        
        loss.backward()
        
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self):
        """
        Perform validation step.
        
        Returns:
            Average validation loss and accuracy
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                if self.has_auxiliary:
                    # In validation, model should return only main output
                    outputs = self.model(inputs, use_auxiliary=False)
                    
                    # Calculate loss using only main classifier output
                    loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy