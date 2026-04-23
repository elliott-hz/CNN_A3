"""
AlexNet Model Definition
Implementation of AlexNet architecture for dog emotion classification
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class AlexNetClassifier(nn.Module):
    """
    AlexNet classifier with standard configuration.
    
    This class implements the standard AlexNet architecture as described
    in the original paper by Krizhevsky et al.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize AlexNet classifier with standard configuration.
        
        Args:
            config: Dictionary with model configuration parameters (optional)
                - num_classes: Number of output classes (default 5)
                - dropout_rate: Dropout rate in classifier layers (default 0.5)
        """
        super(AlexNetClassifier, self).__init__()
        
        # Set default config if none provided
        if config is None:
            config = {}
        
        # Store configuration
        self.num_classes = config.get('num_classes', 5)
        self.dropout_rate = config.get('dropout_rate', 0.5)
        self.freeze_backbone = config.get('freeze_backbone', False)
        
        # Define the AlexNet feature extractor (convolutional layers)
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Second conv block
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Third conv block
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Fourth conv block
            nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Fifth conv block
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # Adaptive pooling to ensure consistent output size regardless of input
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Define the classifier (fully connected layers) - Standard AlexNet structure
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, self.num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Classification logits
        """
        # Extract features using the convolutional layers
        x = self.features(x)
        
        # Apply adaptive average pooling
        x = self.avgpool(x)
        
        # Flatten the output for the classifier
        x = torch.flatten(x, 1)
        
        # Apply the classifier
        x = self.classifier(x)
        
        return x
    
    def unfreeze_backbone(self, unfreeze_all: bool = False):
        """
        Unfreeze feature extractor layers for fine-tuning.
        Note: Since AlexNet is implemented from scratch, all layers are trainable by default.
        
        Args:
            unfreeze_all: If True, unfreeze all layers. If False, follow the same behavior.
        """
        # For AlexNet from scratch, all layers are trainable by default
        # This method is kept for compatibility with the training pipeline
        for param in self.features.parameters():
            param.requires_grad = True
    
    def get_optimizer(self, lr: float = 1e-3, weight_decay: float = 1e-4, 
                     optimizer_type: str = 'adam'):
        """
        Create optimizer based on configuration.
        
        Args:
            lr: Learning rate
            weight_decay: Weight decay (L2 regularization)
            optimizer_type: Type of optimizer ('sgd', 'adam', 'adamw')
            
        Returns:
            PyTorch optimizer
        """
        # Separate parameters for features and classifier
        feature_params = []
        classifier_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'features' in name:
                    feature_params.append(param)
                else:
                    classifier_params.append(param)
        
        # Different learning rates for features and classifier
        param_groups = [
            {'params': classifier_params, 'lr': lr},
        ]
        
        if feature_params:
            # Use lower learning rate for features
            param_groups.append({'params': feature_params, 'lr': lr * 0.1})
        
        # Create optimizer
        if optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                param_groups, 
                momentum=0.9, 
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                param_groups, 
                weight_decay=weight_decay
            )
        else:  # Default to Adam
            optimizer = torch.optim.Adam(
                param_groups, 
                weight_decay=weight_decay
            )
        
        return optimizer
    
    def save(self, save_path: str):
        """
        Save the model state dict and configuration.
        
        Args:
            save_path: Path to save the model
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': {'num_classes': self.num_classes, 'dropout_rate': self.dropout_rate},
            'architecture': 'AlexNetClassifier'
        }
        torch.save(checkpoint, save_path)
    
    @classmethod
    def load(cls, load_path: str, map_location=None):
        """
        Load a saved model.
        
        Args:
            load_path: Path to saved model
            map_location: Device mapping for loading
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(load_path, map_location=map_location)
        
        # Create new instance with saved config
        model = cls(checkpoint['config'])
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model


def create_alexnet_model(config: Dict[str, Any] = None) -> AlexNetClassifier:
    """
    Factory function to create AlexNet model instance.
    
    Args:
        config: Model configuration dictionary (optional)
        
    Returns:
        AlexNetClassifier instance
    """
    return AlexNetClassifier(config)


# Standard AlexNet configuration
STANDARD_ALEXNET_CONFIG = {
    'num_classes': 5,
    'dropout_rate': 0.5,
    'freeze_backbone': False,  # Don't freeze backbone for AlexNet by default
}