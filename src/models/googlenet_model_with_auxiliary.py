"""
GoogLeNet Model Definition with Auxiliary Classifiers
Implementation of GoogLeNet architecture with auxiliary classifiers for dog emotion classification
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class InceptionBlock(nn.Module):
    """
    Inception block as described in the GoogLeNet paper.
    Each inception block contains parallel conv layers of different sizes.
    """
    
    def __init__(self, in_channels: int, ch1x1: int, ch3x3_reduce: int, ch3x3: int, 
                 ch5x5_reduce: int, ch5x5: int, pool_proj: int):
        super(InceptionBlock, self).__init__()
        
        # 1x1 conv branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 -> 3x3 conv branch
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 -> 5x5 conv branch
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )
        
        # 3x3 pool -> 1x1 conv branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class AuxiliaryClassifier(nn.Module):
    """
    Auxiliary classifier to improve training of GoogLeNet
    """
    
    def __init__(self, in_channels: int, num_classes: int = 5):
        super(AuxiliaryClassifier, self).__init__()
        
        self.avg_pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)  # Assuming input is 224x224
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p=0.7)
        
    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GoogLeNetClassifierWithAuxiliary(nn.Module):
    """
    GoogLeNet classifier with auxiliary classifiers for improved training.
    
    This class implements the full GoogLeNet architecture as described
    in the original paper by Szegedy et al., including auxiliary classifiers
    that help with gradient flow during training.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize GoogLeNet classifier with auxiliary classifiers.
        
        Args:
            config: Dictionary with model configuration parameters (optional)
                - num_classes: Number of output classes (default 5)
                - dropout_rate: Dropout rate (default 0.4)
                - freeze_backbone: Whether to freeze feature extractor (default False)
        """
        super(GoogLeNetClassifierWithAuxiliary, self).__init__()
        
        # Set default config if none provided
        if config is None:
            config = {}
        
        # Store configuration
        self.num_classes = config.get('num_classes', 5)
        self.dropout_rate = config.get('dropout_rate', 0.4)
        self.freeze_backbone = config.get('freeze_backbone', False)
        
        # Initial conv layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.lrn1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.lrn2 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)
        
        # Inception blocks
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Auxiliary classifier 1 (after inception4a)
        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.aux_classifier1 = AuxiliaryClassifier(512, self.num_classes)
        
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        
        # Auxiliary classifier 2 (after inception4d)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.aux_classifier2 = AuxiliaryClassifier(528, self.num_classes)
        
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout and classifier
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.classifier = nn.Linear(1024, self.num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, use_auxiliary=True):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            use_auxiliary: Whether to return auxiliary outputs during training (default True)
            
        Returns:
            During training: tuple of (main_output, aux1_output, aux2_output)
            During evaluation: main_output only
        """
        # Initial layers
        x = self.conv1(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.maxpool1(x)
        x = self.lrn1(x)
        
        x = self.conv2(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.conv3(x)
        x = nn.functional.relu(x, inplace=True)
        x = self.maxpool2(x)
        x = self.lrn2(x)
        
        # Inception blocks
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        # Inception 4a and auxiliary classifier 1
        aux1_input = self.inception4a(x)
        if self.training and use_auxiliary:
            aux1 = self.aux_classifier1(aux1_input)
        x = self.inception4b(aux1_input)
        x = self.inception4c(x)
        
        # Inception 4d and auxiliary classifier 2
        aux2_input = self.inception4d(x)
        if self.training and use_auxiliary:
            aux2 = self.aux_classifier2(aux2_input)
        x = self.inception4e(aux2_input)
        x = self.maxpool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        
        # Flatten and classify
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        main_output = self.classifier(x)
        
        if self.training and use_auxiliary:
            return main_output, aux1, aux2
        else:
            # During evaluation/inference, return only main output
            # This ensures compatibility with existing inference/evaluation components
            return main_output
    
    def unfreeze_backbone(self, unfreeze_all: bool = False):
        """
        Unfreeze feature extractor layers for fine-tuning.
        Note: Since GoogLeNet is implemented from scratch, all layers are trainable by default.
        
        Args:
            unfreeze_all: If True, unfreeze all layers. If False, follow the same behavior.
        """
        # For GoogLeNet from scratch, all layers are trainable by default
        # This method is kept for compatibility with the training pipeline
        for param in self.parameters():
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
        # Create optimizer
        if optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(), 
                lr=lr,
                momentum=0.9, 
                weight_decay=weight_decay
            )
        elif optimizer_type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=lr,
                weight_decay=weight_decay
            )
        else:  # Default to Adam
            optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=lr,
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
            'config': {
                'num_classes': self.num_classes, 
                'dropout_rate': self.dropout_rate,
                'freeze_backbone': self.freeze_backbone
            },
            'architecture': 'GoogLeNetClassifierWithAuxiliary'
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


def create_googlenet_with_auxiliary_model(config: Dict[str, Any] = None) -> GoogLeNetClassifierWithAuxiliary:
    """
    Factory function to create GoogLeNet model with auxiliary classifiers instance.
    
    Args:
        config: Model configuration dictionary (optional)
        
    Returns:
        GoogLeNetClassifierWithAuxiliary instance
    """
    return GoogLeNetClassifierWithAuxiliary(config)


# Standard GoogLeNet with auxiliary configuration
STANDARD_GOOGLENET_WITH_AUX_CONFIG = {
    'num_classes': 5,
    'dropout_rate': 0.4,
    'freeze_backbone': False,  # Don't freeze backbone for GoogLeNet by default
    'use_auxiliary': True
}