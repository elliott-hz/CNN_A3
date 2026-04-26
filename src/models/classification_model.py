"""
Classification Model Definition
Multiple classifier architectures: ResNet50, AlexNet, GoogLeNet
Each with configurable parameters for different experiment variants
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Any


class ResNet50Classifier(nn.Module):
    """
    ResNet50 classifier with configurable parameters.
    
    This class wraps the ResNet50 model and allows for different configurations
    through the config parameter, enabling different experiment variants
    (baseline, modified v1, modified v2) using the same class.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ResNet50 classifier with configuration.
        
        Args:
            config: Dictionary with model configuration parameters
                - num_classes: Number of output classes (default 5)
                - dropout_rate: Dropout rate before final layer (default 0.5)
                - pretrained: Whether to use pretrained ImageNet weights (default True)
                - freeze_backbone: Whether to freeze backbone layers initially (default True)
                - additional_fc_layers: Whether to add additional FC layers (default False)
                - use_batch_norm: Whether to use batch normalization in custom layers (default True)
        """
        super(ResNet50Classifier, self).__init__()
        
        # Store configuration
        self.config = config
        self.num_classes = config.get('num_classes', 5)
        self.dropout_rate = config.get('dropout_rate', 0.5)
        self.pretrained = config.get('pretrained', True)
        self.freeze_backbone = config.get('freeze_backbone', True)
        self.additional_fc_layers = config.get('additional_fc_layers', False)
        self.use_batch_norm = config.get('use_batch_norm', True)
        
        # Load pretrained ResNet50
        if self.pretrained:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Get the number of features from the backbone
        num_features = self.backbone.fc.in_features
        
        # Freeze backbone if specified
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace the final fully connected layer
        if self.additional_fc_layers:
            # Add additional FC layers
            layers = []
            
            # First FC layer
            layers.append(nn.Linear(num_features, 512))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(512))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            
            # Second FC layer
            layers.append(nn.Linear(512, 256))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(256))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            
            # Final classification layer
            layers.append(nn.Linear(256, self.num_classes))
            
            self.classifier = nn.Sequential(*layers)
        else:
            # Simple single-layer classifier
            self.classifier = nn.Sequential(
                nn.Dropout(self.dropout_rate),
                nn.Linear(num_features, self.num_classes)
            )
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Classification logits
        """
        # Extract features using backbone (without final FC layer)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Apply classifier
        x = self.classifier(x)
        
        return x
    
    def unfreeze_backbone(self, unfreeze_all: bool = False, unfreeze_layer2: bool = False):
        """
        Unfreeze backbone layers for fine-tuning.
        
        Args:
            unfreeze_all: If True, unfreeze all layers. If False, only unfreeze later layers.
            unfreeze_layer2: If True, also unfreeze layer2 (for ResNet50 extended fine-tuning)
        """
        if unfreeze_all:
            # Unfreeze all layers
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Unfreeze layer3 and layer4 (standard partial unfreeze)
            for param in self.backbone.layer3.parameters():
                param.requires_grad = True
            for param in self.backbone.layer4.parameters():
                param.requires_grad = True
            
            # Also unfreeze layer2 if requested (extended fine-tuning)
            if unfreeze_layer2:
                for param in self.backbone.layer2.parameters():
                    param.requires_grad = True
                print("  Extended unfreeze: layer2 + layer3 + layer4")
            
            # Also unfreeze bn1 if present
            for param in self.backbone.bn1.parameters():
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
        # Separate parameters for backbone and classifier
        backbone_params = []
        classifier_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'backbone' in name:
                    backbone_params.append(param)
                else:
                    classifier_params.append(param)
        
        # Different learning rates for backbone and classifier
        param_groups = [
            {'params': classifier_params, 'lr': lr},
        ]
        
        if backbone_params:
            # Use lower learning rate for backbone
            param_groups.append({'params': backbone_params, 'lr': lr * 0.1})
        
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
            'config': self.config,
            'architecture': 'ResNet50Classifier'
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


class AlexNetClassifier(nn.Module):
    """
    AlexNet classifier with configurable parameters.
    
    This class wraps the AlexNet model and allows for different configurations
    through the config parameter.
    
    Architecture:
    - 5 convolutional layers with max pooling
    - 3 fully connected layers
    - Dropout for regularization
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize AlexNet classifier with configuration.
        
        Args:
            config: Dictionary with model configuration parameters
                - num_classes: Number of output classes (default 5)
                - dropout_rate: Dropout rate in classifier (default 0.5)
                - pretrained: Whether to use pretrained ImageNet weights (default True)
                - freeze_backbone: Whether to freeze feature extractor initially (default True)
        """
        super(AlexNetClassifier, self).__init__()
        
        # Store configuration
        self.config = config
        self.num_classes = config.get('num_classes', 5)
        self.dropout_rate = config.get('dropout_rate', 0.5)
        self.pretrained = config.get('pretrained', True)
        self.freeze_backbone = config.get('freeze_backbone', True)
        
        # Load pretrained AlexNet
        if self.pretrained:
            self.backbone = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.alexnet(weights=None)
        
        # Freeze backbone if specified
        if self.freeze_backbone:
            for param in self.backbone.features.parameters():
                param.requires_grad = False
        
        # Replace classifier head
        # Original AlexNet classifier: 9216 -> 4096 -> 4096 -> 1000
        # We replace with: 9216 -> 512 -> num_classes (simplified, no BatchNorm for stability)
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(9216, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, self.num_classes)
        )
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Classification logits
        """
        # Extract features using backbone
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Apply classifier
        x = self.classifier(x)
        
        return x
    
    def unfreeze_backbone(self, unfreeze_all: bool = False):
        """
        Unfreeze backbone layers for fine-tuning.
        
        Args:
            unfreeze_all: If True, unfreeze all layers. If False, only unfreeze later layers.
        """
        if unfreeze_all:
            # Unfreeze all feature layers
            for param in self.backbone.features.parameters():
                param.requires_grad = True
        else:
            # Only unfreeze last 2 conv layers (more task-specific)
            for param in self.backbone.features[3:].parameters():
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
        # Separate parameters for backbone and classifier
        backbone_params = []
        classifier_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'backbone' in name:
                    backbone_params.append(param)
                else:
                    classifier_params.append(param)
        
        # Different learning rates for backbone and classifier
        param_groups = [
            {'params': classifier_params, 'lr': lr},
        ]
        
        if backbone_params:
            # Use lower learning rate for backbone
            param_groups.append({'params': backbone_params, 'lr': lr * 0.1})
        
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
            'config': self.config,
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


class GoogLeNetClassifier(nn.Module):
    """
    GoogLeNet (Inception v1) classifier with configurable parameters.
    
    This class wraps the GoogLeNet model and allows for different configurations.
    Includes auxiliary classifiers for better gradient flow during training.
    
    Architecture:
    - Inception modules with parallel convolutions
    - Auxiliary classifiers at intermediate layers
    - Global average pooling before final classification
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize GoogLeNet classifier with configuration.
        
        Args:
            config: Dictionary with model configuration parameters
                - num_classes: Number of output classes (default 5)
                - dropout_rate: Dropout rate in classifier (default 0.5)
                - pretrained: Whether to use pretrained ImageNet weights (default True)
                - freeze_backbone: Whether to freeze backbone initially (default True)
                - use_auxiliary: Whether to use auxiliary classifiers during training (default True)
        """
        super(GoogLeNetClassifier, self).__init__()
        
        # Store configuration
        self.config = config
        self.num_classes = config.get('num_classes', 5)
        self.dropout_rate = config.get('dropout_rate', 0.5)
        self.pretrained = config.get('pretrained', True)
        self.freeze_backbone = config.get('freeze_backbone', True)
        self.use_auxiliary = config.get('use_auxiliary', True)
        
        # Load pretrained GoogLeNet
        if self.pretrained:
            self.backbone = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.googlenet(weights=None)
        
        # Freeze backbone if specified
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace main classifier
        # Original: 1024 -> 1000
        # Simplified: 1024 -> num_classes (removed intermediate FC layer for better gradient flow)
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(1024, self.num_classes)
        )
        
        # Replace auxiliary classifiers if present
        if hasattr(self.backbone, 'aux1') and self.use_auxiliary:
            self.backbone.aux1 = nn.Sequential(
                nn.Conv2d(512, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 768),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(768, self.num_classes)
            )
        
        if hasattr(self.backbone, 'aux2') and self.use_auxiliary:
            self.backbone.aux2 = nn.Sequential(
                nn.Conv2d(528, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 768),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(768, self.num_classes)
            )
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            Classification logits (or tuple with auxiliary outputs during training)
        """
        # Manually extract features from GoogLeNet backbone
        # Note: GoogLeNet uses BasicConv2d which includes Conv + BN + ReLU internally
        
        # GoogLeNet structure:
        # conv1 (Conv+BN+ReLU) -> maxpool1
        # conv2 (Conv+BN+ReLU) -> maxpool2
        # conv3 (Conv+BN+ReLU) -> maxpool3
        # inception3a -> inception3b -> maxpool3
        # inception4a -> inception4b -> inception4c -> inception4d -> inception4e -> maxpool4
        # inception5a -> inception5b -> avgpool -> dropout -> fc
        
        # Stage 1: Initial convolutions (BasicConv2d includes ReLU internally)
        x = self.backbone.conv1(x)  # BasicConv2d: Conv → BN → ReLU
        x = self.backbone.maxpool1(x)
        
        x = self.backbone.conv2(x)  # BasicConv2d: Conv → BN → ReLU
        x = self.backbone.maxpool2(x)
        
        x = self.backbone.conv3(x)  # BasicConv2d: Conv → BN → ReLU
        x = self.backbone.maxpool3(x)
        
        # Stage 2: Inception modules (stage 3)
        x = self.backbone.inception3a(x)
        x = self.backbone.inception3b(x)
        x = self.backbone.maxpool3(x)  # Note: maxpool3 used twice in original architecture
        
        # Stage 3: Inception modules (stage 4) + Auxiliary classifier 1
        x = self.backbone.inception4a(x)
        
        # Auxiliary classifier 1 output (from inception4a) - only if enabled and training
        aux1_out = None
        if self.use_auxiliary and self.training and hasattr(self.backbone, 'aux1') and self.backbone.aux1 is not None:
            aux1_out = self.backbone.aux1(x)
        
        x = self.backbone.inception4b(x)
        x = self.backbone.inception4c(x)
        x = self.backbone.inception4d(x)
        
        # Auxiliary classifier 2 output (from inception4d) - only if enabled and training
        aux2_out = None
        if self.use_auxiliary and self.training and hasattr(self.backbone, 'aux2') and self.backbone.aux2 is not None:
            aux2_out = self.backbone.aux2(x)

        x = self.backbone.inception4e(x)
        x = self.backbone.maxpool4(x)
        
        # Stage 4: Inception modules (stage 5)
        x = self.backbone.inception5a(x)
        x = self.backbone.inception5b(x)
        
        # Stage 5: Global average pooling → 1024-dim features
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)  # Shape: (batch_size, 1024)
        
        # Apply dropout before classifier (GoogLeNet has dropout before fc)
        if hasattr(self.backbone, 'dropout'):
            x = self.backbone.dropout(x)
        
        # Apply custom classifier
        main_logits = self.classifier(x)
        
        # Return based on training mode and auxiliary usage
        if self.training and self.use_auxiliary and aux1_out is not None:
            return main_logits, aux1_out, aux2_out
        else:
            return main_logits
    
    def unfreeze_backbone(self, unfreeze_all: bool = False):
        """
        Unfreeze backbone layers for fine-tuning.
        
        Args:
            unfreeze_all: If True, unfreeze all layers. If False, only unfreeze later layers.
        """
        if unfreeze_all:
            # Unfreeze all layers
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Unfreeze inception modules from later stages
            # GoogLeNet has mixed attributes, so we unfreeze most except very early layers
            for name, param in self.backbone.named_parameters():
                # Skip freezing the first few inception layers
                if not any(skip in name for skip in ['conv1', 'conv2', 'maxpool']):
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
        # Separate parameters for backbone and classifier
        backbone_params = []
        classifier_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'backbone' in name:
                    backbone_params.append(param)
                else:
                    classifier_params.append(param)
        
        # Different learning rates for backbone and classifier
        param_groups = [
            {'params': classifier_params, 'lr': lr},
        ]
        
        if backbone_params:
            # Use lower learning rate for backbone
            param_groups.append({'params': backbone_params, 'lr': lr * 0.1})
        
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
            'config': self.config,
            'architecture': 'GoogLeNetClassifier'
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


def create_classification_model(config: Dict[str, Any], architecture: str = 'resnet50'):
    """
    Factory function to create classification model instance.
    
    Args:
        config: Model configuration dictionary
        architecture: Model architecture ('resnet50', 'alexnet', 'googlenet')
        
    Returns:
        Classifier instance
    """
    if architecture.lower() == 'alexnet':
        return AlexNetClassifier(config)
    elif architecture.lower() == 'googlenet':
        return GoogLeNetClassifier(config)
    else:  # Default to ResNet50
        return ResNet50Classifier(config)


# Example configurations for different experiment variants
BASELINE_CLASSIFICATION_CONFIG = {
    'num_classes': 5,
    'dropout_rate': 0.5,
    'pretrained': True,
    'freeze_backbone': True,
    'additional_fc_layers': True,  # Enabled for standardized MLP head comparison
    'use_batch_norm': True
}

MODIFIED_V1_CLASSIFICATION_CONFIG = {
    'num_classes': 5,
    'dropout_rate': 0.7,  # Higher dropout
    'pretrained': True,
    'freeze_backbone': True,
    'additional_fc_layers': True,  # Additional FC layers
    'use_batch_norm': True
}

MODIFIED_V2_CLASSIFICATION_CONFIG = {
    'num_classes': 5,
    'dropout_rate': 0.3,  # Lower dropout
    'pretrained': True,
    'freeze_backbone': False,  # No freezing
    'additional_fc_layers': False,
    'use_batch_norm': True
}

# AlexNet specific configurations
ALEXNET_BASELINE_CONFIG = {
    'num_classes': 5,
    'dropout_rate': 0.5,
    'pretrained': True,
    'freeze_backbone': True
}

# GoogLeNet specific configurations
GOOGLENET_BASELINE_CONFIG = {
    'num_classes': 5,
    'dropout_rate': 0.5,
    'pretrained': True,
    'freeze_backbone': True,
    'use_auxiliary': True
}
