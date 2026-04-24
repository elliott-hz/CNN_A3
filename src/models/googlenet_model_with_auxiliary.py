"""
GoogleNet model with auxiliary classifiers for improved training.
This implementation extends the base GoogLeNetClassifier with two auxiliary classifiers
as described in the original GoogLeNet paper to combat vanishing gradients.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
from .googlenet_model import GoogLeNetClassifier, InceptionBlock, AuxiliaryClassifier


class GoogLeNetClassifierWithAuxiliary(GoogLeNetClassifier):
    """
    GoogLeNet classifier with auxiliary classifiers enabled.
    
    This class adds two auxiliary classifiers to the GoogLeNet architecture
    to improve training by providing additional gradient signals.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize GoogLeNet classifier with auxiliary classifiers.
        
        Args:
            config: Dictionary with model configuration parameters (optional)
                - num_classes: Number of output classes (default 5)
                - dropout_rate: Dropout rate (default 0.4)
        """
        super(GoogLeNetClassifierWithAuxiliary, self).__init__(config)
        
        # Enable auxiliary classifiers
        self.aux_classifier1 = AuxiliaryClassifier(512, self.num_classes)
        self.aux_classifier2 = AuxiliaryClassifier(528, self.num_classes)
    
    def forward(self, x):
        """
        Forward pass through the model with auxiliary classifiers during training.
        
        Args:
            x: Input tensor (batch_size, channels, height, width)
            
        Returns:
            During training: tuple of (main_output, aux1_output, aux2_output)
            During inference: main_output only
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
        
        # First auxiliary classifier (after inception4a equivalent position)
        x = self.inception4a(x)
        aux1 = None
        if self.training and hasattr(self, 'aux_classifier1'):
            aux1 = self.aux_classifier1(x)
        
        x = self.inception4b(x)
        x = self.inception4c(x)
        
        # Second auxiliary classifier (after inception4e equivalent position)
        aux2 = None
        if self.training and hasattr(self, 'aux_classifier2'):
            aux2 = self.aux_classifier2(x)
        
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        
        # Flatten and classify
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        main_output = self.classifier(x)
        
        if self.training:
            return main_output, aux1, aux2
        else:
            # Return only main classifier output during inference
            return main_output