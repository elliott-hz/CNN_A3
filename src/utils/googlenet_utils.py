"""
Utility functions for GoogLeNet model with auxiliary classifiers
Provides loss computation functions for training with auxiliary outputs
"""

import torch
import torch.nn as nn


def compute_combined_loss(main_output, aux1_output, aux2_output, targets, 
                         main_weight=1.0, aux1_weight=0.3, aux2_weight=0.3):
    """
    Compute combined loss from main and auxiliary outputs
    
    Args:
        main_output: Main classifier output (tensor)
        aux1_output: First auxiliary classifier output (tensor)
        aux2_output: Second auxiliary classifier output (tensor)
        targets: Ground truth labels (tensor)
        main_weight: Weight for main loss (default 1.0)
        aux1_weight: Weight for first auxiliary loss (default 0.3)
        aux2_weight: Weight for second auxiliary loss (default 0.3)
        
    Returns:
        total_loss: Combined weighted loss
        main_loss: Main classifier loss
        aux1_loss: First auxiliary classifier loss
        aux2_loss: Second auxiliary classifier loss
    """
    criterion = nn.CrossEntropyLoss()
    
    main_loss = criterion(main_output, targets)
    aux1_loss = criterion(aux1_output, targets)
    aux2_loss = criterion(aux2_output, targets)
    
    total_loss = main_weight * main_loss + aux1_weight * aux1_loss + aux2_weight * aux2_loss
    
    return total_loss, main_loss, aux1_loss, aux2_loss