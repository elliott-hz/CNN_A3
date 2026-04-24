"""
Utility functions for GoogLeNet with auxiliary classifiers.
"""

import torch
import torch.nn as nn
from typing import Tuple


def compute_combined_loss(
    main_output: torch.Tensor,
    aux1_output: torch.Tensor,
    aux2_output: torch.Tensor,
    targets: torch.Tensor,
    criterion: nn.Module,
    main_weight: float = 1.0,
    aux1_weight: float = 0.3,
    aux2_weight: float = 0.3
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute combined loss for GoogLeNet with auxiliary classifiers.
    
    Args:
        main_output: Main classifier output
        aux1_output: First auxiliary classifier output
        aux2_output: Second auxiliary classifier output
        targets: Ground truth labels
        criterion: Loss function to use
        main_weight: Weight for main classifier loss
        aux1_weight: Weight for first auxiliary classifier loss
        aux2_weight: Weight for second auxiliary classifier loss
        
    Returns:
        Combined loss and individual losses
    """
    main_loss = criterion(main_output, targets)
    aux1_loss = criterion(aux1_output, targets) if aux1_output is not None else torch.tensor(0.0, device=targets.device)
    aux2_loss = criterion(aux2_output, targets) if aux2_output is not None else torch.tensor(0.0, device=targets.device)
    
    total_loss = (
        main_weight * main_loss +
        aux1_weight * aux1_loss +
        aux2_weight * aux2_loss
    )
    
    return total_loss, main_loss, aux1_loss, aux2_loss