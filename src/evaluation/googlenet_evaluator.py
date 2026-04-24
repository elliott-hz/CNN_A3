"""
Evaluation module for GoogLeNet with auxiliary classifiers.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


class GoogLeNetEvaluator:
    """
    Evaluator for GoogLeNet models with auxiliary classifiers.
    Handles evaluation in a way that ignores auxiliary outputs during inference.
    """
    
    def __init__(self, model: nn.Module, test_loader, device: torch.device = None):
        """
        Initialize evaluator.
        
        Args:
            model: Trained GoogLeNet model
            test_loader: Test data loader
            device: Device to evaluate on
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Ensure model is in eval mode
        self.model.eval()
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the model on test data."""
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                # Get model outputs (will be single tensor during eval)
                outputs = self.model(data)
                
                # Handle both regular and auxiliary classifier models
                if isinstance(outputs, tuple):
                    # If model returns tuple (during training), take main output
                    main_output = outputs[0]
                else:
                    # Otherwise use the output directly
                    main_output = outputs
                
                # Get predictions
                preds = torch.argmax(main_output, dim=1).cpu().numpy()
                targets_np = targets.cpu().numpy()
                
                all_preds.extend(preds)
                all_targets.extend(targets_np)
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='weighted'
        )
        
        # Detailed classification report
        class_precision, class_recall, class_f1, support = precision_recall_fscore_support(
            all_targets, all_preds, average=None
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "class_metrics": {
                "precision": class_precision.tolist(),
                "recall": class_recall.tolist(),
                "f1": class_f1.tolist(),
                "support": support.tolist()
            },
            "confusion_matrix": cm.tolist(),
            "predictions": all_preds,
            "targets": all_targets
        }
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: list = None, 
                             figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """Plot confusion matrix."""
        if class_names is None:
            class_names = [f"Class {i}" for i in range(cm.shape[0])]
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        return fig