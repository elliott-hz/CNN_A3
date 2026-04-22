"""
Classification Evaluator
Evaluation metrics and visualization for classification models
"""

import torch
import numpy as np
from pathlib import Path
import json
import csv
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report)
from typing import Dict, Any, List


class ClassificationEvaluator:
    """
    Evaluation framework for classification models.
    
    Calculates accuracy, precision, recall, F1-score, confusion matrix, and generates reports.
    """
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize evaluator.
        
        Args:
            class_names: List of class names for reporting
        """
        self.class_names = class_names or ['angry', 'happy', 'relax', 'frown', 'alert']
        self.metrics = {}
    
    def evaluate(self, model, X_test, y_test, output_dir: str):
        """
        Evaluate model on test set.
        
        Args:
            model: ResNet50Classifier model
            X_test: Test images
            y_test: Test labels
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("=" * 80)
        print("CLASSIFICATION MODEL EVALUATION")
        print("=" * 80)
        
        figures_dir = Path(output_dir) / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Get predictions
        y_pred = self._get_predictions(model, X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        per_class_precision = precision_score(y_test, y_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_test, y_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store metrics
        self.metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'per_class_precision': per_class_precision.tolist(),
            'per_class_recall': per_class_recall.tolist(),
            'per_class_f1': per_class_f1.tolist(),
            'confusion_matrix': cm.tolist()
        }
        
        # Print results
        print(f"\nOverall Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        print(f"\nPer-Class Metrics:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {class_name}:")
            print(f"    Precision: {per_class_precision[i]:.4f}")
            print(f"    Recall: {per_class_recall[i]:.4f}")
            print(f"    F1-Score: {per_class_f1[i]:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Save metrics
        metrics_path = Path(output_dir) / "logs" / "evaluation_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        # Save detailed classification report
        report = classification_report(y_test, y_pred, target_names=self.class_names)
        report_path = Path(output_dir) / "logs" / "classification_report.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"\nMetrics saved to: {metrics_path}")
        print(f"Detailed report saved to: {report_path}")
        
        return self.metrics
    
    def _get_predictions(self, model, X_test) -> np.ndarray:
        """
        Get model predictions on test data.
        
        Args:
            model: Trained model
            X_test: Test images
            
        Returns:
            Predicted labels
        """
        model.eval()
        device = next(model.parameters()).device
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_test).permute(0, 3, 1, 2).to(device)
        
        # Get predictions in batches
        all_preds = []
        batch_size = 64
        
        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                outputs = model(batch)
                _, predicted = outputs.max(1)
                all_preds.append(predicted.cpu().numpy())
        
        return np.concatenate(all_preds)
    
    def generate_report(self, output_dir: str):
        """
        Generate comprehensive evaluation report in markdown format.
        
        Args:
            output_dir: Directory containing experiment outputs
        """
        report_path = Path(output_dir) / "logs" / "experiment_report.md"
        
        # Build per-class metrics table
        per_class_rows = ""
        for i, class_name in enumerate(self.class_names):
            per_class_rows += f"| {class_name} | {self.metrics['per_class_precision'][i]:.4f} | {self.metrics['per_class_recall'][i]:.4f} | {self.metrics['per_class_f1'][i]:.4f} |\n"
        
        report = f"""# Experiment Report: Classification Model

## Overall Metrics

| Metric | Value |
|--------|-------|
| Accuracy | {self.metrics.get('accuracy', 'N/A'):.4f} |
| Precision | {self.metrics.get('precision', 'N/A'):.4f} |
| Recall | {self.metrics.get('recall', 'N/A'):.4f} |
| F1-Score | {self.metrics.get('f1_score', 'N/A'):.4f} |

## Per-Class Metrics

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
{per_class_rows}

## Confusion Matrix

```
{np.array(self.metrics['confusion_matrix'])}
```

## Figures

See the `figures/` directory for visualization outputs:
- Confusion matrix heatmap
- ROC curves (one-vs-rest)
- Per-class metric bar charts
"""
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to: {report_path}")
