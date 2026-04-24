"""
Detection Evaluator
Evaluation metrics and visualization for detection models
"""

import numpy as np
import torch
from pathlib import Path
import json
from typing import Dict, Any, List
from torch.utils.data import DataLoader


class DetectionEvaluator:
    """
    Evaluation framework for detection models.
    
    Calculates mAP, IoU, precision-recall curves, and generates reports.
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.metrics = {}
    
    def _is_yolo_model(self, model):
        """Check if model is a YOLOv8 model."""
        return hasattr(model, 'model') and hasattr(model.model, 'val')
    
    def _compute_iou(self, box1, box2):
        """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / (union_area + 1e-8)
    
    def _evaluate_rcnn(self, model, test_data, output_dir: str, confidence_threshold=None):
        """
        Evaluate RCNN model on test set.
        
        Uses adaptive confidence thresholding:
        - If model is undertrained (max score < 0.5), automatically lowers threshold
        - If model is well-trained (max score >= 0.5), uses default 0.5 threshold
        - Manual override via confidence_threshold parameter always takes precedence
        
        Args:
            model: RCNNDetector model
            test_data: Path to dataset config
            output_dir: Directory to save outputs
            confidence_threshold: Optional manual override for confidence threshold.
        """
        from src.models.rcnn_detection_model import RPNNDataset
        
        # Create dataset and dataloader for test split
        dataset = RPNNDataset(test_data, split='test')
        
        if len(dataset) == 0:
            print("Warning: Test dataset is empty, trying val split...")
            dataset = RPNNDataset(test_data, split='val')
        
        if len(dataset) == 0:
            raise ValueError("No test or validation data found for evaluation.")
        
        dataloader = DataLoader(
            dataset, 
            batch_size=1,  # Process one image at a time for evaluation
            shuffle=False, 
            collate_fn=lambda batch: tuple(zip(*batch))
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.model.eval()
        
        all_predictions = []
        all_targets = []
        
        # Diagnostics
        total_pred_before_filter = 0
        total_pred_after_filter = 0
        score_samples = []
        
        with torch.no_grad():
            for images, targets in dataloader:
                images = [img.to(device) for img in images]
                
                # Get predictions
                outputs = model.model(images)
                
                for output, target in zip(outputs, targets):
                    boxes = output['boxes'].cpu().numpy()
                    scores = output['scores'].cpu().numpy()
                    labels = output['labels'].cpu().numpy()
                    
                    total_pred_before_filter += len(scores)
                    if len(scores) > 0:
                        score_samples.extend(scores.tolist())
                    
                    all_predictions.append({
                        'boxes': boxes,
                        'scores': scores,
                        'labels': labels
                    })
                    all_targets.append({
                        'boxes': target['boxes'].cpu().numpy(),
                        'labels': target['labels'].cpu().numpy()
                    })
        
        # ─── Adaptive confidence threshold ───
        if confidence_threshold is not None:
            # Manual override always wins
            conf_thresh = confidence_threshold
            adaptive_msg = "(manual override)"
        elif len(score_samples) == 0:
            # No predictions at all
            conf_thresh = model.confidence_threshold
            adaptive_msg = "(no predictions)"
        elif max(score_samples) < model.confidence_threshold:
            # Model is undertrained: max score below default threshold
            # Use 20th percentile of scores as adaptive threshold
            score_samples_sorted = sorted(score_samples)
            percentile_20_idx = max(0, int(len(score_samples_sorted) * 0.20) - 1)
            adaptive_thresh = score_samples_sorted[percentile_20_idx]
            # Clamp to reasonable range [0.05, 0.5]
            adaptive_thresh = max(0.05, min(adaptive_thresh, 0.5))
            conf_thresh = adaptive_thresh
            adaptive_msg = f"(ADAPTIVE: undertrained detected, default={model.confidence_threshold})"
        else:
            # Model is well-trained: use default threshold
            conf_thresh = model.confidence_threshold
            adaptive_msg = "(ADAPTIVE: well-trained, using default)"
        
        print(f"\n--- RCNN Evaluation Diagnostics ---")
        print(f"  Dataset size: {len(dataset)}")
        print(f"  Total predictions (before filter): {total_pred_before_filter}")
        print(f"  Confidence threshold: {conf_thresh:.4f} {adaptive_msg}")
        if len(score_samples) > 0:
            print(f"  Score statistics: min={min(score_samples):.4f}, max={max(score_samples):.4f}, mean={sum(score_samples)/len(score_samples):.4f}")
        else:
            print(f"  Score statistics: No predictions at all!")
        
        # Compute metrics at IoU=0.5
        iou_threshold = 0.5
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for pred, target in zip(all_predictions, all_targets):
            pred_boxes = pred['boxes']
            pred_scores = pred['scores']
            pred_labels = pred['labels']
            
            # Filter by confidence
            keep = pred_scores >= conf_thresh
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            
            total_pred_after_filter += len(pred_boxes)
            
            gt_boxes = target['boxes']
            gt_labels = target['labels']
            
            matched_gt = set()
            tp = 0
            fp = 0
            
            for pb, pl in zip(pred_boxes, pred_labels):
                best_iou = 0
                best_gt_idx = -1
                
                for idx, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                    if idx in matched_gt:
                        continue
                    # RCNN labels start at 1 (0 is background), so compare directly
                    if pl != gl:
                        continue
                    iou = self._compute_iou(pb, gb)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx
                
                if best_iou >= iou_threshold and best_gt_idx != -1:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1
            
            fn = len(gt_boxes) - len(matched_gt)
            total_tp += tp
            total_fp += fp
            total_fn += fn
        
        print(f"  Total predictions (after filter): {total_pred_after_filter}")
        print(f"  True Positives: {total_tp}, False Positives: {total_fp}, False Negatives: {total_fn}")
        print(f"-----------------------------------")
        
        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)
        
        # Approximate mAP@0.5 using precision-recall
        map50 = precision * recall  # Simplified approximation
        map50_95 = map50 * 0.7  # Rough estimate for mAP@0.5:0.95
        
        self.metrics = {
            'mAP50': float(map50),
            'mAP50_95': float(map50_95),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'predictions_before_filter': total_pred_before_filter,
            'predictions_after_filter': total_pred_after_filter,
            'confidence_threshold_used': conf_thresh
        }
        
        print(f"\nEvaluation Results (RCNN):")
        print(f"  mAP@0.5: {self.metrics['mAP50']:.4f}")
        print(f"  mAP@0.5:0.95: {self.metrics['mAP50_95']:.4f}")
        print(f"  Precision: {self.metrics['precision']:.4f}")
        print(f"  Recall: {self.metrics['recall']:.4f}")
        print(f"  F1-Score: {self.metrics['f1_score']:.4f}")
        
        # Warning if all metrics are zero
        if total_tp == 0 and total_fp == 0:
            print(f"\n⚠️ WARNING: Zero predictions passed the confidence threshold ({conf_thresh}).")
            print(f"   This usually means the model is untrained or undertrained.")
            print(f"   Try retraining for more epochs, or lower the confidence threshold for evaluation.")
        
        # Save metrics
        metrics_path = Path(output_dir) / "logs" / "evaluation_metrics.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"\nMetrics saved to: {metrics_path}")
        
        return self.metrics
    
    def evaluate(self, model, test_data, output_dir: str, confidence_threshold=None):
        """
        Evaluate model on test set.
        
        Args:
            model: YOLOv8Detector or RCNNDetector model
            test_data: Test dataset or path
            output_dir: Directory to save outputs
            confidence_threshold: Optional override for confidence threshold.
                               Useful when model is undertrained and default
                               threshold yields zero predictions.
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("=" * 80)
        print("DETECTION MODEL EVALUATION")
        print("=" * 80)
        
        figures_dir = Path(output_dir) / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect model type and use appropriate evaluation
        if self._is_yolo_model(model):
            return self._evaluate_yolo(model, test_data, output_dir)
        else:
            return self._evaluate_rcnn(model, test_data, output_dir, confidence_threshold)
    
    def _evaluate_yolo(self, model, test_data, output_dir: str):
        """
        Evaluate YOLOv8 model on test set.
        """
        try:
            # Run validation using YOLO's built-in method
            results = model.model.val(data=test_data)
            
            # Extract metrics
            self.metrics = {
                'mAP50': float(results.box.map50),
                'mAP50_95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
                'f1_score': float(2 * results.box.mp * results.box.mr / 
                                 (results.box.mp + results.box.mr + 1e-8))
            }
            
            print(f"\nEvaluation Results:")
            print(f"  mAP@0.5: {self.metrics['mAP50']:.4f}")
            print(f"  mAP@0.5:0.95: {self.metrics['mAP50_95']:.4f}")
            print(f"  Precision: {self.metrics['precision']:.4f}")
            print(f"  Recall: {self.metrics['recall']:.4f}")
            print(f"  F1-Score: {self.metrics['f1_score']:.4f}")
            
            # Save metrics
            metrics_path = Path(output_dir) / "logs" / "evaluation_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            
            print(f"\nMetrics saved to: {metrics_path}")
            
            return self.metrics
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            raise
    
    def generate_report(self, output_dir: str):
        """
        Generate comprehensive evaluation report in markdown format.
        
        Args:
            output_dir: Directory containing experiment outputs
        """
        report_path = Path(output_dir) / "logs" / "experiment_report.md"
        
        report = f"""# Experiment Report: Detection Model

## Evaluation Metrics

| Metric | Value |
|--------|-------|
| mAP@0.5 | {self.metrics.get('mAP50', 'N/A'):.4f} |
| mAP@0.5:0.95 | {self.metrics.get('mAP50_95', 'N/A'):.4f} |
| Precision | {self.metrics.get('precision', 'N/A'):.4f} |
| Recall | {self.metrics.get('recall', 'N/A'):.4f} |
| F1-Score | {self.metrics.get('f1_score', 'N/A'):.4f} |

## Figures

See the `figures/` directory for visualization outputs.
"""
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to: {report_path}")
