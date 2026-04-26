"""
Detection Evaluator
Comprehensive evaluation metrics and visualization for detection models
Supports YOLOv8, Faster R-CNN, and SSD models
"""

import numpy as np
import torch
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import cv2


class DetectionEvaluator:
    """
    Comprehensive evaluation framework for object detection models.
    
    Calculates:
    - mAP@0.5 and mAP@0.5:0.95
    - Precision-Recall curves
    - IoU distribution
    - Per-class metrics
    - Confusion matrix (for multi-class)
    """
    
    def __init__(self, iou_thresholds: List[float] = None):
        """
        Initialize evaluator.
        
        Args:
            iou_thresholds: IoU thresholds for mAP calculation
                           Default: [0.5, 0.55, 0.6, ..., 0.95] for mAP@0.5:0.95
        """
        if iou_thresholds is None:
            self.iou_thresholds = np.arange(0.5, 0.96, 0.05).tolist()
        else:
            self.iou_thresholds = iou_thresholds
        
        self.metrics = {}
        self.class_names = []
    
    def evaluate(self, model, test_dataset, output_dir: str, 
                 model_type: str = 'torchvision', conf_threshold: float = 0.5):
        """
        Evaluate model on test set.
        
        Args:
            model: Detection model (YOLOv8Detector, FasterRCNNDetector, or SSDDetector)
            test_dataset: Test dataset (Dataset object for torchvision, path string for YOLOv8)
            output_dir: Directory to save outputs
            model_type: 'yolov8' or 'torchvision'
            conf_threshold: Confidence threshold for detections
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("=" * 80)
        print("DETECTION MODEL EVALUATION")
        print("=" * 80)
        
        figures_dir = Path(output_dir) / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if model_type == 'yolov8':
                # Use YOLOv8's built-in validation
                results = model.model.val(data=test_dataset)
                
                self.metrics = {
                    'mAP50': float(results.box.map50),
                    'mAP50_95': float(results.box.map),
                    'precision': float(results.box.mp),
                    'recall': float(results.box.mr),
                    'f1_score': float(2 * results.box.mp * results.box.mr / 
                                     (results.box.mp + results.box.mr + 1e-8))
                }
            else:
                # Manual evaluation for torchvision models
                self.metrics = self._evaluate_torchvision_model(
                    model, test_dataset, conf_threshold
                )
            
            print(f"\nEvaluation Results:")
            print(f"  mAP@0.5: {self.metrics['mAP50']:.4f}")
            print(f"  mAP@0.5:0.95: {self.metrics['mAP50_95']:.4f}")
            print(f"  Precision: {self.metrics['precision']:.4f}")
            print(f"  Recall: {self.metrics['recall']:.4f}")
            print(f"  F1-Score: {self.metrics['f1_score']:.4f}")
            
            if 'per_class_metrics' in self.metrics:
                print(f"\nPer-class Metrics:")
                for class_name, class_metrics in self.metrics['per_class_metrics'].items():
                    print(f"  {class_name}:")
                    print(f"    AP@0.5: {class_metrics.get('AP50', 0):.4f}")
                    print(f"    Precision: {class_metrics.get('precision', 0):.4f}")
                    print(f"    Recall: {class_metrics.get('recall', 0):.4f}")
            
            # Save metrics
            metrics_path = Path(output_dir) / "logs" / "evaluation_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
            
            print(f"\nMetrics saved to: {metrics_path}")
            
            # Generate visualizations
            self._generate_visualizations(figures_dir)
            
            return self.metrics
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _evaluate_torchvision_model(self, model, test_dataset, conf_threshold: float = 0.5):
        """
        Evaluate torchvision-based models (Faster R-CNN, SSD).
        
        Implements COCO-style mAP calculation.
        
        Args:
            model: Detection model
            test_dataset: Test dataset
            conf_threshold: Confidence threshold for filtering predictions
            
        Note: For poorly trained models, consider lowering conf_threshold to 0.1 or 0.01
        """
        from torch.utils.data import DataLoader
        
        model.eval()
        device = next(model.parameters()).device
        
        # Collect all predictions and ground truths
        all_predictions = []
        all_ground_truths = []
        
        # Custom collate function for detection
        def collate_fn(batch):
            return tuple(zip(*batch))
        
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                collate_fn=collate_fn, num_workers=2)
        
        print(f"\nEvaluating on {len(test_loader)} images...")
        print(f"Confidence threshold: {conf_threshold}")
        
        # Debug counters
        total_gt_boxes = 0
        total_pred_boxes_before_filter = 0
        total_pred_boxes_after_filter = 0
        images_with_predictions = 0
        
        with torch.no_grad():
            for images, targets in tqdm(test_loader, desc="Evaluating"):
                image = images[0].to(device)
                target = targets[0]
                
                # Count ground truth boxes
                total_gt_boxes += len(target['boxes'])
                
                # Get raw predictions first (before confidence filtering)
                raw_preds = model.model([image])
                raw_pred = raw_preds[0]
                total_pred_boxes_before_filter += len(raw_pred['boxes'])
                
                # Apply confidence filtering using model's predict method
                if hasattr(model, 'predict'):
                    preds = model.predict([image], conf_threshold=conf_threshold)
                    pred = preds[0]
                else:
                    # Manual filtering if predict method not available
                    keep = raw_pred['scores'] >= conf_threshold
                    pred = {
                        'boxes': raw_pred['boxes'][keep],
                        'labels': raw_pred['labels'][keep],
                        'scores': raw_pred['scores'][keep]
                    }
                
                # Count filtered predictions
                num_preds = len(pred['boxes'])
                total_pred_boxes_after_filter += num_preds
                if num_preds > 0:
                    images_with_predictions += 1
                
                # Store predictions and ground truths
                all_predictions.append({
                    'boxes': pred['boxes'].cpu().numpy(),
                    'scores': pred['scores'].cpu().numpy(),
                    'labels': pred['labels'].cpu().numpy()
                })
                
                all_ground_truths.append({
                    'boxes': target['boxes'].cpu().numpy(),
                    'labels': target['labels'].cpu().numpy()
                })
        
        # Print diagnostic statistics
        avg_raw_preds = total_pred_boxes_before_filter / max(len(test_loader), 1)
        avg_filtered_preds = total_pred_boxes_after_filter / max(len(test_loader), 1)
        filter_rate = 1 - (total_pred_boxes_after_filter / max(total_pred_boxes_before_filter, 1))
        
        print(f"\n{'='*60}")
        print(f"Evaluation Statistics:")
        print(f"{'='*60}")
        print(f"Total ground truth boxes: {total_gt_boxes}")
        print(f"Raw predictions (before filtering): {total_pred_boxes_before_filter} (avg {avg_raw_preds:.1f}/img)")
        print(f"Filtered predictions (conf≥{conf_threshold}): {total_pred_boxes_after_filter} (avg {avg_filtered_preds:.1f}/img)")
        print(f"Images with predictions: {images_with_predictions}/{len(test_loader)} ({100*images_with_predictions/len(test_loader):.1f}%)")
        print(f"Filter rate: {100*filter_rate:.1f}% of predictions filtered out")
        
        if total_pred_boxes_after_filter == 0:
            print(f"\n⚠️  WARNING: No predictions passed the confidence threshold!")
            print(f"   This will result in mAP@0.5 = 0")
            print(f"   Suggestions:")
            print(f"   1. Model may need more training epochs")
            print(f"   2. Try lowering conf_threshold to 0.1 or 0.01 for debugging")
            print(f"   3. Check if model weights were loaded correctly")
        
        print(f"{'='*60}\n")
        
        # Calculate metrics
        metrics = self._calculate_detection_metrics(all_predictions, all_ground_truths)
        
        return metrics
    
    def _calculate_detection_metrics(self, predictions: List[Dict], 
                                    ground_truths: List[Dict]):
        """
        Calculate comprehensive detection metrics.
        
        Args:
            predictions: List of prediction dicts with 'boxes', 'scores', 'labels'
            ground_truths: List of ground truth dicts with 'boxes', 'labels'
            
        Returns:
            Dictionary of metrics
        """
        # Get unique classes
        all_labels = set()
        for gt in ground_truths:
            all_labels.update(gt['labels'].tolist())
        for pred in predictions:
            all_labels.update(pred['labels'].tolist())
        
        num_classes = len(all_labels)
        class_ids = sorted(list(all_labels))
        
        # Calculate mAP at different IoU thresholds
        ap_at_iou = {}
        for iou_thresh in self.iou_thresholds:
            aps = []
            for class_id in class_ids:
                ap = self._calculate_ap_for_class(
                    predictions, ground_truths, class_id, iou_thresh
                )
                aps.append(ap)
            
            ap_at_iou[iou_thresh] = np.mean(aps) if aps else 0.0
        
        # mAP@0.5
        mAP50 = ap_at_iou.get(0.5, 0.0)
        
        # mAP@0.5:0.95 (average across all thresholds)
        mAP50_95 = np.mean(list(ap_at_iou.values()))
        
        # Calculate overall precision and recall at IoU=0.5
        precision, recall, f1 = self._calculate_precision_recall_f1(
            predictions, ground_truths, iou_threshold=0.5
        )
        
        # Per-class metrics
        per_class_metrics = {}
        for class_id in class_ids:
            class_ap = self._calculate_ap_for_class(
                predictions, ground_truths, class_id, 0.5
            )
            class_prec, class_rec, class_f1 = self._calculate_precision_recall_f1(
                predictions, ground_truths, iou_threshold=0.5, class_id=class_id
            )
            
            per_class_metrics[str(class_id)] = {
                'AP50': class_ap,
                'precision': class_prec,
                'recall': class_rec,
                'f1_score': class_f1
            }
        
        # IoU statistics
        iou_stats = self._calculate_iou_statistics(predictions, ground_truths)
        
        metrics = {
            'mAP50': float(mAP50),
            'mAP50_95': float(mAP50_95),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'per_class_metrics': per_class_metrics,
            'ap_at_iou': {str(k): float(v) for k, v in ap_at_iou.items()},
            'iou_statistics': iou_stats
        }
        
        return metrics
    
    def _calculate_ap_for_class(self, predictions: List[Dict], 
                               ground_truths: List[Dict],
                               class_id: int, iou_threshold: float = 0.5):
        """
        Calculate Average Precision for a specific class.
        
        Uses the standard 11-point interpolation method.
        """
        # Collect all predictions and ground truths for this class
        class_predictions = []
        class_ground_truths = []
        
        for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            # Ground truths
            gt_mask = gt['labels'] == class_id
            class_ground_truths.extend([
                {'box': box, 'img_idx': img_idx, 'matched': False}
                for box in gt['boxes'][gt_mask]
            ])
            
            # Predictions (sorted by confidence)
            pred_mask = pred['labels'] == class_id
            pred_boxes = pred['boxes'][pred_mask]
            pred_scores = pred['scores'][pred_mask]
            
            for box, score in zip(pred_boxes, pred_scores):
                class_predictions.append({
                    'box': box,
                    'score': score,
                    'img_idx': img_idx,
                    'matched': False
                })
        
        if len(class_predictions) == 0:
            return 0.0
        
        # Sort predictions by confidence (descending)
        class_predictions.sort(key=lambda x: x['score'], reverse=True)
        
        # Match predictions to ground truths
        tp = np.zeros(len(class_predictions))
        fp = np.zeros(len(class_predictions))
        
        for pred_idx, pred in enumerate(class_predictions):
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt in enumerate(class_ground_truths):
                if gt['matched']:
                    continue
                
                if pred['img_idx'] != gt['img_idx']:
                    continue
                
                iou = self._calculate_iou(pred['box'], gt['box'])
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold and best_gt_idx >= 0:
                tp[pred_idx] = 1
                class_ground_truths[best_gt_idx]['matched'] = True
            else:
                fp[pred_idx] = 1
        
        # Calculate cumulative precision and recall
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)
        
        precisions = cum_tp / (cum_tp + cum_fp + 1e-8)
        recalls = cum_tp / (len(class_ground_truths) + 1e-8)
        
        # Calculate AP using 11-point interpolation
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            precisions_at_recall = precisions[recalls >= t]
            if len(precisions_at_recall) > 0:
                ap += np.max(precisions_at_recall) / 11
        
        return ap
    
    def _calculate_precision_recall_f1(self, predictions: List[Dict],
                                      ground_truths: List[Dict],
                                      iou_threshold: float = 0.5,
                                      class_id: int = None):
        """
        Calculate overall precision, recall, and F1 score.
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        for pred, gt in zip(predictions, ground_truths):
            # Filter by class if specified
            if class_id is not None:
                pred_mask = pred['labels'] == class_id
                gt_mask = gt['labels'] == class_id
                pred_boxes = pred['boxes'][pred_mask]
                gt_boxes = gt['boxes'][gt_mask]
            else:
                pred_boxes = pred['boxes']
                gt_boxes = gt['boxes']
            
            # Match predictions to ground truths
            matched_gt = set()
            
            for pred_box in pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_idx in matched_gt:
                        continue
                    
                    iou = self._calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    total_tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    total_fp += 1
            
            total_fn += len(gt_boxes) - len(matched_gt)
        
        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return precision, recall, f1
    
    def _calculate_iou_statistics(self, predictions: List[Dict],
                                 ground_truths: List[Dict]):
        """
        Calculate IoU distribution statistics.
        """
        all_ious = []
        
        for pred, gt in zip(predictions, ground_truths):
            for pred_box in pred['boxes']:
                best_iou = 0
                
                for gt_box in gt['boxes']:
                    iou = self._calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                
                if best_iou > 0:
                    all_ious.append(best_iou)
        
        if len(all_ious) == 0:
            return {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0
            }
        
        ious_array = np.array(all_ious)
        
        return {
            'mean': float(np.mean(ious_array)),
            'median': float(np.median(ious_array)),
            'std': float(np.std(ious_array)),
            'min': float(np.min(ious_array)),
            'max': float(np.max(ious_array)),
            'histogram': np.histogram(ious_array, bins=20, range=(0, 1))[0].tolist()
        }
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """
        Calculate Intersection over Union between two boxes.
        
        Args:
            box1: [x1, y1, x2, y2] or [x, y, w, h]
            box2: [x1, y1, x2, y2] or [x, y, w, h]
            
        Returns:
            IoU value between 0 and 1
        """
        # Convert to [x1, y1, x2, y2] format if needed
        if len(box1) == 4:
            if box1[2] < box1[0] or box1[3] < box1[1]:
                # Assume [x, y, w, h] format
                box1 = np.array([box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]])
            if box2[2] < box2[0] or box2[3] < box2[1]:
                box2 = np.array([box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]])
        
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate union
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _generate_visualizations(self, figures_dir: Path):
        """
        Generate comprehensive visualization plots.
        """
        print("\nGenerating visualizations...")
        
        # 1. IoU Distribution
        if 'iou_statistics' in self.metrics:
            self._plot_iou_distribution(figures_dir)
        
        # 2. mAP at different IoU thresholds
        if 'ap_at_iou' in self.metrics:
            self._plot_map_vs_iou(figures_dir)
        
        # 3. Per-class metrics
        if 'per_class_metrics' in self.metrics:
            self._plot_per_class_metrics(figures_dir)
        
        print(f"Visualizations saved to: {figures_dir}")
    
    def _plot_iou_distribution(self, figures_dir: Path):
        """Plot IoU distribution histogram."""
        iou_stats = self.metrics['iou_statistics']
        
        plt.figure(figsize=(10, 6))
        
        if 'histogram' in iou_stats:
            hist = np.array(iou_stats['histogram'])
            bins = np.linspace(0, 1, 21)
            plt.bar(bins[:-1], hist, width=0.05, alpha=0.7, color='steelblue', edgecolor='black')
        
        plt.axvline(x=iou_stats['mean'], color='red', linestyle='--', linewidth=2, 
                   label=f"Mean: {iou_stats['mean']:.3f}")
        plt.axvline(x=iou_stats['median'], color='green', linestyle='--', linewidth=2,
                   label=f"Median: {iou_stats['median']:.3f}")
        
        plt.xlabel('IoU', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('IoU Distribution of Matched Detections', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(figures_dir / 'IoU_distribution.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved IoU_distribution.png")
    
    def _plot_map_vs_iou(self, figures_dir: Path):
        """Plot mAP at different IoU thresholds."""
        ap_at_iou = self.metrics['ap_at_iou']
        
        iou_values = [float(k) for k in ap_at_iou.keys()]
        ap_values = list(ap_at_iou.values())
        
        plt.figure(figsize=(10, 6))
        plt.plot(iou_values, ap_values, 'o-', linewidth=2, markersize=8, 
                color='steelblue', markerfacecolor='white', markeredgewidth=2)
        
        plt.xlabel('IoU Threshold', fontsize=12)
        plt.ylabel('Average Precision', fontsize=12)
        plt.title('mAP vs IoU Threshold', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(iou_values, rotation=45)
        plt.tight_layout()
        
        plt.savefig(figures_dir / 'mAP_vs_IoU.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved mAP_vs_IoU.png")
    
    def _plot_per_class_metrics(self, figures_dir: Path):
        """Plot per-class precision, recall, and F1."""
        per_class = self.metrics['per_class_metrics']
        
        if not per_class:
            return
        
        class_names = list(per_class.keys())
        precisions = [per_class[c]['precision'] for c in class_names]
        recalls = [per_class[c]['recall'] for c in class_names]
        f1_scores = [per_class[c]['f1_score'] for c in class_names]
        
        x = np.arange(len(class_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width, precisions, width, label='Precision', 
                      color='steelblue', alpha=0.8)
        bars2 = ax.bar(x, recalls, width, label='Recall', 
                      color='coral', alpha=0.8)
        bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', 
                      color='mediumseagreen', alpha=0.8)
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Detection Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(figures_dir / 'per_class_metrics.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved per_class_metrics.png")
    
    def generate_report(self, output_dir: str):
        """
        Generate comprehensive evaluation report in markdown format.
        
        Args:
            output_dir: Directory containing experiment outputs
        """
        report_path = Path(output_dir) / "logs" / "experiment_report.md"
        
        # Build per-class table
        per_class_table = ""
        if 'per_class_metrics' in self.metrics:
            per_class_table = "\n### Per-Class Metrics\n\n"
            per_class_table += "| Class | AP@0.5 | Precision | Recall | F1-Score |\n"
            per_class_table += "|-------|--------|-----------|--------|----------|\n"
            
            for class_name, class_metrics in self.metrics['per_class_metrics'].items():
                per_class_table += f"| {class_name} | {class_metrics.get('AP50', 0):.4f} | "
                per_class_table += f"{class_metrics.get('precision', 0):.4f} | "
                per_class_table += f"{class_metrics.get('recall', 0):.4f} | "
                per_class_table += f"{class_metrics.get('f1_score', 0):.4f} |\n"
        
        # Build IoU statistics section
        iou_section = ""
        if 'iou_statistics' in self.metrics:
            iou_stats = self.metrics['iou_statistics']
            iou_section = f"""
### IoU Statistics

| Statistic | Value |
|-----------|-------|
| Mean IoU | {iou_stats['mean']:.4f} |
| Median IoU | {iou_stats['median']:.4f} |
| Std Dev | {iou_stats['std']:.4f} |
| Min IoU | {iou_stats['min']:.4f} |
| Max IoU | {iou_stats['max']:.4f} |
"""
        
        report = f"""# Experiment Report: Detection Model

## Overall Metrics

| Metric | Value |
|--------|-------|
| **mAP@0.5** | {self.metrics.get('mAP50', 'N/A'):.4f} |
| **mAP@0.5:0.95** | {self.metrics.get('mAP50_95', 'N/A'):.4f} |
| **Precision** | {self.metrics.get('precision', 'N/A'):.4f} |
| **Recall** | {self.metrics.get('recall', 'N/A'):.4f} |
| **F1-Score** | {self.metrics.get('f1_score', 'N/A'):.4f} |

{per_class_table}
{iou_section}

## Visualizations

See the `figures/` directory for:
- `IoU_distribution.png` - Distribution of IoU values for matched detections
- `mAP_vs_IoU.png` - mAP at different IoU thresholds
- `per_class_metrics.png` - Per-class precision, recall, and F1 scores

## Interpretation

- **mAP@0.5**: Primary metric, measures detection accuracy at IoU=0.5
- **mAP@0.5:0.95**: More stringent, averages across multiple IoU thresholds
- **Higher mAP** indicates better detection quality
- **Balanced Precision/Recall** suggests good generalization
"""
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(f"Report saved to: {report_path}")
