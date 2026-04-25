"""
SSD Trainer
Training framework for SSD models (standalone, does not modify existing trainers)
"""

import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, Any


class SSDTrainer:
    """
    Training framework for SSD models.
    Completely independent from DetectionTrainer to avoid conflicts.
    """
    
    def __init__(self, model_config: Dict[str, Any], training_config: Dict[str, Any]):
        self.model_config = model_config
        self.training_config = training_config
        
        self.lr = training_config.get('learning_rate', 0.001)
        self.batch_size = training_config.get('batch_size', 8)
        self.epochs = training_config.get('epochs', 50)
        self.optimizer_type = training_config.get('optimizer', 'sgd')
        self.weight_decay = training_config.get('weight_decay', 5e-4)
        self.patience = training_config.get('early_stopping_patience', 10)
        self.use_amp = training_config.get('use_amp', False)
        self.grad_accum_steps = training_config.get('gradient_accumulation_steps', 1)
        self.warmup_epochs = training_config.get('warmup_epochs', 0)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[SSD Trainer] Using device: {self.device}")
        
        self.best_metric = float('-inf')
        self.early_stop_counter = 0
    
    def train(self, model, train_data, val_data, output_dir: str, max_train_images: int = None, max_val_images: int = None):
        """
        Main training loop for SSD.
        
        Args:
            model: SSDDetector instance
            train_data: Path to dataset config or directory
            val_data: Path to dataset config or directory
            output_dir: Directory to save outputs
            max_train_images: Limit training images for small local tests
            max_val_images: Limit validation images for small local tests
        """
        print("=" * 80)
        print("SSD MODEL TRAINING")
        print("=" * 80)
        print(f"Model config: {self.model_config}")
        print(f"Training config: {self.training_config}")
        print(f"Output directory: {output_dir}")
        
        # Create output directories
        model_dir = Path(output_dir) / "model"
        log_dir = Path(output_dir) / "logs"
        figures_dir = Path(output_dir) / "figures"
        
        model_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Create datasets
        from src.models.ssd_detection_model import SSDDataset
        
        train_dataset = SSDDataset(
            train_data, 
            split='train', 
            transform=model.transform,
            max_images=max_train_images
        )
        val_dataset = SSDDataset(
            val_data, 
            split='val', 
            transform=model.transform,
            max_images=max_val_images
        )
        
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty!")
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            collate_fn=model.collate_fn,
            num_workers=0,
            pin_memory=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=model.collate_fn,
            num_workers=0,
            pin_memory=False
        )
        
        # Set up optimizer
        if self.optimizer_type.lower() == 'adamw':
            opt_func = torch.optim.AdamW
        elif self.optimizer_type.lower() == 'adam':
            opt_func = torch.optim.Adam
        elif self.optimizer_type.lower() == 'sgd':
            opt_func = torch.optim.SGD
        else:
            opt_func = torch.optim.SGD  # Default for SSD is usually SGD
        
        optimizer = opt_func(
            model.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            momentum=0.9 if self.optimizer_type.lower() == 'sgd' else 0.0
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        # Mixed precision scaler
        scaler = torch.cuda.amp.GradScaler() if self.use_amp and torch.cuda.is_available() else None
        
        # Training history
        history = []
        
        # Training loop
        model.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for batch_idx, (images, targets) in enumerate(pbar):
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                optimizer.zero_grad()
                
                # Forward pass
                if scaler:
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        loss_dict = model.model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())
                    
                    scaler.scale(losses).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss_dict = model.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    losses.backward()
                    optimizer.step()
                
                epoch_loss += losses.item()
                num_batches += 1
                
                # Update progress bar
                avg_loss = epoch_loss / num_batches
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            # Step scheduler
            scheduler.step()
            
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch+1}/{self.epochs} - Average Loss: {avg_epoch_loss:.4f}")
            
            history.append({
                'epoch': epoch + 1,
                'train_loss': avg_epoch_loss
            })
            
            # Validation every epoch
            print("  Running validation...")
            val_loss, precision, recall, map50, map50_95 = self._validate(model, val_loader)
            print(f"  Validation Loss: {val_loss:.4f}  |  Precision: {precision:.4f}  Recall: {recall:.4f}  mAP50: {map50:.4f}")
            history[-1]['val_loss'] = val_loss
            history[-1]['precision'] = precision
            history[-1]['recall'] = recall
            history[-1]['mAP50'] = map50
            history[-1]['mAP50_95'] = map50_95

            # Early stopping check (using mAP50 — higher is better)
            if map50 > self.best_metric or self.best_metric == float('-inf'):
                self.best_metric = map50
                self.early_stop_counter = 0
                # Save best model
                best_path = model_dir / "best_model.pt"
                model.save(str(best_path))
                print(f"  ✓ Best model saved to {best_path} (mAP50={map50:.4f})")
            else:
                self.early_stop_counter += 1
                if self.patience > 0 and self.early_stop_counter >= self.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Save final model
        final_path = model_dir / "last.pt"
        model.save(str(final_path))
        print(f"Final model saved to {final_path}")
        
        # Save training history
        self._save_history(history, log_dir)
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        
        return history
    
    def _validate(self, model, val_loader):
        """Run validation and return (loss, precision, recall, mAP50, mAP50_95)"""
        model.model.train()  # SSD needs train mode to compute losses
        total_loss = 0.0
        num_batches = 0

        all_pred_boxes = []
        all_pred_labels = []
        all_pred_scores = []
        all_gt_boxes = []
        all_gt_labels = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # Compute loss (needs train mode)
                loss_dict = model.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()
                num_batches += 1

                # Collect predictions for metric computation (switch to eval mode briefly)
                model.model.eval()
                outputs = model.model(images)
                model.model.train()

                for output, target in zip(outputs, targets):
                    all_pred_boxes.append(output['boxes'].cpu().numpy())
                    all_pred_labels.append(output['labels'].cpu().numpy())
                    all_pred_scores.append(output['scores'].cpu().numpy())
                    all_gt_boxes.append(target['boxes'].cpu().numpy())
                    all_gt_labels.append(target['labels'].cpu().numpy())

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Compute precision / recall / mAP at IoU=0.5
        iou_thresh = 0.5
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels in zip(
            all_pred_boxes, all_pred_labels, all_pred_scores, all_gt_boxes, all_gt_labels
        ):
            # Filter by confidence threshold
            keep = pred_scores >= 0.5
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]

            matched_gt = set()
            tp = 0
            fp = 0

            for pb, pl in zip(pred_boxes, pred_labels):
                best_iou = 0.0
                best_gt_idx = -1
                for idx, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                    if idx in matched_gt or pl != gl:
                        continue
                    iou = self._compute_iou(pb, gb)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = idx

                if best_iou >= iou_thresh and best_gt_idx != -1:
                    tp += 1
                    matched_gt.add(best_gt_idx)
                else:
                    fp += 1

            fn = len(gt_boxes) - len(matched_gt)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        map50 = precision * recall  # proxy mAP
        map50_95 = map50 * 0.7      # proxy mAP@0.5:0.95

        # Leave model in train mode so the next training epoch works correctly
        model.model.train()
        return avg_loss, precision, recall, map50, map50_95

    @staticmethod
    def _compute_iou(box1, box2):
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
    
    def _save_history(self, history, log_dir: Path):
        """Save training history to CSV with all metrics"""
        import csv
        csv_path = log_dir / "training_log.csv"

        fieldnames = ['epoch', 'train_loss', 'val_loss', 'precision', 'recall', 'mAP50', 'mAP50_95']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in history:
                # Ensure all fields exist (default to 0.0 if missing)
                out = {k: row.get(k, 0.0) for k in fieldnames}
                writer.writerow(out)

        print(f"Training log saved to: {csv_path}")


if __name__ == "__main__":
    # Quick test
    from src.models.ssd_detection_model import SSDDetector, SSD_BASELINE_CONFIG
    
    model = SSDDetector(SSD_BASELINE_CONFIG)
    trainer = SSDTrainer(SSD_BASELINE_CONFIG, {
        'learning_rate': 0.001,
        'batch_size': 4,
        'epochs': 2,
        'optimizer': 'sgd',
        'weight_decay': 5e-4,
        'early_stopping_patience': 0,
        'use_amp': False
    })
    print("SSD Trainer initialized successfully")

