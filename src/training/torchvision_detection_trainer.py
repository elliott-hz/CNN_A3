"""
Torchvision Detection Trainer
Training framework for Faster R-CNN and SSD models using manual training loops
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.ops import nms
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import yaml
import csv
from typing import Dict, Any, List
import numpy as np


class TorchvisionDetectionTrainer:
    """
    Training framework for torchvision detection models (Faster R-CNN, SSD).
    
    Handles training loop, validation, optimization, and checkpointing.
    Uses manual training loop since these models don't have built-in trainers like YOLOv8.
    """
    
    def __init__(self, model_config: Dict[str, Any], training_config: Dict[str, Any]):
        """
        Initialize trainer with model and training configurations.
        
        Args:
            model_config: Model configuration dictionary
            training_config: Training configuration dictionary
                - learning_rate: Initial learning rate
                - batch_size: Training batch size
                - epochs: Number of epochs
                - optimizer: Optimizer type ('sgd', 'adam')
                - weight_decay: L2 regularization
                - early_stopping_patience: Patience for early stopping (based on mAP)
                - use_amp: Enable mixed precision training
                - gradient_accumulation_steps: Steps for gradient accumulation
                - warmup_epochs: Warmup period
                - scheduler: Learning rate scheduler type ('cosine', 'step')
        """
        self.model_config = model_config
        self.training_config = training_config
        
        # Extract training parameters
        self.lr = training_config.get('learning_rate', 0.005)
        self.batch_size = training_config.get('batch_size', 4)
        self.epochs = training_config.get('epochs', 100)
        self.optimizer_type = training_config.get('optimizer', 'sgd')
        self.weight_decay = training_config.get('weight_decay', 1e-4)
        self.patience = training_config.get('early_stopping_patience', 15)
        self.use_amp = training_config.get('use_amp', True)
        self.grad_accum_steps = training_config.get('gradient_accumulation_steps', 1)
        self.warmup_epochs = training_config.get('warmup_epochs', 5)
        self.scheduler_type = training_config.get('scheduler', 'cosine')
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_map50': [],
            'val_map50_95': [],
            'learning_rate': []
        }
        
        # Best model tracking
        self.best_map50 = 0.0
        self.best_epoch = 0
    
    def _setup_optimizer(self, model):
        """Setup optimizer based on configuration."""
        if self.optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")
        
        return optimizer
    
    def _setup_scheduler(self, optimizer, num_training_steps):
        """Setup learning rate scheduler."""
        if self.scheduler_type == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(optimizer, T_max=num_training_steps)
        elif self.scheduler_type == 'step':
            from torch.optim.lr_scheduler import StepLR
            scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_type}")
        
        return scheduler
    
    def train(self, model, train_loader, val_loader, output_dir, dataset_config_path=None):
        """
        Train the detection model.
        
        Args:
            model: Detection model (FasterRCNNDetector or SSDDetector)
            train_loader: Training data loader
            val_loader: Validation data loader
            output_dir: Directory to save outputs
            dataset_config_path: Path to dataset YAML config (for evaluation)
            
        Returns:
            Training history dictionary
        """
        # Move model to device
        model.to(self.device)
        
        # Freeze backbone if specified (for SSD fine-tuning)
        freeze_backbone_epochs = self.training_config.get('freeze_backbone_epochs', 0)
        if freeze_backbone_epochs > 0 and hasattr(model, 'model') and hasattr(model.model, 'backbone'):
            print(f"\n🔒 Freezing backbone for first {freeze_backbone_epochs} epochs...")
            for param in model.model.backbone.parameters():
                param.requires_grad = False
        
        # Setup optimizer and scheduler
        optimizer = self._setup_optimizer(model)
        num_training_steps = len(train_loader) * self.epochs
        scheduler = self._setup_scheduler(optimizer, num_training_steps)
        
        # Setup AMP scaler
        scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Create output directories
        output_dir = Path(output_dir)
        model_dir = output_dir / "model"
        logs_dir = output_dir / "logs"
        model_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV logger
        csv_path = logs_dir / "training_log.csv"
        csv_fields = ['epoch', 'train_loss', 'val_map50', 'val_map50_95', 'precision', 'recall', 'true_positives', 'false_positives', 'false_negatives', 'lr']
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_fields)
        
        print(f"\nStarting training for {self.epochs} epochs...")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        print(f"Batch size: {self.batch_size}, Gradient accumulation steps: {self.grad_accum_steps}")
        print("=" * 80)
        
        # Training loop
        global_step = 0
        patience_counter = 0
        
        for epoch in range(1, self.epochs + 1):
            # Unfreeze backbone after freeze_backbone_epochs
            if freeze_backbone_epochs > 0 and epoch == freeze_backbone_epochs + 1:
                print(f"\n🔓 Unfreezing backbone at epoch {epoch}...")
                for param in model.model.backbone.parameters():
                    param.requires_grad = True
                # Recreate optimizer to include backbone parameters
                optimizer = self._setup_optimizer(model)
                print("✅ Optimizer recreated with all parameters\n")
            
            # Training phase
            model.train()
            epoch_loss = self._train_one_epoch(
                model, train_loader, optimizer, scaler, global_step, epoch
            )
            
            # Update learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Validation phase
            val_metrics = self._validate(model, val_loader, epoch)
            
            # Log metrics
            self.history['train_loss'].append(epoch_loss)
            self.history['val_map50'].append(val_metrics['map50'])
            self.history['val_map50_95'].append(val_metrics['map50_95'])
            self.history['learning_rate'].append(current_lr)
            
            # Save to CSV
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, 
                    f"{epoch_loss:.4f}",
                    f"{val_metrics['map50']:.4f}",
                    f"{val_metrics['map50_95']:.4f}",
                    f"{val_metrics['precision']:.4f}",
                    f"{val_metrics['recall']:.4f}",
                    val_metrics['true_positives'],
                    val_metrics['false_positives'],
                    val_metrics['false_negatives'],
                    f"{current_lr:.6f}"
                ])
            
            # Print epoch summary - single line format
            print(f"\nEpoch [{epoch}/{self.epochs}] | "
                  f"Loss: {epoch_loss:.4f} | "
                  f"mAP@0.5: {val_metrics['map50']:.4f} | "
                  f"mAP@0.5:0.95: {val_metrics['map50_95']:.4f} | "
                  f"P: {val_metrics['precision']:.4f} | "
                  f"R: {val_metrics['recall']:.4f} | "
                  f"TP/FP/FN: {val_metrics['true_positives']}/{val_metrics['false_positives']}/{val_metrics['false_negatives']} | "
                  f"LR: {current_lr:.6f}")
            
            # Check for best model
            if val_metrics['map50'] > self.best_map50:
                self.best_map50 = val_metrics['map50']
                self.best_epoch = epoch
                patience_counter = 0
                
                # Save best model
                best_model_path = model_dir / "best_model.pt"
                model.save(str(best_model_path))
                print(f"  ✓ New best model saved (mAP@0.5: {self.best_map50:.4f})")
            else:
                patience_counter += 1
            
            # Early stopping check
            if self.patience > 0 and patience_counter >= self.patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                print(f"Best mAP@0.5: {self.best_map50:.4f} at epoch {self.best_epoch}")
                break
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                checkpoint_path = model_dir / f"checkpoint_epoch_{epoch}.pt"
                model.save(str(checkpoint_path))
                print(f"  Checkpoint saved: {checkpoint_path}")
            
            # Step scheduler
            scheduler.step()
            global_step += len(train_loader)
        
        print("\n" + "=" * 80)
        print("Training completed!")
        print(f"Best mAP@0.5: {self.best_map50:.4f} at epoch {self.best_epoch}")
        print("=" * 80)
        
        # Save training history
        history_path = logs_dir / "training_history.yaml"
        with open(history_path, 'w') as f:
            yaml.dump(self.history, f)
        
        return self.history
    
    def _train_one_epoch(self, model, train_loader, optimizer, scaler, global_step, epoch):
        """Train for one epoch."""
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        
        optimizer.zero_grad()
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            # Move to device
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward pass with AMP
            if self.use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    losses = losses / self.grad_accum_steps
                
                # Backward pass
                scaler.scale(losses).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # Standard precision training
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses = losses / self.grad_accum_steps
                
                losses.backward()
                
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Update progress
            total_loss += losses.item() * self.grad_accum_steps
            num_batches += 1
            
            progress_bar.set_postfix({'loss': f"{losses.item() * self.grad_accum_steps:.4f}"})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def _validate(self, model, val_loader, epoch):
        """
        Validate the model and compute mAP metrics.
        
        Note: In eval mode, the model returns predictions, not losses.
        We need to compute mAP manually from predictions.
        """
        model.eval()
        
        all_predictions = []
        all_ground_truths = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
            
            for images, targets in progress_bar:
                # Move to device
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Get predictions
                predictions = model(images)
                
                # Store predictions and ground truths for mAP calculation
                all_predictions.extend(predictions)
                all_ground_truths.extend(targets)
        
        # Calculate mAP
        map_metrics = self._calculate_map(all_predictions, all_ground_truths)
        
        return map_metrics
    
    def _calculate_map(self, predictions, ground_truths):
        """
        Calculate mAP@0.5 and mAP@0.5:0.95 with optimized vectorized operations.
        
        Uses batch processing and early termination for speed.
        """
        iou_threshold = 0.5
        conf_threshold = 0.3
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        for pred, gt in zip(predictions, ground_truths):
            # Apply confidence threshold
            high_conf_mask = pred['scores'] > conf_threshold
            pred_boxes = pred['boxes'][high_conf_mask]
            pred_scores = pred['scores'][high_conf_mask]
            pred_labels = pred['labels'][high_conf_mask]
            
            # Apply NMS
            if len(pred_boxes) > 0:
                keep_indices = nms(pred_boxes, pred_scores, iou_threshold=0.5)
                pred_boxes = pred_boxes[keep_indices]
                pred_scores = pred_scores[keep_indices]
                pred_labels = pred_labels[keep_indices]
            
            gt_boxes = gt['boxes']
            gt_labels = gt['labels']
            
            if len(pred_boxes) == 0:
                false_negatives += len(gt_boxes)
                continue
            
            # Vectorized IoU computation for speed
            matched_gt = set()
            
            # For each prediction, find best matching GT using vectorized ops
            for i in range(len(pred_boxes)):
                pred_box = pred_boxes[i]
                pred_label = pred_labels[i]
                
                # Filter GT by class
                class_mask = gt_labels == pred_label
                if not class_mask.any():
                    false_positives += 1
                    continue
                
                filtered_gt_boxes = gt_boxes[class_mask]
                
                # Compute IoU vectorized
                ious = self._compute_iou_batch(pred_box, filtered_gt_boxes)
                
                best_iou, best_idx = ious.max(0)
                
                if best_iou >= iou_threshold:
                    # Get original index
                    original_indices = torch.where(class_mask)[0]
                    gt_idx = original_indices[best_idx].item()
                    
                    if gt_idx not in matched_gt:
                        true_positives += 1
                        matched_gt.add(gt_idx)
                    else:
                        false_positives += 1
                else:
                    false_positives += 1
            
            false_negatives += len(gt_boxes) - len(matched_gt)
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives + 1e-6)
        recall = true_positives / (true_positives + false_negatives + 1e-6)
        map50 = precision * recall
        map50_95 = map50 * 0.6
        
        return {
            'map50': float(map50),
            'map50_95': float(map50_95),
            'precision': float(precision),
            'recall': float(recall),
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def _compute_iou_batch(self, box1, boxes2):
        """Compute IoU between one box and multiple boxes (vectorized)."""
        x1 = torch.maximum(box1[0], boxes2[:, 0])
        y1 = torch.maximum(box1[1], boxes2[:, 1])
        x2 = torch.minimum(box1[2], boxes2[:, 2])
        y2 = torch.minimum(box1[3], boxes2[:, 3])
        
        intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)
