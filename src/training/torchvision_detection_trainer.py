"""
Torchvision Detection Trainer
Training framework for Faster R-CNN and SSD models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
from PIL import Image
from pathlib import Path
import yaml
import csv
import json
import numpy as np
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
import os


class DetectionDataset(Dataset):
    """
    Custom dataset for detection models using COCO or VOC format.
    
    Handles image loading, annotation parsing, and data augmentation.
    """
    
    def __init__(self, images_dir: str, annotations_file: str = None, 
                 annotations_dir: str = None, transform=None, is_voc: bool = False,
                 class_names: list = None):
        """
        Initialize dataset.
        
        Args:
            images_dir: Directory containing images
            annotations_file: COCO JSON file path (for COCO format)
            annotations_dir: Directory containing XML files (for VOC format)
            transform: Data augmentation transforms
            is_voc: Whether using VOC format
            class_names: List of class names for VOC format (optional, defaults to ['dog'])
        """
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.is_voc = is_voc
        
        # Set up class name to ID mapping for VOC format
        if is_voc:
            if class_names is None:
                # Default to single class 'dog' for this project
                self.class_names = ['dog']
            else:
                self.class_names = class_names
            self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Load annotations
        if is_voc:
            self.annotations_dir = Path(annotations_dir)
            self.image_files = list(self.images_dir.glob('*.jpg')) + list(self.images_dir.glob('*.png'))
            self.annotations = self._load_voc_annotations()
        else:
            # COCO format
            with open(annotations_file, 'r') as f:
                self.coco_data = json.load(f)
            
            # Build image to annotations mapping
            self.image_to_anns = {}
            for ann in self.coco_data['annotations']:
                img_id = ann['image_id']
                if img_id not in self.image_to_anns:
                    self.image_to_anns[img_id] = []
                self.image_to_anns[img_id].append(ann)
            
            # Get image info
            self.images_info = {img['id']: img for img in self.coco_data['images']}
            self.image_ids = list(self.images_info.keys())
    
    def _load_voc_annotations(self):
        """Load VOC XML annotations."""
        import xml.etree.ElementTree as ET
        
        annotations = {}
        for img_path in self.image_files:
            xml_path = self.annotations_dir / f"{img_path.stem}.xml"
            if xml_path.exists():
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                boxes = []
                labels = []
                
                for obj in root.findall('object'):
                    name = obj.find('name').text
                    bndbox = obj.find('bndbox')
                    
                    xmin = int(bndbox.find('xmin').text)
                    ymin = int(bndbox.find('ymin').text)
                    xmax = int(bndbox.find('xmax').text)
                    ymax = int(bndbox.find('ymax').text)
                    
                    boxes.append([xmin, ymin, xmax, ymax])
                    
                    # Map class name to ID
                    if name in self.class_to_id:
                        labels.append(self.class_to_id[name])
                    else:
                        # Unknown class, skip this object or assign to background
                        print(f"Warning: Unknown class '{name}' in {img_path.stem}, skipping")
                        continue
                
                if boxes:  # Only add annotation if there are valid boxes
                    annotations[str(img_path)] = {
                        'boxes': torch.as_tensor(boxes, dtype=torch.float32),
                        'labels': torch.as_tensor(labels, dtype=torch.int64)
                    }
        
        return annotations
    
    def __len__(self):
        if self.is_voc:
            return len(self.image_files)
        else:
            return len(self.image_ids)
    
    def __getitem__(self, idx):
        if self.is_voc:
            img_path = self.image_files[idx]
            image = Image.open(str(img_path)).convert("RGB")
            image = F.to_tensor(image)
            target = self.annotations[str(img_path)]
        else:
            img_id = self.image_ids[idx]
            img_info = self.images_info[img_id]
            
            img_path = self.images_dir / img_info['file_name']
            image = Image.open(str(img_path)).convert("RGB")
            image = F.to_tensor(image)
            
            # Build target with validation
            anns = self.image_to_anns.get(img_id, [])
            boxes = []
            labels = []
            areas = []
            
            for ann in anns:
                bbox = ann['bbox']  # [x, y, width, height]
                x1, y1, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                x2, y2 = x1 + w, y1 + h
                
                # Filter out invalid boxes (must have positive width and height)
                if w > 0 and h > 0 and x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2])
                    labels.append(ann['category_id'])
                    areas.append(ann.get('area', w * h))
            
            target = {
                'boxes': torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
                'labels': torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
                'image_id': torch.tensor([img_id]),
                'area': torch.as_tensor(areas, dtype=torch.float32) if areas else torch.zeros((0,)),
                'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
            }
        
        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
        
        return image, target


class TorchvisionDetectionTrainer:
    """
    Training framework for Faster R-CNN and SSD models.
    
    Implements manual training loop since torchvision models don't have
    built-in training methods like Ultralytics YOLO.
    """
    
    def __init__(self, model_config: Dict[str, Any], training_config: Dict[str, Any]):
        """
        Initialize trainer.
        
        Args:
            model_config: Model configuration
            training_config: Training configuration
        """
        self.model_config = model_config
        self.training_config = training_config
        
        # Extract training parameters
        self.lr = training_config.get('learning_rate', 0.005)
        self.batch_size = training_config.get('batch_size', 4)
        self.epochs = training_config.get('epochs', 150)
        self.optimizer_type = training_config.get('optimizer', 'sgd')
        self.weight_decay = training_config.get('weight_decay', 5e-4)
        self.patience = training_config.get('early_stopping_patience', 15)
        self.use_amp = training_config.get('use_amp', True)
        self.grad_accum_steps = training_config.get('gradient_accumulation_steps', 1)
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Training state
        self.best_metric = float('-inf')
        self.early_stop_counter = 0
        self.training_history = []
    
    def train(self, model, train_dataset, val_dataset, output_dir: str):
        """
        Main training loop.
        
        Args:
            model: FasterRCNNDetector or SSDDetector
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Directory to save outputs
        """
        print("=" * 80)
        print("TORCHVISION DETECTION MODEL TRAINING")
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
        
        # Save model config
        with open(model_dir / "model_config.json", 'w') as f:
            yaml.dump(self.model_config, f)
        
        # Move model to device
        model = model.to(self.device)
        
        # Setup optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        if self.optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=self.weight_decay)
        elif self.optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_type}")
        
        # Setup learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        # Setup mixed precision
        scaler = torch.cuda.amp.GradScaler() if self.use_amp and self.device.type == 'cuda' else None
        
        # Create data loaders
        # Note: For detection, we need custom collate function
        def collate_fn(batch):
            return tuple(zip(*batch))
        
        # Use num_workers=0 to avoid potential deadlocks on some systems
        # Can be increased to 2-4 if needed for performance
        num_workers = 2 if torch.cuda.is_available() else 0
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, 
                                  collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                                collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
        
        # Training loop
        print(f"\nStarting training for {self.epochs} epochs...")
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        
        for epoch in range(1, self.epochs + 1):
            # Train one epoch
            train_metrics = self._train_one_epoch(model, train_loader, optimizer, scaler, epoch)
            
            # Validate
            val_metrics = self._validate(model, val_loader)
            
            # Update learning rate
            scheduler.step()
            
            # Log metrics
            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'learning_rate': scheduler.get_last_lr()[0]
            })
            
            # Print progress
            print(f"Epoch [{epoch}/{self.epochs}] "
                  f"Train Loss: {train_metrics['loss']:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if val_metrics['loss'] < self.best_metric or self.best_metric == float('-inf'):
                self.best_metric = val_metrics['loss']
                best_model_path = model_dir / "best_model.pt"
                model.save(str(best_model_path))
                print(f"  ✓ Saved best model (loss: {val_metrics['loss']:.4f})")
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
            
            # Early stopping
            if self.patience > 0 and self.early_stop_counter >= self.patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break
        
        # Save final model
        final_model_path = model_dir / "final_model.pt"
        model.save(str(final_model_path))
        
        # Log training history
        self._log_training_history(log_dir)
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        
        return self.training_history
    
    def _train_one_epoch(self, model, train_loader, optimizer, scaler, epoch):
        """Train for one epoch."""
        model.train()
        total_loss = 0
        num_batches = 0
        
        optimizer.zero_grad()
        
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            # Move to device
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    loss_output = model(images, targets)
                    
                    # Handle different return types from different models
                    if isinstance(loss_output, dict):
                        # Faster R-CNN style: {'loss_classifier': ..., 'loss_box_reg': ...}
                        losses = sum(loss for loss in loss_output.values())
                    elif isinstance(loss_output, (list, tuple)):
                        # SSD style: could be list of dicts or list of tensors
                        if len(loss_output) > 0 and isinstance(loss_output[0], dict):
                            # List of dicts - sum scalar values to avoid dimension mismatch
                            # Use .mean() for multi-element tensors to reduce to scalar
                            losses = sum(
                                sum(v.mean() if v.numel() > 1 else v.item() if hasattr(v, 'item') else v 
                                    for v in d.values()) 
                                for d in loss_output
                            )
                            losses = torch.tensor(losses, device=self.device)
                        else:
                            # List of tensors - direct sum
                            losses = sum(loss_output)
                    else:
                        # Single tensor
                        losses = loss_output
                    
                    losses = losses / self.grad_accum_steps
                
                # Backward pass
                scaler.scale(losses).backward()
                
                # Gradient accumulation
                if (num_batches + 1) % self.grad_accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss_output = model(images, targets)
                
                # Handle different return types from different models
                if isinstance(loss_output, dict):
                    # Faster R-CNN style
                    losses = sum(loss for loss in loss_output.values())
                elif isinstance(loss_output, (list, tuple)):
                    # SSD style
                    if len(loss_output) > 0 and isinstance(loss_output[0], dict):
                        # List of dicts - sum scalar values to avoid dimension mismatch
                        # Use .mean() for multi-element tensors to reduce to scalar
                        losses = sum(
                            sum(v.mean() if v.numel() > 1 else v.item() if hasattr(v, 'item') else v 
                                for v in d.values()) 
                            for d in loss_output
                        )
                        losses = torch.tensor(losses, device=self.device)
                    else:
                        # List of tensors
                        losses = sum(loss_output)
                else:
                    # Single tensor
                    losses = loss_output
                
                losses = losses / self.grad_accum_steps
                
                losses.backward()
                
                if (num_batches + 1) % self.grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            total_loss += losses.item() * self.grad_accum_steps
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}
    
    def _validate(self, model, val_loader, conf_threshold: float = 0.1):
        """Validate model and calculate a proxy metric for early stopping.
        
        Note: Torchvision detection models do NOT return losses in eval mode.
        Instead of returning dummy loss, we calculate a simple validation metric
        based on the number of valid predictions (above confidence threshold) to guide early stopping.
        
        Args:
            model: Detection model
            val_loader: Validation data loader
            conf_threshold: Minimum confidence score to count a prediction (default: 0.1)
        """
        model.eval()
        
        total_predictions = 0
        num_batches = 0
        num_images = 0
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation", leave=False):
                images = [img.to(self.device) for img in images]
                
                # Run forward pass in eval mode (returns predictions)
                predictions = model(images)
                
                # Count predictions above confidence threshold
                batch_predictions = 0
                for pred in predictions:
                    # Filter by confidence threshold
                    if len(pred['scores']) > 0:
                        high_conf_mask = pred['scores'] >= conf_threshold
                        batch_predictions += high_conf_mask.sum().item()
                
                total_predictions += batch_predictions
                num_batches += 1
                num_images += len(images)
        
        # Use average number of predictions per image as a proxy metric
        # This helps detect if model is learning to make meaningful predictions
        avg_predictions_per_image = total_predictions / max(num_images, 1)
        
        # Return negative value so that more predictions = lower "loss"
        # This heuristic guides early stopping toward models that produce confident detections
        return {'loss': -avg_predictions_per_image}

    def _log_training_history(self, log_dir: Path):
        """Save training history to CSV."""
        csv_path = log_dir / "training_log.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss', 'learning_rate'])
            writer.writeheader()
            writer.writerows(self.training_history)
        
        print(f"Training log saved to: {csv_path}")
