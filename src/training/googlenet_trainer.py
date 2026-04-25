"""
Training module for GoogLeNet with auxiliary classifiers.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any, Tuple, List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from torchvision import transforms
from PIL import Image

from src.utils.logger import setup_logger
from src.utils.googlenet_utils import compute_combined_loss
from src.evaluation.googlenet_evaluator import GoogLeNetEvaluator


class AugmentedDataset(Dataset):
    """
    Custom dataset with data augmentation for training.
    """
    
    def __init__(self, X, y, augment=True):
        """
        Args:
            X: Images array (N, H, W, C) in range [0, 1]
            y: Labels array (N,)
            augment: Whether to apply data augmentation
        """
        self.X = X
        self.y = y
        self.augment = augment
        
        # Define augmentation transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ])
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Get image and label
        img = self.X[idx]  # Shape: (H, W, C), values in [0, 1]
        label = self.y[idx]
        
        # Convert to PIL Image for transforms
        # img is in [0, 1], need to convert to [0, 255] for PIL
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        
        # Apply augmentation if enabled
        if self.transform and self.augment:
            img_pil = self.transform(img_pil)
        
        # Convert back to tensor
        img_tensor = transforms.ToTensor()(img_pil)  # Converts to [0, 1] and (C, H, W)
        
        return img_tensor, label


class GoogLeNetTrainer:
    """
    Trainer class specifically designed for GoogLeNet with auxiliary classifiers.
    Handles the complexity of training with auxiliary losses.
    """
    
    def __init__(self, model_config: Dict[str, Any], training_config: Dict[str, Any]):
        """
        Initialize trainer for GoogLeNet with auxiliary classifiers.
        
        Args:
            model_config: Model configuration dictionary
            training_config: Training configuration dictionary
                - learning_rate: Initial learning rate
                - batch_size: Training batch size
                - epochs: Number of epochs
                - optimizer: Optimizer type ('sgd', 'adam', 'adamw')
                - weight_decay: L2 regularization
                - early_stopping_patience: Patience for early stopping
                - use_amp: Enable mixed precision training
                - gradient_accumulation_steps: Steps for gradient accumulation
                - label_smoothing: Label smoothing epsilon
                - class_weighting: Whether to use class weights
        """
        self.model_config = model_config
        self.training_config = training_config
        
        # Extract training parameters
        self.lr = training_config.get('learning_rate', 0.001)
        self.batch_size = training_config.get('batch_size', 32)
        self.epochs = training_config.get('epochs', 30)
        self.optimizer_type = training_config.get('optimizer', 'adam')
        self.weight_decay = training_config.get('weight_decay', 1e-4)
        self.patience = training_config.get('early_stopping_patience', 0)
        self.use_amp = training_config.get('use_amp', False)
        self.grad_accum_steps = training_config.get('gradient_accumulation_steps', 1)
        self.label_smoothing = training_config.get('label_smoothing', 0.0)
        self.class_weighting = training_config.get('class_weighting', False)
        
        # Training state
        self.training_history = []
        self.best_val_acc = 0.0
        self.early_stop_counter = 0
        
        # Auxiliary loss weights
        self.main_weight = training_config.get("main_weight", 1.0)
        self.aux1_weight = training_config.get("aux1_weight", 0.3)
        self.aux2_weight = training_config.get("aux2_weight", 0.3)
    
    def _prepare_dataloaders(self, X_train, y_train, X_valid, y_valid):
        """
        Prepare PyTorch dataloaders from numpy arrays with data augmentation.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_valid: Validation features
            y_valid: Validation labels
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Create augmented dataset for training
        train_dataset = AugmentedDataset(X_train, y_train, augment=True)
        
        # No augmentation for validation
        val_dataset = AugmentedDataset(X_valid, y_valid, augment=False)
        
        # Create dataloaders with improved settings
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,  # Increased for faster loading
            pin_memory=True,  # Faster GPU transfer
            drop_last=True  # Drop last incomplete batch for BN stability
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def _setup_criterion(self, y_train):
        """
        Setup loss criterion based on training config.
        
        Args:
            y_train: Training labels to calculate class weights if needed
            
        Returns:
            Initialized loss criterion
        """
        if self.class_weighting:
            # Calculate class weights
            from sklearn.utils.class_weight import compute_class_weight
            classes = np.unique(y_train)
            weights = compute_class_weight(
                class_weight='balanced',
                classes=classes,
                y=y_train
            )
            weights = torch.FloatTensor(weights).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=self.label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        return criterion
    
    def train(self, model, X_train, y_train, X_valid, y_valid, output_dir):
        """
        Train the model.
        
        Args:
            model: Model to train
            X_train: Training features
            y_train: Training labels
            X_valid: Validation features
            y_valid: Validation labels
            output_dir: Output directory for saving results
            
        Returns:
            Training history
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Prepare data loaders
        train_loader, val_loader = self._prepare_dataloaders(X_train, y_train, X_valid, y_valid)
        
        # Setup loss criterion
        criterion = self._setup_criterion(y_train)
        
        # Setup optimizer
        optimizer = model.get_optimizer(
            lr=self.lr,
            weight_decay=self.weight_decay,
            optimizer_type=self.optimizer_type
        )
        
        # Setup learning rate scheduler - IMPROVED
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            verbose=True,
            min_lr=1e-6
        )
        
        # Setup scaler for mixed precision
        scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Gradient clipping value
        max_grad_norm = 1.0
        
        # Create output directories
        model_dir = Path(output_dir) / "models"
        log_dir = Path(output_dir) / "logs"
        model_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        print("=" * 80)
        print("STARTING TRAINING")
        print("=" * 80)
        print(f"Device: {device}")
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_valid)}")
        print(f"Learning rate: {self.lr}, Batch size: {self.batch_size}, Epochs: {self.epochs}")
        print(f"Optimizer: {self.optimizer_type}, Weight decay: {self.weight_decay}")
        print(f"Mixed precision: {self.use_amp}, Class weighting: {self.class_weighting}")
        print(f"Auxiliary weights - Main: {self.main_weight}, Aux1: {self.aux1_weight}, Aux2: {self.aux2_weight}")
        print("=" * 80)
        
        # Main training loop
        for epoch in range(self.epochs):
            # Training
            model.train()
            train_loss, train_acc = 0.0, 0.0
            num_batches = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
            for batch_idx, (data, target) in enumerate(train_pbar):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                
                if self.use_amp and scaler is not None:
                    with autocast():
                        outputs = model(data)
                        if isinstance(outputs, tuple):
                            main_output, aux1_output, aux2_output = outputs
                            loss, _, _, _ = compute_combined_loss(
                                main_output, aux1_output, aux2_output, target, 
                                criterion, self.main_weight, self.aux1_weight, self.aux2_weight
                            )
                        else:
                            loss = criterion(outputs, target)
                    
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(data)
                    if isinstance(outputs, tuple):
                        main_output, aux1_output, aux2_output = outputs
                        loss, _, _, _ = compute_combined_loss(
                            main_output, aux1_output, aux2_output, target, 
                            criterion, self.main_weight, self.aux1_weight, self.aux2_weight
                        )
                    else:
                        loss = criterion(outputs, target)
                    
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    optimizer.step()
                
                # Calculate accuracy
                with torch.no_grad():
                    if isinstance(outputs, tuple):
                        main_output = outputs[0]  # Use main output for accuracy calculation
                    acc = (main_output.argmax(dim=1) == target).float().mean()
                
                train_loss += loss.item()
                train_acc += acc.item()
                num_batches += 1
                
                # Update progress bar
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{acc.item():.4f}'
                })
            
            avg_train_loss = train_loss / num_batches
            avg_train_acc = train_acc / num_batches
            
            # Validation
            model.eval()
            val_loss, val_acc = 0.0, 0.0
            num_val_batches = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Valid]")
                for data, target in val_pbar:
                    data, target = data.to(device), target.to(device)
                    
                    if self.use_amp and scaler is not None:
                        with autocast():
                            outputs = model(data)
                            if isinstance(outputs, tuple):
                                main_output = outputs[0]  # Use main output for validation
                            else:
                                main_output = outputs
                            
                            loss = criterion(main_output, target)
                    else:
                        outputs = model(data)
                        if isinstance(outputs, tuple):
                            main_output = outputs[0]  # Use main output for validation
                        else:
                            main_output = outputs
                        
                        loss = criterion(main_output, target)
                    
                    # Calculate accuracy
                    acc = (main_output.argmax(dim=1) == target).float().mean()
                    
                    val_loss += loss.item()
                    val_acc += acc.item()
                    num_val_batches += 1
                    
                    # Update progress bar
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{acc.item():.4f}'
                    })
            
            avg_val_loss = val_loss / num_val_batches
            avg_val_acc = val_acc / num_val_batches
            
            # Update learning rate scheduler
            self._update_scheduler(optimizer, avg_val_loss, epoch)
            
            # Log metrics
            metrics = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'train_acc': avg_train_acc,
                'val_loss': avg_val_loss,
                'val_acc': avg_val_acc,
                'lr': optimizer.param_groups[0]['lr']
            }
            self.training_history.append(metrics)
            
            print(f"Epoch {epoch+1}/{self.epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}")
            
            # Save best model
            if avg_val_acc > self.best_val_acc:
                self.best_val_acc = avg_val_acc
                self.early_stop_counter = 0
                
                best_model_path = model_dir / "best_model.pth"
                model.save(str(best_model_path))
                print(f"  ✓ New best model saved (Val Acc: {avg_val_acc:.4f})")
            else:
                # Only increment counter if early stopping is enabled (patience > 0)
                if self.patience > 0:
                    self.early_stop_counter += 1
            
            # Early stopping check
            if self.patience > 0 and self.early_stop_counter >= self.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            # Save training log
            self._save_training_log(log_dir)
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        
        # Save final model
        final_model_path = model_dir / "final_model.pth"
        model.save(str(final_model_path))
        
        return self.training_history
    
    def _update_scheduler(self, optimizer, val_loss, epoch):
        """
        Update learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            val_loss: Current validation loss
            epoch: Current epoch
        """
        # Use ReduceLROnPlateau scheduler if available
        if hasattr(self, 'scheduler'):
            self.scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            if epoch > 0 and epoch % 5 == 0:
                print(f"  Current learning rate: {current_lr:.6f}")
    
    def _save_training_log(self, log_dir):
        """
        Save training history to CSV.
        
        Args:
            log_dir: Directory to save logs
        """
        import csv
        from pathlib import Path
        
        csv_path = log_dir / "training_log.csv"
        
        # Write to CSV
        with open(csv_path, 'w', newline='') as f:
            if self.training_history:
                import csv
                fieldnames = self.training_history[0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.training_history)