"""
Training module for GoogLeNet with auxiliary classifiers.
"""

import os
import json
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
from tqdm import tqdm
from typing import Dict, Any, Tuple, List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import setup_logger
from src.utils.googlenet_utils import compute_combined_loss
from src.evaluation.googlenet_evaluator import GoogLeNetEvaluator


class GoogLeNetTrainer:
    """
    Trainer class specifically designed for GoogLeNet with auxiliary classifiers.
    Handles the complexity of training with auxiliary losses.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: Dict[str, Any],
        device: torch.device = None
    ):
        """
        Initialize trainer for GoogLeNet with auxiliary classifiers.
        
        Args:
            model: GoogLeNet model with auxiliary classifiers
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration dictionary
            device: Device to train on
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = setup_logger(config.get("experiment_name", "googlenet_training"))
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        optimizer_params = {
            "lr": config.get("learning_rate", 1e-3),
            "weight_decay": config.get("weight_decay", 1e-4)
        }
        
        if config.get("optimizer", "sgd").lower() == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                **optimizer_params,
                momentum=config.get("momentum", 0.9)
            )
        elif config.get("optimizer", "sgd").lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                **optimizer_params
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                **optimizer_params
            )
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.get("label_smoothing", 0.0))
        
        # Setup metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Early stopping parameters
        self.early_stopping_patience = config.get("early_stopping_patience", 0)
        self.early_stopping_counter = 0
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Mixed precision training
        self.use_amp = config.get("use_amp", False)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Auxiliary loss weights
        self.main_weight = config.get("main_weight", 1.0)
        self.aux1_weight = config.get("aux1_weight", 0.3)
        self.aux2_weight = config.get("aux2_weight", 0.3)
        
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    outputs = self.model(data)
                    if isinstance(outputs, tuple):
                        main_output, aux1_output, aux2_output = outputs
                        loss, _, _, _ = compute_combined_loss(
                            main_output, aux1_output, aux2_output, targets, 
                            self.criterion, self.main_weight, self.aux1_weight, self.aux2_weight
                        )
                    else:
                        # Fallback for models without auxiliary classifiers
                        outputs = self.model(data)
                        loss = self.criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(data)
                if isinstance(outputs, tuple):
                    main_output, aux1_output, aux2_output = outputs
                    loss, _, _, _ = compute_combined_loss(
                        main_output, aux1_output, aux2_output, targets, 
                        self.criterion, self.main_weight, self.aux1_weight, self.aux2_weight
                    )
                else:
                    # Fallback for models without auxiliary classifiers
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)
                
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        num_batches = 0
        
        with torch.no_grad():
            for data, targets in tqdm(self.val_loader, desc="Validating"):
                data, targets = data.to(self.device), targets.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(data)
                        if isinstance(outputs, tuple):
                            main_output = outputs[0]  # Take main output for validation metrics
                        else:
                            main_output = outputs
                        
                        loss = self.criterion(main_output, targets)
                else:
                    outputs = self.model(data)
                    if isinstance(outputs, tuple):
                        main_output = outputs[0]  # Take main output for validation metrics
                    else:
                        main_output = outputs
                    
                    loss = self.criterion(main_output, targets)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Collect predictions for metrics calculation
                preds = torch.argmax(main_output, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets.cpu().numpy())
        
        avg_loss = total_loss / num_batches
        accuracy = accuracy_score(all_targets, all_preds)
        
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        
        return avg_loss, accuracy
    
    def train(self) -> Dict[str, List]:
        """Train the model for the specified number of epochs."""
        self.logger.info(f"Starting training for {self.config['epochs']} epochs...")
        
        for epoch in range(self.config['epochs']):
            self.logger.info(f"Epoch {epoch+1}/{self.config['epochs']}")
            
            # Train for one epoch
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Calculate train accuracy (approximately)
            train_acc = max(0.0, 1.0 - train_loss / 5.0)  # Rough estimate
            
            self.train_accuracies.append(train_acc)
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch+1}: "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_acc:.4f}"
            )
            
            # Early stopping check
            if self.early_stopping_patience > 0:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.early_stopping_counter = 0
                    
                    # Save best model
                    self.save_checkpoint(f"best_model.pth")
                else:
                    self.early_stopping_counter += 1
                    if self.early_stopping_counter >= self.early_stopping_patience:
                        self.logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                # Always save the latest model when early stopping is disabled
                self.save_checkpoint(f"latest_model.pth")
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies
        }
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': len(self.train_losses),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
        }
        torch.save(checkpoint, filepath)