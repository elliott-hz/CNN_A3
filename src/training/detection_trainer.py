"""
Detection Trainer
Training framework for detection models (YOLOv8)
"""

import torch
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import yaml
import csv
from typing import Dict, Any


class DetectionTrainer:
    """
    Training framework for detection models.
    
    Handles training loop, validation, optimization, and checkpointing.
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
                - optimizer: Optimizer type ('sgd', 'adam', 'adamw')
                - weight_decay: L2 regularization
                - early_stopping_patience: Patience for early stopping
                - use_amp: Enable mixed precision training
                - gradient_accumulation_steps: Steps for gradient accumulation
                - warmup_epochs: Warmup period
        """
        self.model_config = model_config
        self.training_config = training_config
        
        # Extract training parameters
        self.lr = training_config.get('learning_rate', 0.001)
        self.batch_size = training_config.get('batch_size', 16)
        self.epochs = training_config.get('epochs', 50)
        self.optimizer_type = training_config.get('optimizer', 'adam')
        self.weight_decay = training_config.get('weight_decay', 1e-4)
        self.patience = training_config.get('early_stopping_patience', 10)
        self.use_amp = training_config.get('use_amp', True)
        self.grad_accum_steps = training_config.get('gradient_accumulation_steps', 1)
        self.warmup_epochs = training_config.get('warmup_epochs', 5)
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Training state
        self.best_metric = float('-inf')
        self.early_stop_counter = 0
        self.training_history = []
    
    def train(self, model, train_data, val_data, output_dir: str):
        """
        Main training loop.
        
        Args:
            model: YOLOv8Detector model
            train_data: Training dataset or path
            val_data: Validation dataset or path
            output_dir: Directory to save outputs
        """
        print("=" * 80)
        print("DETECTION MODEL TRAINING")
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
        
        # For YOLOv8, we use the built-in training method
        # Prepare data paths (assuming data is in YOLO format)
        # This is a simplified version - adapt based on your data structure
        
        # Check if there's a previous checkpoint to resume from
        last_checkpoint = model_dir / "last.pt"
        resume_training = last_checkpoint.exists()
        
        if resume_training:
            print(f"\n✓ Found previous checkpoint: {last_checkpoint}")
            print("  Resuming training from last checkpoint...")
        else:
            print("\n  Starting fresh training...")
        
        try:
            # Train using YOLO's built-in training
            results = model.train_model(
                data=train_data,  # Path to dataset config or data
                epochs=self.epochs,
                imgsz=self.model_config.get('input_size', 640),
                batch=self.batch_size,
                lr0=self.lr,
                weight_decay=self.weight_decay,
                patience=self.patience,
                amp=self.use_amp,
                optimizer=self.optimizer_type.upper(),  # Pass optimizer type
                warmup_epochs=self.warmup_epochs,       # Pass warmup epochs
                name="detection_training",
                save_dir=str(output_dir),  # Use save_dir instead of project to avoid nested structure
                exist_ok=True,
                resume=resume_training  # Enable resume if checkpoint exists
            )
            
            print("\n" + "=" * 80)
            print("TRAINING COMPLETE")
            print("=" * 80)
            
            # Save best model
            best_model_path = model_dir / "best_model.pt"
            model.save(str(best_model_path))
            print(f"Best model saved to: {best_model_path}")
            
            # Log training history
            self._log_training_history(results, log_dir)
            
            return results
            
        except Exception as e:
            print(f"Error during training: {e}")
            raise
    
    def _log_training_history(self, results, log_dir: Path):
        """
        Log training history to CSV file.
        
        Args:
            results: Training results from YOLO
            log_dir: Directory to save logs
        """
        # Create CSV log
        csv_path = log_dir / "training_log.csv"
        
        # Write header
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'val_loss', 'precision', 'recall', 'mAP50', 'mAP50_95'])
            
            # Write epoch data (adapt based on actual results structure)
            if hasattr(results, 'results'):
                for i, result in enumerate(results.results):
                    writer.writerow([
                        i + 1,
                        result.get('train/box_loss', 0),
                        result.get('val/box_loss', 0),
                        result.get('metrics/precision(B)', 0),
                        result.get('metrics/recall(B)', 0),
                        result.get('metrics/mAP50(B)', 0),
                        result.get('metrics/mAP50-95(B)', 0)
                    ])
        
        print(f"Training log saved to: {csv_path}")
    
    def validate(self, model, val_data):
        """
        Validate model on validation set.
        
        Args:
            model: YOLOv8Detector model
            val_data: Validation dataset
            
        Returns:
            Validation metrics
        """
        # Use YOLO's validation method
        results = model.model.val(data=val_data)
        
        metrics = {
            'mAP50': results.box.map50,
            'mAP50_95': results.box.map,
            'precision': results.box.mp,
            'recall': results.box.mr
        }
        
        return metrics


if __name__ == "__main__":
    # Example usage
    from src.models.detection_model import YOLOv8Detector, BASELINE_DETECTION_CONFIG
    
    model_config = BASELINE_DETECTION_CONFIG
    training_config = {
        'learning_rate': 0.001,
        'batch_size': 16,
        'epochs': 50,
        'optimizer': 'adam',
        'weight_decay': 1e-4,
        'early_stopping_patience': 10,
        'use_amp': True,
        'gradient_accumulation_steps': 1,
        'warmup_epochs': 5,
        'scheduler': 'cosine'
    }
    
    trainer = DetectionTrainer(model_config, training_config)
    model = YOLOv8Detector(model_config)
    
    # trainer.train(model, train_data, val_data, "outputs/test_run")
