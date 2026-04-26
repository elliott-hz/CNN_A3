"""
Classification Trainer
Training framework for classification models (ResNet50)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import yaml
import csv
from typing import Dict, Any, Tuple
from PIL import Image


class AugmentedDataset(Dataset):
    """
    Custom Dataset with on-the-fly data augmentation.
    Only applies augmentation when __getitem__ is called, saving memory.
    """
    
    def __init__(self, X, y, augment: bool = False):
        """
        Initialize dataset.
        
        Args:
            X: Images array (N, H, W, C) as numpy array
            y: Labels array (N,)
            augment: Whether to apply data augmentation
        """
        self.X = X
        self.y = y
        self.augment = augment
        
        if augment:
            # Define augmentation transforms
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1)
                ),
            ])
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Get image and label
        img = self.X[idx]
        label = self.y[idx]
        
        # Handle both normalized [0,1] and raw [0,255] images
        if img.max() <= 1.0:
            # Image is already normalized [0, 1], convert to uint8 for PIL
            img_uint8 = (img * 255.0).astype('uint8')
        else:
            # Image is in [0, 255] range
            img_uint8 = img.astype('uint8')
        
        # Convert numpy array to PIL Image
        pil_img = Image.fromarray(img_uint8)
        
        # Apply augmentation if enabled
        if self.transform:
            pil_img = self.transform(pil_img)
        
        # Convert to tensor (ToTensor automatically normalizes to [0, 1])
        tensor_img = transforms.ToTensor()(pil_img)
        
        return tensor_img, label


class ClassificationTrainer:
    """
    Training framework for classification models.
    
    Handles training loop, validation, optimization, early stopping, and checkpointing.
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
        self.patience = training_config.get('early_stopping_patience', 7)
        self.use_amp = training_config.get('use_amp', True)
        self.grad_accum_steps = training_config.get('gradient_accumulation_steps', 1)
        self.label_smoothing = training_config.get('label_smoothing', 0.1)
        self.use_class_weighting = training_config.get('class_weighting', True)
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Training state
        self.best_val_acc = 0.0
        self.early_stop_counter = 0
        self.training_history = []
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
    
    def train(self, model, X_train, y_train, X_valid, y_valid, output_dir: str):
        """
        Main training loop with two-phase training (frozen → fine-tune).
        
        Args:
            model: ResNet50Classifier model
            X_train: Training images
            y_train: Training labels
            X_valid: Validation images
            y_valid: Validation labels
            output_dir: Directory to save outputs
        """
        print("=" * 80)
        print("CLASSIFICATION MODEL TRAINING")
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
        
        # Prepare data loaders
        train_loader = self._create_dataloader(X_train, y_train, train=True)
        val_loader = self._create_dataloader(X_valid, y_valid, train=False)
        
        # Calculate class weights if needed
        class_weights = None
        if self.use_class_weighting:
            class_weights = self._calculate_class_weights(y_train)
            print(f"Class weights: {class_weights}")
        
        # Setup loss function
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(
                weight=torch.FloatTensor(class_weights).to(self.device),
                label_smoothing=self.label_smoothing
            )
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)
        
        # Phase 1: Train with frozen backbone
        if self.model_config.get('freeze_backbone', True):
            print("\n" + "=" * 80)
            print("PHASE 1: Training with frozen backbone")
            print("=" * 80)
            
            freeze_epochs = min(10, self.epochs // 3)
            optimizer = model.get_optimizer(lr=self.lr, weight_decay=self.weight_decay, 
                                          optimizer_type=self.optimizer_type)
            
            self._train_phase(model, train_loader, val_loader, criterion, optimizer, 
                            freeze_epochs, model_dir, log_dir, phase_name="Phase 1 (Frozen)")
            
            # Phase 2: Fine-tune with unfrozen backbone
            print("\n" + "=" * 80)
            print("PHASE 2: Fine-tuning with unfrozen backbone")
            print("=" * 80)
            
            model.unfreeze_backbone(unfreeze_all=False)
            finetune_lr = self.lr * 0.1  # Lower learning rate for fine-tuning
            optimizer = model.get_optimizer(lr=finetune_lr, weight_decay=self.weight_decay,
                                          optimizer_type=self.optimizer_type)
            
            remaining_epochs = self.epochs - freeze_epochs
            self._train_phase(model, train_loader, val_loader, criterion, optimizer,
                            remaining_epochs, model_dir, log_dir, phase_name="Phase 2 (Fine-tune)")
        else:
            # No freezing, train all at once
            print("\n" + "=" * 80)
            print("TRAINING: All layers trainable from start")
            print("=" * 80)
            
            optimizer = model.get_optimizer(lr=self.lr, weight_decay=self.weight_decay,
                                          optimizer_type=self.optimizer_type)
            self._train_phase(model, train_loader, val_loader, criterion, optimizer,
                            self.epochs, model_dir, log_dir, phase_name="Training")
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETE")
        print("=" * 80)
        print(f"Best validation accuracy: {self.best_val_acc:.4f}")
        print(f"Best model saved to: {model_dir / 'best_model.pth'}")
        
        # Save final model
        final_model_path = model_dir / "final_model.pth"
        model.save(str(final_model_path))
        
        return self.training_history
    
    def _train_phase(self, model, train_loader, val_loader, criterion, optimizer,
                    num_epochs, model_dir, log_dir, phase_name: str):
        """
        Train for a specific phase.
        
        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            num_epochs: Number of epochs for this phase
            model_dir: Directory to save models
            log_dir: Directory to save logs
            phase_name: Name of the training phase
        """
        for epoch in range(num_epochs):
            # Training
            train_loss, train_acc = self._train_epoch(
                model, train_loader, criterion, optimizer, epoch
            )
            
            # Validation
            val_loss, val_acc = self._validate(model, val_loader, criterion)
            
            # Update learning rate scheduler
            self._update_scheduler(optimizer, val_loss, epoch)
            
            # Log metrics
            metrics = {
                'epoch': epoch + 1,
                'phase': phase_name,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': optimizer.param_groups[0]['lr']
            }
            self.training_history.append(metrics)
            
            # Print progress
            print(f"[{phase_name}] Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.early_stop_counter = 0
                
                best_model_path = model_dir / "best_model.pth"
                model.save(str(best_model_path))
                print(f"  ✓ New best model saved (Val Acc: {val_acc:.4f})")
            else:
                self.early_stop_counter += 1
            
            # Early stopping check - only apply if patience > 0
            if self.patience > 0 and self.early_stop_counter >= self.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
            
            # Save training log
            self._save_training_log(log_dir)
    
    def _train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """
        Train for one epoch.
        
        Args:
            model: The model to train
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            epoch: Current epoch number
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Mixed precision training
            if self.use_amp:
                # Autocast for forward pass and loss calculation
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    
                    # Handle GoogLeNet auxiliary classifiers
                    if isinstance(outputs, tuple):
                        main_out, aux1_out, aux2_out = outputs
                        # Main loss + auxiliary losses (weighted)
                        main_loss = criterion(main_out, targets)
                        aux1_loss = criterion(aux1_out, targets)
                        aux2_loss = criterion(aux2_out, targets)
                        # GoogLeNet uses 0.3 weight for auxiliary losses
                        loss = main_loss + 0.3 * (aux1_loss + aux2_loss)
                    else:
                        loss = criterion(outputs, targets)
                    
                    loss = loss / self.grad_accum_steps
                
                # Backward pass with scaled loss
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad()
            else:
                outputs = model(inputs)
                
                # Handle GoogLeNet auxiliary classifiers
                if isinstance(outputs, tuple):
                    main_out, aux1_out, aux2_out = outputs
                    main_loss = criterion(main_out, targets)
                    aux1_loss = criterion(aux1_out, targets)
                    aux2_loss = criterion(aux2_out, targets)
                    loss = main_loss + 0.3 * (aux1_loss + aux2_loss)
                else:
                    loss = criterion(outputs, targets)
                
                loss = loss / self.grad_accum_steps
                loss.backward()
                
                if (batch_idx + 1) % self.grad_accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Statistics - use main output for accuracy calculation
            if isinstance(outputs, tuple):
                main_out = outputs[0]
            else:
                main_out = outputs
            
            running_loss += loss.item() * self.grad_accum_steps
            _, predicted = main_out.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{running_loss / (batch_idx + 1):.4f}",
                'acc': f"{100. * correct / total:.2f}%"
            })
        
        avg_loss = running_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate(self, model, val_loader, criterion):
        """
        Validate model on validation set.
        
        Args:
            model: The model to validate
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                if self.use_amp:
                    # Autocast for validation forward pass and loss calculation
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        
                        # During validation, only use main output
                        if isinstance(outputs, tuple):
                            main_out = outputs[0]
                        else:
                            main_out = outputs
                        
                        loss = criterion(main_out, targets)
                else:
                    outputs = model(inputs)
                    
                    # During validation, only use main output
                    if isinstance(outputs, tuple):
                        main_out = outputs[0]
                    else:
                        main_out = outputs
                    
                    loss = criterion(main_out, targets)
                
                running_loss += loss.item()
                _, predicted = main_out.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = running_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _create_dataloader(self, X, y, train: bool = True) -> DataLoader:
        """
        Create PyTorch DataLoader with on-the-fly data augmentation.
        
        Args:
            X: Images array (N, H, W, C)
            y: Labels array (N,)
            train: Whether this is training data
            
        Returns:
            PyTorch DataLoader
        """
        # Create custom dataset with optional augmentation
        dataset = AugmentedDataset(X, y, augment=train)
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=2,  # Reduced from 4 to save memory
            pin_memory=True,
            drop_last=train
        )
        
        return dataloader
    
    def _calculate_class_weights(self, y_train):
        """
        Calculate inverse frequency class weights.
        
        Args:
            y_train: Training labels
            
        Returns:
            Array of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        
        return weights
    
    def _update_scheduler(self, optimizer, val_loss, epoch):
        """
        Update learning rate scheduler.
        
        Args:
            optimizer: Optimizer
            val_loss: Current validation loss
            epoch: Current epoch
        """
        # Cosine annealing with warm restarts - more gradual decay
        # Only apply in Phase 2 (fine-tuning) to avoid aggressive early decay
        if epoch > 0 and epoch % 20 == 0:  # Changed from 10 to 20 for slower decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.7  # Changed from 0.5 to 0.7 for gentler decay
            print(f"  Learning rate reduced to {optimizer.param_groups[0]['lr']:.6f}")
    
    def _save_training_log(self, log_dir: Path):
        """
        Save training history to CSV.
        
        Args:
            log_dir: Directory to save logs
        """
        csv_path = log_dir / "training_log.csv"
        
        # Write to CSV
        with open(csv_path, 'w', newline='') as f:
            if self.training_history:
                writer = csv.DictWriter(f, fieldnames=self.training_history[0].keys())
                writer.writeheader()
                writer.writerows(self.training_history)


if __name__ == "__main__":
    # Example usage
    from src.models.classification_model import ResNet50Classifier, BASELINE_CLASSIFICATION_CONFIG
    
    model_config = BASELINE_CLASSIFICATION_CONFIG
    training_config = {
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 30,
        'optimizer': 'adam',
        'weight_decay': 1e-4,
        'early_stopping_patience': 7,
        'use_amp': True,
        'gradient_accumulation_steps': 1,
        'label_smoothing': 0.1,
        'class_weighting': True
    }
    
    trainer = ClassificationTrainer(model_config, training_config)
    model = ResNet50Classifier(model_config)
    
    # trainer.train(model, X_train, y_train, X_valid, y_valid, "outputs/test_run")
