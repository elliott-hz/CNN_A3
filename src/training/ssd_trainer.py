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
            
            # Validation every few epochs
            if (epoch + 1) % 5 == 0 or epoch == self.epochs - 1:
                val_loss = self._validate(model, val_loader)
                print(f"  Validation Loss: {val_loss:.4f}")
                history[-1]['val_loss'] = val_loss
                
                # Early stopping check (using validation loss)
                if val_loss < self.best_metric or self.best_metric == float('-inf'):
                    self.best_metric = val_loss
                    self.early_stop_counter = 0
                    # Save best model
                    best_path = model_dir / "best_model.pt"
                    model.save(str(best_path))
                    print(f"  ✓ Best model saved to {best_path}")
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
        """Run validation and return average loss"""
        model.model.train()  # SSD needs train mode to compute losses
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                loss_dict = model.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                total_loss += losses.item()
                num_batches += 1
        
        model.model.eval()
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def _save_history(self, history, log_dir: Path):
        """Save training history to CSV"""
        import csv
        csv_path = log_dir / "training_log.csv"
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'val_loss'])
            writer.writeheader()
            for row in history:
                writer.writerow(row)
        
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

