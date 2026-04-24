"""
Experiment 11: Classification with GoogLeNet Model with Auxiliary Classifiers
Implementation of GoogLeNet architecture with auxiliary classifiers for dog emotion classification
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.googlenet_model_with_auxiliary import (
    GoogLeNetClassifierWithAuxiliary, 
    create_googlenet_with_auxiliary_model,
    STANDARD_GOOGLENET_WITH_AUX_CONFIG
)
from training.googlenet_trainer import CustomClassificationTrainer
from evaluation.googlenet_evaluator import CustomClassificationEvaluator
from data_processing.emotion_preprocessor import create_emotion_dataloaders
from utils.logger import Logger


def run_experiment(config_path: str = None):
    """
    Run the GoogLeNet with auxiliary classifiers experiment.
    
    Args:
        config_path: Path to the configuration file (optional)
    """
    # Use default config if no config_path provided
    if config_path is None:
        config = {
            'model': STANDARD_GOOGLENET_WITH_AUX_CONFIG,
            'data': {
                'path': 'data/splitting/emotion_split',
                'batch_size': 32,
            },
            'training': {
                'epochs': 120,
                'learning_rate': 0.001,
                'weight_decay': 0.0005,
                'optimizer': 'adam',
                'scheduler': 'step',
                'patience': 15,
                'main_weight': 1.0,
                'aux1_weight': 0.3,
                'aux2_weight': 0.3,
            }
        }
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("outputs") / f"exp11_googlenet_with_auxiliary_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    print("Creating model...")
    model = create_googlenet_with_auxiliary_model(config['model'])
    model.to(device)
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = create_emotion_dataloaders(
        data_path=config['data']['path'],
        batch_size=config['data'].get('batch_size', 32),
        transforms=config['data'].get('transforms', {})
    )
    
    # Create trainer
    print("Initializing trainer...")
    trainer_config = config['training']
    trainer = CustomClassificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config,
        device=device
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Evaluate the model
    print("Evaluating model...")
    evaluator = CustomClassificationEvaluator(
        model=model,
        test_loader=test_loader,
        device=device
    )
    results = evaluator.evaluate()
    
    # Save results
    results_path = output_dir / "results.yaml"
    with open(results_path, 'w') as f:
        yaml.dump(results, f)
    
    # Save model
    model_save_path = output_dir / "best_model.pth"
    model.save(model_save_path)
    
    # Create experiment log
    log_content = f"""
GoogLeNet with Auxiliary Classifiers Experiment
==============================================

Timestamp: {timestamp}
Device: {device}

Model Configuration:
- Architecture: GoogLeNet with Auxiliary Classifiers
- Number of classes: {config['model']['num_classes']}
- Dropout rate: {config['model']['dropout_rate']}
- Freeze backbone: {config['model']['freeze_backbone']}

Training Configuration:
- Epochs: {config['training']['epochs']}
- Learning Rate: {config['training']['learning_rate']}
- Weight Decay: {config['training']['weight_decay']}
- Optimizer: {config['training']['optimizer']}
- Patience: {config['training']['patience']}
- Main Loss Weight: {config['training']['main_weight']}
- Aux1 Loss Weight: {config['training']['aux1_weight']}
- Aux2 Loss Weight: {config['training']['aux2_weight']}

Results:
- Accuracy: {results['accuracy']:.4f}
- Classification Report:
{yaml.dump(results['classification_report'])}

Model saved to: {model_save_path}
Detailed results saved to: {results_path}
"""
    
    log_path = output_dir / "experiment_log.md"
    with open(log_path, 'w') as f:
        f.write(log_content)
    
    print(f"Experiment completed. Results saved to {output_dir}")


if __name__ == "__main__":
    # Allow passing config file as argument
    config_file = sys.argv[1] if len(sys.argv) > 1 else None
    run_experiment(config_file)