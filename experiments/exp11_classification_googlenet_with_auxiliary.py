"""
Experiment 11: Classification with GoogLeNet with auxiliary classifiers
This experiment replicates exp10 but with auxiliary classifiers enabled.
"""

import os
import sys
import torch
import argparse
from torch.utils.data import DataLoader
import yaml

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.googlenet_model_with_auxiliary import GoogLeNetClassifierWithAuxiliary
from src.training.googlenet_trainer import GoogLeNetTrainer
from src.evaluation.googlenet_evaluator import GoogLeNetEvaluator
from src.data_processing.emotion_preprocessor import EmotionPreprocessor
from src.utils.file_utils import create_experiment_directory


def run_experiment(use_small_subset=False):
    """
    Run the GoogLeNet with auxiliary classifiers experiment.
    
    Args:
        use_small_subset: Whether to use a small subset of data for faster testing
    """
    # Create experiment directory
    exp_dir = create_experiment_directory("exp11_classification_googlenet_with_auxiliary")
    
    # Configuration - mirroring exp10 but with auxiliary classifier specifics
    config = {
        'model': {
            'num_classes': 5,
            'dropout_rate': 0.4,
            'freeze_backbone': False
        },
        'training': {
            'epochs': 120,
            'batch_size': 32,
            'learning_rate': 0.01,  # SGD-specific learning rate
            'weight_decay': 0.0005,  # 5e-4
            'optimizer': 'sgd',
            'momentum': 0.9,
            'label_smoothing': 0.1,
            'use_amp': True,
            'early_stopping_patience': 0,  # Disabled as per spec
            'main_weight': 1.0,
            'aux1_weight': 0.3,
            'aux2_weight': 0.3
        },
        'data': {
            'dataset_path': 'data/splitting/emotion_split',
            'image_size': (224, 224)
        },
        'experiment_name': 'exp11_googlenet_aux'
    }
    
    # Save config
    config_path = os.path.join(exp_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    train_dataset = EmotionPreprocessor(
        split='train',
        dataset_path=config['data']['dataset_path'],
        image_size=config['data']['image_size']
    )
    
    val_dataset = EmotionPreprocessor(
        split='val',
        dataset_path=config['data']['dataset_path'],
        image_size=config['data']['image_size']
    )
    
    test_dataset = EmotionPreprocessor(
        split='test',
        dataset_path=config['data']['dataset_path'],
        image_size=config['data']['image_size']
    )
    
    # Use small subset if requested
    if use_small_subset:
        train_dataset.data = train_dataset.data[:50]
        val_dataset.data = val_dataset.data[:20]
        test_dataset.data = test_dataset.data[:20]
        print("Using small subset of data for testing")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model
    model = GoogLeNetClassifierWithAuxiliary(config['model'])
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Initialize trainer
    trainer = GoogLeNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training'],
        device=device
    )
    
    # Train the model
    print("Starting training...")
    metrics = trainer.train()
    
    # Evaluate on test set
    evaluator = GoogLeNetEvaluator(model, test_loader, device)
    test_results = evaluator.evaluate()
    
    print(f"Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Test Precision: {test_results['precision']:.4f}")
    print(f"Test Recall: {test_results['recall']:.4f}")
    print(f"Test F1 Score: {test_results['f1_score']:.4f}")
    
    # Save results
    results_path = os.path.join(exp_dir, 'results.yaml')
    results = {
        'test_metrics': test_results,
        'training_config': config['training'],
        'model_config': config['model']
    }
    
    with open(results_path, 'w') as f:
        yaml.dump(results, f)
    
    # Plot metrics
    import matplotlib.pyplot as plt
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curve
    ax1.plot(metrics['train_losses'], label='Train Loss')
    ax1.plot(metrics['val_losses'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy curve
    ax2.plot(metrics['train_accuracies'], label='Train Accuracy (approx)')
    ax2.plot(metrics['val_accuracies'], label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(exp_dir, 'training_curves.png'))
    plt.close()
    
    # Plot confusion matrix
    cm_fig = evaluator.plot_confusion_matrix(
        np.array(test_results['confusion_matrix']),
        class_names=['Alert', 'Angry', 'Frown', 'Happy', 'Relax']
    )
    cm_fig.savefig(os.path.join(exp_dir, 'confusion_matrix.png'))
    plt.close(cm_fig)
    
    print(f"Experiment completed. Results saved to {exp_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GoogLeNet with auxiliary classifiers experiment")
    parser.add_argument('--use-small-subset', action='store_true',
                        help="Use a small subset of data for faster testing")
    
    args = parser.parse_args()
    
    run_experiment(use_small_subset=args.use_small_subset)