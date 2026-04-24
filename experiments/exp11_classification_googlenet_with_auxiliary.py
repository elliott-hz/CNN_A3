"""
Experiment 11: Classification with GoogLeNet with auxiliary classifiers
This experiment replicates exp10 but with auxiliary classifiers enabled.
"""

import os
import sys
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
import yaml
from sklearn.model_selection import train_test_split

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.googlenet_model_with_auxiliary import GoogLeNetClassifierWithAuxiliary
from src.training.googlenet_trainer import GoogLeNetTrainer
from src.evaluation.googlenet_evaluator import GoogLeNetEvaluator
from src.data_processing.emotion_preprocessor import EmotionPreprocessor
from src.utils.file_utils import create_experiment_dir
from src.utils.logger import setup_logger


def get_subset_data(X, y, subset_size_per_class=50, random_seed=42):
    """
    Randomly sample a subset of data while maintaining class balance.
    
    Args:
        X: Input features/images
        y: Labels
        subset_size_per_class: Number of samples per class to include in subset
        random_seed: Random seed for reproducibility
        
    Returns:
        Subset of X and y with balanced classes
    """
    np.random.seed(random_seed)
    
    unique_labels = np.unique(y)
    subset_indices = []
    
    for label in unique_labels:
        # Find all indices with this label
        label_indices = np.where(y == label)[0]
        
        # Randomly select subset_size_per_class samples from this class
        if len(label_indices) >= subset_size_per_class:
            selected_indices = np.random.choice(label_indices, subset_size_per_class, replace=False)
        else:
            # If class has fewer samples than desired, use all available samples
            selected_indices = label_indices
        
        subset_indices.extend(selected_indices)
    
    # Convert to numpy array and shuffle to mix classes
    subset_indices = np.array(subset_indices)
    np.random.shuffle(subset_indices)
    
    return X[subset_indices], y[subset_indices]


def run_experiment(use_small_subset=False):
    """
    Run the GoogLeNet with auxiliary classifiers experiment.
    
    Args:
        use_small_subset: Whether to use a small subset of data for faster testing
    """
    # Create experiment directory
    exp_dir = create_experiment_dir("exp11_classification_googlenet_with_auxiliary")
    
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
            'aux2_weight': 0.3,
            'gradient_accumulation_steps': 1,
            'class_weighting': True
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
    preprocessor = EmotionPreprocessor()
    
    # Verify that data has been preprocessed
    if not preprocessor.is_processed():
        print("Emotion dataset has not been preprocessed yet!")
        print("Please run src/data_processing/emotion_preprocessor.py first.")
        sys.exit(1)
    
    # Load data splits with error handling
    try:
        X_train, y_train = preprocessor.load_split('train')
        X_val, y_val = preprocessor.load_split('val')
        X_test, y_test = preprocessor.load_split('test')
    except Exception as e:
        print(f"Failed to load data splits: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Validate loaded data
    print(f"Data loaded successfully:")
    print(f"  Train: {len(X_train)} samples, shape: {X_train.shape[1:]}")
    print(f"  Valid: {len(X_val)} samples, shape: {X_val.shape[1:]}")
    print(f"  Test: {len(X_test)} samples, shape: {X_test.shape[1:]}")
    
    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        print("One or more splits are empty!")
        return
    
    # If using small subset, extract subset from full dataset
    if use_small_subset:
        print(f"\n[Data Subset] Creating subset with 20 samples per class...")
        
        X_train, y_train = get_subset_data(X_train, y_train, 20)
        X_val, y_val = get_subset_data(X_val, y_val, 5)  # Use fewer validation samples
        X_test, y_test = get_subset_data(X_test, y_test, 5)  # Use fewer test samples
        
        print(f"Subset created:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Valid: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples")
    
    # Initialize model
    model = GoogLeNetClassifierWithAuxiliary(config['model'])
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Initialize trainer
    trainer = GoogLeNetTrainer(
        model_config=config['model'],
        training_config=config['training']
    )
    
    # Train the model
    print("Starting training...")
    metrics = trainer.train(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_valid=X_val,
        y_valid=y_val,
        output_dir=str(exp_dir)
    )
    
    # Evaluate on test set
    # Need to create a temporary dataloader for test evaluation
    X_test_tensor = torch.from_numpy(X_test).permute(0, 3, 1, 2).float()
    y_test_tensor = torch.from_numpy(y_test).long()
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
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
    ax1.plot([m['train_loss'] for m in metrics], label='Train Loss')
    ax1.plot([m['val_loss'] for m in metrics], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy curve
    ax2.plot([m['train_acc'] for m in metrics], label='Train Accuracy')
    ax2.plot([m['val_acc'] for m in metrics], label='Validation Accuracy')
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