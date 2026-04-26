"""
Experiment 06: Classification Model - GoogLeNet (Inception v1)

This experiment trains a GoogLeNet model for dog emotion classification.
Configuration: dropout=0.5, pretrained=True, freeze_backbone=True, use_auxiliary=True

GoogLeNet Architecture:
- Inception modules with parallel convolutions (1x1, 3x3, 5x5)
- Auxiliary classifiers at intermediate layers for better gradient flow
- Global average pooling instead of fully connected layers
- ~7M parameters (much more efficient than ResNet50)
"""

import sys
from pathlib import Path
import argparse
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing.emotion_preprocessor import EmotionPreprocessor
from src.models.classification_model import GoogLeNetClassifier, GOOGLENET_BASELINE_CONFIG
from src.training.classification_trainer import ClassificationTrainer
from src.evaluation.classification_evaluator import ClassificationEvaluator
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


def main():
    """Run Experiment 06: GoogLeNet Classification."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Experiment 06: GoogLeNet Classification')
    parser.add_argument('--use_small_subset', '-s', action='store_true', 
                        help='Use a small subset of data for quick testing')
    parser.add_argument('--subset_size_per_class', type=int, default=50,
                        help='Number of samples per class when using small subset (default: 50)')
    args = parser.parse_args()
    
    experiment_name = "exp06_classification_GoogLeNet"
    logger = setup_logger(experiment_name)
    
    logger.info("=" * 80)
    logger.info(f"STARTING EXPERIMENT: {experiment_name}")
    logger.info("Model: GoogLeNet/Inception v1 (with auxiliary classifiers)")
    logger.info("=" * 80)
    
    # Step 1: Load preprocessed data
    logger.info("\n[Step 1/4] Loading preprocessed data...")
    preprocessor = EmotionPreprocessor()
    
    # Verify that data has been preprocessed
    if not preprocessor.is_processed():
        logger.error("Emotion dataset has not been preprocessed yet!")
        logger.error("Please run src/data_processing/emotion_preprocessor.py first.")
        sys.exit(1)
    
    # Load data splits with error handling
    try:
        X_train, y_train = preprocessor.load_split('train')
        X_valid, y_valid = preprocessor.load_split('valid')
        X_test, y_test = preprocessor.load_split('test')
    except Exception as e:
        logger.error(f"Failed to load data splits: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Validate loaded data
    logger.info(f"Data loaded successfully:")
    logger.info(f"  Train: {len(X_train)} samples, shape: {X_train.shape[1:]}")
    logger.info(f"  Valid: {len(X_valid)} samples, shape: {X_valid.shape[1:]}")
    logger.info(f"  Test: {len(X_test)} samples, shape: {X_test.shape[1:]}")
    
    if len(X_train) == 0 or len(X_valid) == 0 or len(X_test) == 0:
        logger.error("One or more splits are empty!")
        return
    
    # If using small subset, extract subset from full dataset
    if args.use_small_subset:
        logger.info(f"\n[Data Subset] Creating subset with {args.subset_size_per_class} samples per class...")
        
        X_train, y_train = get_subset_data(X_train, y_train, args.subset_size_per_class)
        X_valid, y_valid = get_subset_data(X_valid, y_valid, min(args.subset_size_per_class//4, 10))  # Use fewer validation samples
        X_test, y_test = get_subset_data(X_test, y_test, min(args.subset_size_per_class//4, 10))  # Use fewer test samples
        
        logger.info(f"Subset created:")
        logger.info(f"  Train: {len(X_train)} samples")
        logger.info(f"  Valid: {len(X_valid)} samples")
        logger.info(f"  Test: {len(X_test)} samples")
    
    # Step 2: Initialize model and trainer
    logger.info("\n[Step 2/4] Initializing model and trainer...")
    
    model_config = GOOGLENET_BASELINE_CONFIG.copy()
    
    # Training configuration optimized for GoogLeNet
    training_config = {
        'learning_rate': 0.001,       # Standard LR for GoogLeNet
        'batch_size': 32,             # Moderate batch size
        'epochs': 120,                 # Same as other experiments for fair comparison
        'optimizer': 'adam',          # Adam optimizer works well with Inception
        'weight_decay': 1e-4,
        'early_stopping_patience': 15, # Early stopping enabled
        'use_amp': True,              # Mixed precision training
        'gradient_accumulation_steps': 1,
        'label_smoothing': 0.1,
        'class_weighting': True
    }
    
    logger.info(f"Model config: {model_config}")
    logger.info(f"Training config: {training_config}")
    
    output_dir = create_experiment_dir(experiment_name)
    
    # Initialize GoogLeNet model
    model = GoogLeNetClassifier(model_config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    trainer = ClassificationTrainer(model_config, training_config)
    
    # Step 3: Train model
    logger.info("\n[Step 3/4] Training model...")
    try:
        history = trainer.train(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            output_dir=str(output_dir)
        )
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 4: Evaluate model
    logger.info("\n[Step 4/4] Evaluating model on test set...")
    
    # Load class names dynamically from preprocessor instead of hardcoding
    class_names = preprocessor.classes  # Using dynamic class names from preprocessor
    logger.info(f"Using class names: {class_names}")
    
    evaluator = ClassificationEvaluator(class_names=class_names)
    
    try:
        metrics = evaluator.evaluate(
            model=model,
            X_test=X_test,
            y_test=y_test,
            output_dir=str(output_dir)
        )
        
        # Generate report
        evaluator.generate_report(str(output_dir))
        
        logger.info("\n" + "=" * 80)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"Best Validation Accuracy: {trainer.best_val_acc:.4f}")
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Test Precision: {metrics['precision']:.4f}")
        logger.info(f"Test Recall: {metrics['recall']:.4f}")
        logger.info(f"Test F1-Score: {metrics['f1_score']:.4f}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
