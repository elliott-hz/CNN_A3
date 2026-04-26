"""
Experiment 04: Classification Model - Modified Version 1 (ResNet50 with Additional Layers)

This experiment trains a modified ResNet50 model with additional FC layers and higher dropout.
Configuration: dropout=0.7, additional_fc_layers=True, pretrained=True
"""

import sys
from pathlib import Path
import argparse
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing.emotion_preprocessor import EmotionPreprocessor
from src.models.classification_model import ResNet50Classifier, MODIFIED_V1_CLASSIFICATION_CONFIG
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
    """Run Experiment 04: Classification Modified V1."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Experiment 04: Classification Modified V1')
    parser.add_argument('--use_small_subset', '-s', action='store_true', 
                        help='Use a small subset of data for quick testing')
    parser.add_argument('--subset_size_per_class', type=int, default=50,
                        help='Number of samples per class when using small subset (default: 50)')
    args = parser.parse_args()
    
    experiment_name = "exp04_classification_ResNet50_v1"
    logger = setup_logger(experiment_name)
    
    logger.info("=" * 80)
    logger.info(f"STARTING EXPERIMENT: {experiment_name}")
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
    
    model_config = MODIFIED_V1_CLASSIFICATION_CONFIG.copy()
    
    # Adjust training configuration for 120 epochs
    training_config = {
        'learning_rate': 0.0005,  # Lower learning rate for complex model
        'batch_size': 16,         # Smaller batch due to larger model
        'epochs': 120,             # Increased from 40 to 120
        'optimizer': 'adamw',     # AdamW optimizer
        'weight_decay': 1e-4,
        'early_stopping_patience': 15,  # Increased to 15 for 120 epochs (12.5% of epochs)
        'use_amp': True,
        'gradient_accumulation_steps': 2,
        'label_smoothing': 0.1,
        'class_weighting': True
    }
    
    logger.info(f"Model config: {model_config}")
    logger.info(f"Training config: {training_config}")
    
    output_dir = create_experiment_dir(experiment_name)
    model = ResNet50Classifier(model_config)
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
        
        logger.info("Training completed!")
        
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
        
        evaluator.generate_report(str(output_dir))
        
        logger.info("\n" + "=" * 80)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"Best Validation Accuracy: {trainer.best_val_acc:.4f}")
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")


if __name__ == "__main__":
    main()