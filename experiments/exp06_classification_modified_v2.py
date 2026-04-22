"""
Experiment 06: Classification Model - Modified Version 2 (ResNet50 No Freeze)

This experiment trains a modified ResNet50 model without freezing backbone.
Configuration: dropout=0.3, freeze_backbone=False, all layers trainable
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing.processed_datasets_verify import verify_processed_datasets
from src.data_processing.emotion_preprocessor import EmotionPreprocessor
from src.models.classification_model import ResNet50Classifier, MODIFIED_V2_CLASSIFICATION_CONFIG
from src.training.classification_trainer import ClassificationTrainer
from src.evaluation.classification_evaluator import ClassificationEvaluator
from src.utils.file_utils import create_experiment_dir
from src.utils.logger import setup_logger


def main():
    """Run Experiment 06: Classification Modified V2."""
    
    experiment_name = "exp06_classification_modified_v2"
    logger = setup_logger(experiment_name)
    
    logger.info("=" * 80)
    logger.info(f"STARTING EXPERIMENT: {experiment_name}")
    logger.info("=" * 80)
    
    # Step 1: Verify datasets are processed
    logger.info("\n[Step 1/5] Verifying processed datasets...")
    if not verify_processed_datasets():
        logger.error("Datasets not ready. Please run preprocessing first.")
        logger.error("Run: python src/data_processing/detection_preprocessor.py")
        logger.error("Run: python src/data_processing/emotion_preprocessor.py")
        sys.exit(1)
    
    # Step 2: Load preprocessed data
    logger.info("\n[Step 2/5] Loading preprocessed data...")
    preprocessor = EmotionPreprocessor()
    X_train, y_train = preprocessor.load_split('train')
    X_valid, y_valid = preprocessor.load_split('valid')
    X_test, y_test = preprocessor.load_split('test')
    
    logger.info(f"Train: {len(X_train)}, Valid: {len(X_valid)}, Test: {len(X_test)}")
    
    # Step 3: Initialize model and trainer
    logger.info("\n[Step 4/5] Initializing model and trainer...")
    
    model_config = MODIFIED_V2_CLASSIFICATION_CONFIG.copy()
    
    training_config = {
        'learning_rate': 0.0001,  # Very low learning rate for fine-tuning
        'batch_size': 32,
        'epochs': 25,             # Fewer epochs since all layers trainable
        'optimizer': 'sgd',       # SGD with momentum
        'weight_decay': 1e-4,
        'early_stopping_patience': 5,
        'use_amp': True,
        'gradient_accumulation_steps': 1,
        'label_smoothing': 0.05,  # Less smoothing
        'class_weighting': True
    }
    
    logger.info(f"Model config: {model_config}")
    logger.info(f"Training config: {training_config}")
    
    output_dir = create_experiment_dir(experiment_name)
    model = ResNet50Classifier(model_config)
    trainer = ClassificationTrainer(model_config, training_config)
    
    # Step 4: Train and evaluate
    logger.info("\n[Step 5/5] Training model...")
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
    
    logger.info("\nEvaluating model...")
    # Load class names from config
    class_names = ['angry', 'happy', 'relax', 'frown', 'alert']
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
