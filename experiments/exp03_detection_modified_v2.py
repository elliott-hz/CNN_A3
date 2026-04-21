"""
Experiment 03: Detection Model - Modified Version 2 (YOLOv8 Small)

This experiment trains a modified YOLOv8 model with smaller backbone for faster inference.
Configuration: backbone='s', input_size=640, confidence=0.4
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing.processed_datasets_verify import verify_processed_datasets
from src.data_processing.detection_preprocessor import DetectionPreprocessor
from src.models.detection_model import YOLOv8Detector, MODIFIED_V2_DETECTION_CONFIG
from src.training.detection_trainer import DetectionTrainer
from src.evaluation.detection_evaluator import DetectionEvaluator
from src.utils.file_utils import create_experiment_dir
from src.utils.logger import setup_logger


def main():
    """Run Experiment 03: Detection Modified V2."""
    
    experiment_name = "exp03_detection_modified_v2"
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
    preprocessor = DetectionPreprocessor()
    X_train, y_train = preprocessor.load_split('train')
    X_valid, y_valid = preprocessor.load_split('valid')
    X_test, y_test = preprocessor.load_split('test')
    
    logger.info(f"Train: {len(X_train)}, Valid: {len(X_valid)}, Test: {len(X_test)}")
    
    # Step 3: Initialize model and trainer
    logger.info("\n[Step 4/5] Initializing model and trainer...")
    
    model_config = MODIFIED_V2_DETECTION_CONFIG.copy()
    
    training_config = {
        'learning_rate': 0.002,   # Higher learning rate for smaller model
        'batch_size': 32,         # Larger batch size
        'epochs': 40,             # Fewer epochs
        'optimizer': 'sgd',       # SGD optimizer
        'weight_decay': 1e-4,
        'early_stopping_patience': 8,
        'use_amp': True,
        'gradient_accumulation_steps': 1,
        'warmup_epochs': 3,
        'scheduler': 'step'
    }
    
    logger.info(f"Model config: {model_config}")
    logger.info(f"Training config: {training_config}")
    
    output_dir = create_experiment_dir(experiment_name)
    model = YOLOv8Detector(model_config)
    trainer = DetectionTrainer(model_config, training_config)
    
    # Step 4: Train and evaluate
    logger.info("\n[Step 4/5] Training model...")
    try:
        results = trainer.train(
            model=model,
            train_data="path/to/train/data",
            val_data="path/to/val/data",
            output_dir=str(output_dir)
        )
        
        logger.info("Training completed!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return
    
    logger.info("\nEvaluating model...")
    evaluator = DetectionEvaluator()
    
    try:
        metrics = evaluator.evaluate(
            model=model,
            test_data="path/to/test/data",
            output_dir=str(output_dir)
        )
        
        evaluator.generate_report(str(output_dir))
        
        logger.info("\n" + "=" * 80)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")


if __name__ == "__main__":
    main()
