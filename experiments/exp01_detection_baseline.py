"""
Experiment 01: Detection Model - Baseline (YOLOv8 Medium)

This experiment trains a baseline YOLOv8 model for dog face detection.
Configuration: backbone='m', input_size=640, confidence=0.5
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing.processed_datasets_verify import verify_processed_datasets
from src.data_processing.detection_preprocessor import DetectionPreprocessor
from src.models.detection_model import YOLOv8Detector, BASELINE_DETECTION_CONFIG
from src.training.detection_trainer import DetectionTrainer
from src.evaluation.detection_evaluator import DetectionEvaluator
from src.utils.file_utils import create_experiment_dir
from src.utils.logger import setup_logger
import yaml


def main():
    """Run Experiment 01: Detection Baseline."""
    
    # Setup
    experiment_name = "exp01_detection_baseline"
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
    logger.info("\n[Step 3/5] Initializing model and trainer...")
    
    # Use baseline configuration
    model_config = BASELINE_DETECTION_CONFIG.copy()
    
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
    
    logger.info(f"Model config: {model_config}")
    logger.info(f"Training config: {training_config}")
    
    # Create output directory
    output_dir = create_experiment_dir(experiment_name)
    
    # Initialize model
    model = YOLOv8Detector(model_config)
    
    # Initialize trainer
    trainer = DetectionTrainer(model_config, training_config)
    
    # Step 4: Train model
    logger.info("\n[Step 4/5] Training model...")
    # Note: For actual training, you need to prepare data in YOLO format
    # This is a placeholder - adapt based on your data structure
    try:
        results = trainer.train(
            model=model,
            train_data="path/to/train/data",  # Replace with actual path
            val_data="path/to/val/data",      # Replace with actual path
            output_dir=str(output_dir)
        )
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.info("Note: Ensure data is prepared in YOLO format before training")
        return
    
    # Evaluate model
    logger.info("\nEvaluating model on test set...")
    evaluator = DetectionEvaluator()
    
    try:
        metrics = evaluator.evaluate(
            model=model,
            test_data="path/to/test/data",  # Replace with actual path
            output_dir=str(output_dir)
        )
        
        # Generate report
        evaluator.generate_report(str(output_dir))
        
        logger.info("\n" + "=" * 80)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"Metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")


if __name__ == "__main__":
    main()
