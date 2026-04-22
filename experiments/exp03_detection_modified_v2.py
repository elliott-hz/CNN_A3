"""
Experiment 03: Detection Model - Modified Version 2 (YOLOv8 Small)

This experiment trains a modified YOLOv8 model with smaller backbone for faster inference.
Configuration: backbone='s', input_size=640, confidence=0.4
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.data_processing.processed_datasets_verify import verify_processed_datasets
from src.models.detection_model import YOLOv8Detector, MODIFIED_V2_DETECTION_CONFIG
from src.training.detection_trainer import DetectionTrainer
from src.evaluation.detection_evaluator import DetectionEvaluator
from src.utils.file_utils import create_experiment_dir
from src.utils.logger import setup_logger
import yaml


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
        logger.error("Run: bash scripts/run_data_preprocessing.sh")
        sys.exit(1)
    
    # Step 2: Load dataset configuration
    logger.info("\n[Step 2/5] Loading dataset configuration...")
    dataset_config_path = Path("data/processed/detection/dataset.yaml")
    
    if not dataset_config_path.exists():
        logger.error(f"Dataset config not found: {dataset_config_path}")
        logger.error("Please run preprocessing first: bash scripts/run_data_preprocessing.sh")
        sys.exit(1)
    
    with open(dataset_config_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    logger.info(f"Dataset config loaded from: {dataset_config_path}")
    logger.info(f"Dataset root: {dataset_config['path']}")
    logger.info(f"Classes: {dataset_config['nc']} ({dataset_config['names']})")
    
    # Step 3: Initialize model and trainer
    logger.info("\n[Step 3/5] Initializing model and trainer...")
    
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
    
    # Step 4: Train model
    logger.info("\n[Step 4/5] Training model...")
    try:
        results = trainer.train(
            model=model,
            train_data=str(dataset_config_path),  # Pass dataset config path
            val_data=str(dataset_config_path),    # YOLO uses same config for train/val
            output_dir=str(output_dir)
        )
        
        logger.info("Training completed!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 5: Evaluate model
    logger.info("\n[Step 5/5] Evaluating model on test set...")
    
    # Reload best model weights for evaluation
    best_model_path = output_dir / "model" / "best_model.pt"
    if best_model_path.exists():
        logger.info(f"Reloading best model weights from: {best_model_path}")
        from ultralytics import YOLO
        best_yolo_model = YOLO(str(best_model_path))
        model.model = best_yolo_model  # Replace internal model
        logger.info("Best model loaded successfully")
    else:
        logger.warning("Best model file not found, using current model state")
    
    evaluator = DetectionEvaluator()
    
    try:
        metrics = evaluator.evaluate(
            model=model,
            test_data=str(dataset_config_path),  # Pass dataset config path
            output_dir=str(output_dir)
        )
        
        evaluator.generate_report(str(output_dir))
        
        logger.info("\n" + "=" * 80)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
