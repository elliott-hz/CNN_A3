"""
Experiment 01: Detection Model - Baseline (YOLOv8 Medium)

This experiment trains a baseline YOLOv8 model for dog face detection.
Configuration: backbone='m', input_size=640, confidence=0.5
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.detection_model import YOLOv8Detector, BASELINE_DETECTION_CONFIG
from src.training.detection_trainer import DetectionTrainer
from src.evaluation.detection_evaluator import DetectionEvaluator
from src.utils.file_utils import create_experiment_dir
from src.utils.logger import setup_logger
import yaml


def main():
    """Run Experiment 01: Detection Baseline."""
    
    # Configuration flag for using small subset
    USE_SMALL_SUBSET = True  # Set to False to use full dataset
    
    # Setup
    experiment_name = "exp01_detection_baseline"
    logger = setup_logger(experiment_name)
    
    logger.info("=" * 80)
    logger.info(f"STARTING EXPERIMENT: {experiment_name}")
    if USE_SMALL_SUBSET:
        logger.info("MODE: Using SMALL SUBSET for quick testing")
    else:
        logger.info("MODE: Using FULL DATASET")
    logger.info("=" * 80)
    
    # Step 1: Load dataset configuration
    logger.info("\n[Step 1/5] Loading dataset configuration...")
    
    if USE_SMALL_SUBSET:
        dataset_config_path = Path("data/processed/detection_small/dataset.yaml")
    else:
        dataset_config_path = Path("data/processed/detection/dataset.yaml")
    
    if not dataset_config_path.exists():
        logger.error(f"Dataset config not found: {dataset_config_path}")
        if USE_SMALL_SUBSET:
            logger.error("Please create subset first: python src/data_processing/create_detection_subset.py")
        else:
            logger.error("Please run preprocessing first: bash scripts/run_data_preprocessing.sh")
        sys.exit(1)
    
    with open(dataset_config_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    logger.info(f"Dataset config loaded from: {dataset_config_path}")
    logger.info(f"Dataset root: {dataset_config['path']}")
    logger.info(f"Classes: {dataset_config['nc']} ({dataset_config['names']})")
    
    # Step 2: Initialize model and trainer
    logger.info("[Step 3/5] Initializing model and trainer...")
    
    # Use baseline configuration
    model_config = BASELINE_DETECTION_CONFIG.copy()
    
    training_config = {
        'learning_rate': 0.001,
        'batch_size': 16,
        'epochs': 50,
        'optimizer': 'adam',
        'weight_decay': 1e-4,
        'early_stopping_patience': 10,
        'use_amp': False,
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
    
    # Step 3: Train model
    logger.info("\n[Step 3/5] Training model...")
    try:
        results = trainer.train(
            model=model,
            train_data=str(dataset_config_path),  # Pass dataset config path
            val_data=str(dataset_config_path),    # YOLO uses same config for train/val
            output_dir=str(output_dir)
        )
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Evaluate model
    logger.info("\n[Step 4/5] Evaluating model on test set...")
    
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
        
        # Generate report
        evaluator.generate_report(str(output_dir))
        
        logger.info("\n" + "=" * 80)
        logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"Metrics: {metrics}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
