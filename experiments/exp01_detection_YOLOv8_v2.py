"""
Experiment 01: Detection Model - Modified Version 2 (YOLOv8 Small)

This experiment trains a modified YOLOv8 model with smaller backbone for faster inference.
Configuration: backbone='s', input_size=640, confidence=0.4
"""

import sys
import argparse
from pathlib import Path
import glob # <--- Added import

sys.path.append(str(Path(__file__).parent.parent))

from src.models.detection_model import YOLOv8Detector, MODIFIED_V2_DETECTION_CONFIG
from src.training.detection_trainer import DetectionTrainer
from src.evaluation.detection_evaluator import DetectionEvaluator
from src.utils.file_utils import create_experiment_dir
from src.utils.logger import setup_logger
import yaml


def get_latest_run_dir(experiment_name):
    """
    Helper function to find the latest run directory for resuming.
    """
    base_output_dir = Path("outputs") / experiment_name
    if not base_output_dir.exists():
        return None
    
    # Find all subdirectories that look like run timestamps
    # Assumes format: run_YYYYMMDD_HHMMSS
    run_dirs = [d for d in base_output_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    
    if not run_dirs:
        return None
    
    # Sort by name (timestamp) to get the latest one
    latest_run = sorted(run_dirs, key=lambda x: x.name)[-1]
    return latest_run


def main():
    """Run Experiment 01: Detection Modified V2."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Experiment 01: Detection Modified V2')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from the latest checkpoint in the latest run directory')
    
    args = parser.parse_args()
    
    # Configuration flags
    RESUME_TRAINING = args.resume
    
    experiment_name = "exp01_detection_YOLOv8_v2"
    
    # --- Key modification 1: Determine output_dir based on whether to resume ---
    if RESUME_TRAINING:
        output_dir = get_latest_run_dir(experiment_name)
        if output_dir is None:
            print(f"Error: No previous runs found for {experiment_name} to resume.")
            sys.exit(1)
        logger = setup_logger(experiment_name, log_file=output_dir / "logs" / "training.log") # Optional: append log
        logger.info("=" * 80)
        logger.info(f"RESUMING EXPERIMENT: {experiment_name}")
        logger.info(f"Resuming from directory: {output_dir}")
        logger.info("=" * 80)
    else:
        output_dir = create_experiment_dir(experiment_name)
        logger = setup_logger(experiment_name)
        logger.info("=" * 80)
        logger.info(f"STARTING NEW EXPERIMENT: {experiment_name}")
        logger.info("=" * 80)
    
    # Step 1: Load dataset configuration
    logger.info("\n[Step 1/4] Loading dataset configuration...")
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
    
    # Step 2: Initialize model and trainer
    logger.info("\n[Step 2/4] Initializing model and trainer...")
    
    model_config = MODIFIED_V2_DETECTION_CONFIG.copy()
    
    # --- Key modification 2: Training Config includes resume flag ---
    training_config = {
        'learning_rate': 0.002,   # Higher learning rate for smaller model
        'batch_size': 32,         # Larger batch size
        'epochs': 120,            # Increased from 40 to 120
        'optimizer': 'sgd',       # SGD optimizer
        'weight_decay': 1e-4,
        'early_stopping_patience': 15,  # Increased proportionally
        'use_amp': True,
        'gradient_accumulation_steps': 1,
        'warmup_epochs': 10,      # Increased to 10% of total epochs
        'scheduler': 'step',
        'resume': RESUME_TRAINING  # <--- Pass resume status to Trainer
    }
    
    logger.info(f"Model config: {model_config}")
    logger.info(f"Training config: {training_config}")
    
    model = YOLOv8Detector(model_config)
    trainer = DetectionTrainer(model_config, training_config)
    
    # Step 3: Train model
    logger.info("\n[Step 3/4] Training model...")
    try:
        results = trainer.train(
            model=model,
            train_data=str(dataset_config_path),  # Pass dataset config path
            val_data=str(dataset_config_path),    # YOLO uses same config for train/val
            output_dir=str(output_dir) # <--- Key: Pass determined output_dir (new or old)
        )
        
        logger.info("Training completed!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Evaluate model
    logger.info("\n[Step 4/4] Evaluating model on test set...")
    
    # Reload best model weights for evaluation
    # Standard Ultralytics saves to weights/best.pt, but your code might move it to model/best_model.pt
    best_model_path = output_dir / "weights" / "best.pt" # Standard Ultralytics path
    
    # Fallback to your custom path if standard path doesn't exist
    if not best_model_path.exists():
        alt_best_path = output_dir / "model" / "best_model.pt"
        if alt_best_path.exists():
            best_model_path = alt_best_path
    
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