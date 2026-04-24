"""
Experiment 08: Detection Model - SSD (SSD300 VGG16)

This experiment trains an SSD300 model with VGG16 backbone.
Configuration: input_size=300, confidence=0.5, nms_iou_threshold=0.5

This is a completely standalone experiment that does NOT modify any existing files.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.ssd_detection_model import SSDDetector, SSD_BASELINE_CONFIG
from src.training.ssd_trainer import SSDTrainer
from src.evaluation.ssd_evaluator import SSDEvaluator
from src.utils.file_utils import create_experiment_dir
from src.utils.logger import setup_logger
import yaml


def get_latest_run_dir(experiment_name):
    """Find the latest run directory for resuming."""
    base_output_dir = Path("outputs") / experiment_name
    if not base_output_dir.exists():
        return None
    
    run_dirs = [d for d in base_output_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not run_dirs:
        return None
    
    latest_run = sorted(run_dirs, key=lambda x: x.name)[-1]
    return latest_run


def main():
    """Run Experiment 08: Detection SSD."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Experiment 08: Detection SSD')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from the latest checkpoint')
    parser.add_argument('--small-test', action='store_true',
                        help='Run on small dataset subset for quick local testing (prevents overheating)')
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip training and run evaluation only')
    parser.add_argument('--max-train', type=int, default=None,
                        help='Max training images (for small tests)')
    parser.add_argument('--max-val', type=int, default=None,
                        help='Max validation images (for small tests)')
    parser.add_argument('--max-test', type=int, default=None,
                        help='Max test images (for small tests)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for training')
    
    args = parser.parse_args()
    
    # Small test defaults
    if args.small_test:
        args.max_train = args.max_train or 100
        args.max_val = args.max_val or 30
        args.max_test = args.max_test or 50
        args.epochs = args.epochs if args.epochs != 50 else 3
        args.batch_size = args.batch_size if args.batch_size != 4 else 2
        print("=" * 80)
        print("SMALL TEST MODE: Using limited dataset to prevent overheating")
        print(f"  Train images: {args.max_train}, Val: {args.max_val}, Test: {args.max_test}")
        print(f"  Epochs: {args.epochs}, Batch size: {args.batch_size}")
        print("=" * 80)
    
    RESUME_TRAINING = args.resume
    EVAL_ONLY = args.eval_only
    
    # Setup
    experiment_name = "exp08_detection_ssd"
    
    # Determine output_dir
    if RESUME_TRAINING:
        output_dir = get_latest_run_dir(experiment_name)
        if output_dir is None:
            print(f"Error: No previous runs found for {experiment_name} to resume.")
            sys.exit(1)
        logger = setup_logger(experiment_name, log_file=output_dir / "logs" / "training.log")
        logger.info("=" * 80)
        logger.info(f"RESUMING EXPERIMENT: {experiment_name}")
        logger.info(f"Resuming from: {output_dir}")
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
    logger.info("\n[Step 2/4] Initializing SSD model and trainer...")
    
    model_config = SSD_BASELINE_CONFIG.copy()
    
    training_config = {
        'learning_rate': 0.001,
        'batch_size': 16,
        'epochs': 120,
        'optimizer': 'sgd',       # SGD is standard for SSD
        'weight_decay': 5e-4,
        'early_stopping_patience': 0,  # No early stopping
        'use_amp': False,
        'gradient_accumulation_steps': 1,
        'warmup_epochs': 0,
        'scheduler': 'step'
    }
    
    logger.info(f"Model config: {model_config}")
    logger.info(f"Training config: {training_config}")
    
    # Initialize model
    model = SSDDetector(model_config)
    logger.info(f"Model parameters: {model.get_model_params():,}")
    
    # Initialize trainer
    trainer = SSDTrainer(model_config, training_config)
    
    # Step 3: Train model
    if not EVAL_ONLY:
        logger.info("\n[Step 3/4] Training SSD model...")
        try:
            results = trainer.train(
                model=model,
                train_data=str(dataset_config_path),
                val_data=str(dataset_config_path),
                output_dir=str(output_dir),
                max_train_images=args.max_train,
                max_val_images=args.max_val
            )
            
            logger.info("Training completed successfully!")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        logger.info("\n[Step 3/4] Skipping training (eval-only mode)")
        # Load existing weights
        checkpoint_path = Path(output_dir) / "model" / "last.pt"
        if checkpoint_path.exists():
            logger.info(f"Loading checkpoint: {checkpoint_path}")
            model.load_weights(str(checkpoint_path))
        else:
            logger.warning("No checkpoint found, using pretrained weights")
    
    # Step 4: Evaluate model
    logger.info("\n[Step 4/4] Evaluating SSD model on test set...")
    
    evaluator = SSDEvaluator()
    
    try:
        metrics = evaluator.evaluate(
            model=model,
            test_data=str(dataset_config_path),
            output_dir=str(output_dir),
            max_test_images=args.max_test
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

