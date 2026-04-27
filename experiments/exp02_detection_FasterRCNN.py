"""
Experiment 02: Detection Model - Faster R-CNN (ResNet50+FPN)

This experiment trains a Faster R-CNN model for dog face detection.
Two-stage detection architecture with high accuracy.
Configuration: ResNet50 backbone + FPN, SGD optimizer, lr=0.005
"""

import sys
import argparse
from pathlib import Path
import glob

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.detection_model import FasterRCNNDetector, FASTER_RCNN_CONFIG
from src.training.torchvision_detection_trainer import TorchvisionDetectionTrainer
from src.data_processing.torchvision_detection_dataset import create_detection_dataloaders
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
    run_dirs = [d for d in base_output_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    
    if not run_dirs:
        return None
    
    # Sort by name (timestamp) to get the latest one
    latest_run = sorted(run_dirs, key=lambda x: x.name)[-1]
    return latest_run


def main():
    """Run Experiment 02: Faster R-CNN Detection."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Experiment 02: Faster R-CNN Detection')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from the latest checkpoint')
    
    args = parser.parse_args()
    
    RESUME_TRAINING = args.resume
    
    # Setup
    experiment_name = "exp02_detection_FasterRCNN"
    
    # Determine output_dir based on whether to resume
    if RESUME_TRAINING:
        output_dir = get_latest_run_dir(experiment_name)
        if output_dir is None:
            print(f"Error: No previous runs found for {experiment_name} to resume.")
            sys.exit(1)
        logger = setup_logger(experiment_name, log_file=output_dir / "logs" / "training.log")
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
    logger.info("\n[Step 1/5] Loading dataset configuration...")
    
    # Use COCO format dataset
    dataset_config_path = Path("data/processed/detection_coco/dataset.yaml")
    
    if not dataset_config_path.exists():
        logger.error(f"Dataset config not found: {dataset_config_path}")
        logger.error("Please run format conversion first: python src/data_processing/convert_detection_format.py")
        sys.exit(1)
    
    with open(dataset_config_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    logger.info(f"Dataset config loaded from: {dataset_config_path}")
    logger.info(f"Dataset root: {dataset_config['path']}")
    logger.info(f"Classes: {dataset_config['nc']} ({dataset_config['names']})")
    
    # Step 2: Initialize model and trainer
    logger.info("[Step 2/5] Initializing model and trainer...")
    
    # Configure Faster R-CNN model - Optimized for faster training
    model_config = FASTER_RCNN_CONFIG.copy()
    model_config['num_classes'] = dataset_config['nc'] + 1  # +1 for background class
    
    # Training configuration optimized for speed
    training_config = {
        'learning_rate': 0.005,
        'batch_size': 8,  # Increased from 4 to utilize GPU memory better
        'epochs': 120,     # Reduced from 120 (faster convergence with larger batch)
        'optimizer': 'sgd',
        'weight_decay': 1e-4,
        'early_stopping_patience': 15,
        'use_amp': True,
        'gradient_accumulation_steps': 1,  # No need with larger batch
        'warmup_epochs': 10,
        'scheduler': 'cosine',
        'resume': RESUME_TRAINING
    }
    
    logger.info(f"Model config: {model_config}")
    logger.info(f"Training config: {training_config}")
    
    # Initialize model
    model = FasterRCNNDetector(model_config)
    
    # Initialize trainer
    trainer = TorchvisionDetectionTrainer(model_config, training_config)
    
    # Create dataloaders
    logger.info("\n[Step 2.5/5] Creating dataloaders...")
    try:
        train_loader, val_loader, test_loader = create_detection_dataloaders(
            dataset_config_path=str(dataset_config_path),
            batch_size=training_config['batch_size'],
            num_workers=4,
            model_type='faster_rcnn'
        )
        logger.info(f"Train loader: {len(train_loader.dataset)} samples")
        logger.info(f"Val loader: {len(val_loader.dataset)} samples")
        logger.info(f"Test loader: {len(test_loader.dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to create dataloaders: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 3: Train model
    logger.info("\n[Step 3/5] Training model...")
    try:
        results = trainer.train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=str(output_dir),
            dataset_config_path=str(dataset_config_path)
        )
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Evaluate model on test set
    logger.info("\n[Step 4/5] Evaluating model on test set...")
    
    # Reload best model weights for evaluation
    best_model_path = output_dir / "model" / "best_model.pt"
    
    if best_model_path.exists():
        logger.info(f"Reloading best model weights from: {best_model_path}")
        model.load(str(best_model_path))
        logger.info("Best model loaded successfully")
    else:
        logger.warning("Best model file not found, using current model state")
    
    # Run evaluation on test set
    try:
        from src.evaluation.detection_evaluator import DetectionEvaluator
        
        evaluator = DetectionEvaluator()
        
        # For torchvision models, we need custom evaluation
        # This is a simplified version - you may want to enhance it
        logger.info("Running test evaluation...")
        
        # Move model to eval mode
        model.model.eval()
        
        # Simple evaluation loop
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        all_predictions = []
        all_ground_truths = []
        
        with torch.no_grad():
            for images, targets in test_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                predictions = model(images)
                
                all_predictions.extend(predictions)
                all_ground_truths.extend(targets)
        
        # Calculate metrics
        map_metrics = trainer._calculate_map(all_predictions, all_ground_truths)
        
        logger.info("\n" + "=" * 80)
        logger.info("TEST EVALUATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"mAP@0.5: {map_metrics['map50']:.4f}")
        # logger.info(f"mAP@0.5:0.95: {map_metrics['map50_95']:.4f}")  # Not available in simplified metrics
        logger.info(f"Precision: {map_metrics['precision']:.4f}")
        logger.info(f"Recall: {map_metrics['recall']:.4f}")
        logger.info(f"True Positives: {map_metrics['true_positives']}")
        logger.info(f"False Positives: {map_metrics['false_positives']}")
        logger.info(f"False Negatives: {map_metrics['false_negatives']}")
        
        # Save evaluation results
        eval_results = {
            'test_map50': map_metrics['map50'],
            # 'test_map50_95': map_metrics['map50_95'],  # Not available in simplified metrics
            'test_precision': map_metrics['precision'],
            'test_recall': map_metrics['recall'],
            'test_true_positives': map_metrics['true_positives'],
            'test_false_positives': map_metrics['false_positives'],
            'test_false_negatives': map_metrics['false_negatives'],
            'best_train_map50': trainer.best_map50,
            'best_epoch': trainer.best_epoch
        }
        
        eval_results_path = output_dir / "evaluation_results.yaml"
        with open(eval_results_path, 'w') as f:
            yaml.dump(eval_results, f)
        
        logger.info(f"Evaluation results saved to: {eval_results_path}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 5: Generate summary report
    logger.info("\n[Step 5/5] Generating summary report...")
    
    summary = f"""
# Experiment 02: Faster R-CNN Detection Results

## Model Configuration
- Architecture: Faster R-CNN with ResNet50+FPN backbone
- Number of classes: {dataset_config['nc']} (dog)
- Input size: {model_config.get('min_size', 800)}x{model_config.get('max_size', 1333)}
- Pretrained: {model_config['pretrained']}

## Training Configuration
- Optimizer: SGD with momentum (0.9)
- Learning rate: {training_config['learning_rate']}
- Batch size: {training_config['batch_size']} (effective: {training_config['batch_size'] * training_config['gradient_accumulation_steps']})
- Epochs: {training_config['epochs']}
- Weight decay: {training_config['weight_decay']}
- AMP: {training_config['use_amp']}
- Scheduler: Cosine annealing with {training_config['warmup_epochs']} warmup epochs

## Dataset
- Format: COCO JSON
- Training samples: {len(train_loader.dataset)}
- Validation samples: {len(val_loader.dataset)}
- Test samples: {len(test_loader.dataset)}

## Results
- Best mAP@0.5 (validation): {trainer.best_map50:.4f} at epoch {trainer.best_epoch}
- Test mAP@0.5: {map_metrics['map50']:.4f}

## Key Characteristics
- Two-stage detection architecture
- High accuracy with slower inference speed
- Region Proposal Network (RPN) generates candidate boxes
- ROI pooling extracts features for classification and regression
"""
    
    summary_path = output_dir / "EXPERIMENT_SUMMARY.md"
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    logger.info(f"Summary report saved to: {summary_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Metrics: mAP@0.5={map_metrics['map50']:.4f}")
    # logger.info(f"Metrics: mAP@0.5={map_metrics['map50']:.4f}, mAP@0.5:0.95={map_metrics['map50_95']:.4f}")


if __name__ == "__main__":
    main()
