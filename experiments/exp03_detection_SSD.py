"""
Experiment 03: Detection Model - SSD (VGG16)

This experiment trains an SSD model for dog face detection.
Configuration: Single-stage multi-scale detector with VGG16 backbone
"""

import sys
from pathlib import Path
import torch
import numpy as np
import random

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.torchvision_detection import SSDDetector
from src.training.torchvision_detection_trainer import TorchvisionDetectionTrainer, DetectionDataset
from src.evaluation.detection_evaluator import DetectionEvaluator
from src.utils.file_utils import create_experiment_dir
from src.utils.logger import setup_logger


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """Run Experiment 03: SSD Detection."""
    
    # Set random seed
    set_seed(42)
    
    # Setup
    experiment_name = "exp03_detection_SSD"
    output_dir = create_experiment_dir(experiment_name)
    logger = setup_logger(experiment_name)
    
    logger.info("=" * 80)
    logger.info(f"STARTING EXPERIMENT: {experiment_name}")
    logger.info("Model: SSD with VGG16 backbone")
    logger.info("=" * 80)
    
    # Step 1: Load dataset
    logger.info("\n[Step 1/5] Loading dataset...")
    
    # SSD can use either COCO or VOC format - here we demonstrate VOC format support
    voc_annotations_path = Path("data/processed/detection_voc/annotations")
    images_base_path = Path("data/processed/detection_voc/images")
    
    # Check if VOC format data exists
    if not voc_annotations_path.exists():
        logger.error(f"VOC annotations not found: {voc_annotations_path}")
        logger.error("Please run conversion first: python src/data_processing/convert_detection_format.py --format voc")
        sys.exit(1)
    
    logger.info(f"Loading VOC format dataset from: {voc_annotations_path}")
    logger.info("Note: SSD experiment uses VOC XML format to demonstrate format flexibility")
    
    # Create datasets (VOC format)
    train_dataset = DetectionDataset(
        images_dir=str(images_base_path / "train"),
        annotations_dir=str(voc_annotations_path / "train"),
        is_voc=True
    )
    
    val_dataset = DetectionDataset(
        images_dir=str(images_base_path / "val"),
        annotations_dir=str(voc_annotations_path / "val"),
        is_voc=True
    )
    
    logger.info(f"Train dataset: {len(train_dataset)} images")
    logger.info(f"Val dataset: {len(val_dataset)} images")
    
    # Step 2: Initialize model and trainer
    logger.info("[Step 2/5] Initializing model and trainer...")
    
    # Model configuration
    model_config = {
        'architecture': 'ssd',
        'backbone': 'vgg16',
        'num_classes': 2,  # background + dog_face
        'pretrained': True
    }
    
    # Training configuration (optimized for T4 GPU)
    training_config = {
        'learning_rate': 0.001,           # Lower LR for fine-tuning
        'batch_size': 16,                 # Larger batch possible (SSD is lighter)
        'epochs': 150,                    # Same epochs as Faster R-CNN for fair comparison
        'optimizer': 'sgd',               # SGD with momentum
        'weight_decay': 5e-4,             # Standard regularization
        'early_stopping_patience': 20,    # Patient early stopping
        'use_amp': True,                  # Mixed precision
        'gradient_accumulation_steps': 1, # No accumulation needed
    }
    
    logger.info(f"Model config: {model_config}")
    logger.info(f"Training config: {training_config}")
    
    # Initialize model
    model = SSDDetector(num_classes=model_config['num_classes'], pretrained=True)
    
    # Initialize trainer
    trainer = TorchvisionDetectionTrainer(model_config, training_config)
    
    # Step 3: Train model
    logger.info("\n[Step 3/5] Training model...")
    try:
        results = trainer.train(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            output_dir=str(output_dir)
        )
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Step 4: Evaluate model
    logger.info("\n[Step 4/5] Evaluating model on test set...")
    
    # Reload best model for evaluation
    best_model_path = output_dir / "model" / "best_model.pt"
    if best_model_path.exists():
        logger.info(f"Loading best model from: {best_model_path}")
        model.load(str(best_model_path))
    else:
        logger.warning("Best model not found, using current model state")
    
    # Create test dataset
    test_annotations_path = Path("data/processed/detection_coco/annotations/instances_test.json")
    test_images_path = Path("data/processed/detection_coco/images/test")
    
    if test_annotations_path.exists():
        test_dataset = DetectionDataset(
            images_dir=str(test_images_path),
            annotations_file=str(test_annotations_path)
        )
        
        logger.info(f"Test dataset: {len(test_dataset)} images")
        
        # Initialize evaluator
        evaluator = DetectionEvaluator()
        
        try:
            metrics = evaluator.evaluate(
                model=model,
                test_dataset=test_dataset,
                output_dir=str(output_dir),
                model_type='torchvision',
                conf_threshold=0.1  # Lowered from 0.5 to capture more predictions
            )
            
            # Generate report
            evaluator.generate_report(str(output_dir))
            
            logger.info("\n" + "=" * 80)
            logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            logger.info(f"Results saved to: {output_dir}")
            logger.info(f"Metrics: mAP@0.5={metrics['mAP50']:.4f}, mAP@0.5:0.95={metrics['mAP50_95']:.4f}")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        logger.warning(f"Test annotations not found: {test_annotations_path}")
        logger.warning("Skipping evaluation")
        
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETED (Evaluation Skipped)")
        logger.info("=" * 80)
        logger.info(f"Model saved to: {output_dir}")

if __name__ == "__main__":
    main()
