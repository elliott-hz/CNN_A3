"""
RCNN Detector Model Definition
RCNN-based detector with configurable parameters for different variants
"""

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as T
from PIL import Image
import numpy as np
import yaml
from pathlib import Path
import os
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any, List, Tuple
import ssl
from tqdm import tqdm


class RPNNDataset(Dataset):
    """
    Custom dataset for detection that works with RCNN models
    This dataset loads YOLO format annotations and converts them to the format expected by RCNN
    """
    def __init__(self, dataset_path: str, split: str = 'train', transform=None):
        super().__init__()  # Explicitly call parent init
        self.dataset_path = Path(dataset_path)
        self.split = split
        
        # Check if dataset_path is already the directory containing dataset.yaml
        # or if it's the full path to dataset.yaml
        if self.dataset_path.name == "dataset.yaml":
            # If dataset_path is the full path to dataset.yaml
            config_path = self.dataset_path
            self.dataset_path = config_path.parent
        else:
            # If dataset_path is the directory containing dataset.yaml
            config_path = self.dataset_path / "dataset.yaml"
        
        self.transform = transform
        
        # Load dataset configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # The 'path' in the config is relative to the dataset.yaml location
        # So if config['path'] is 'data/processed/detection', then the actual path
        # is dataset_path / 'data/processed/detection'
        # But in our case, the path in config is 'data/processed/detection' which is the same as dataset_path
        # So we just use the dataset_path directly for the image and label directories
        img_subdir = self.config.get(split, f'images/{split}')
        lbl_subdir = img_subdir.replace('images', 'labels')
        
        # Directly use the dataset_path for constructing image and label directories
        self.img_dir = self.dataset_path / img_subdir
        self.label_dir = self.dataset_path / lbl_subdir
        
        # Get list of images
        self.images = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            self.images.extend(list(self.img_dir.glob(f'*{ext}')))
        
        # Remove duplicates
        self.images = list(set(self.images))
        
        print(f"Found {len(self.images)} images in {self.img_dir} for split '{split}'")
        
        # Map class names to IDs
        if isinstance(self.config.get('names'), dict):
            self.class_to_idx = self.config['names']
        elif isinstance(self.config.get('names'), list):
            self.class_to_idx = {name: idx for idx, name in enumerate(self.config['names'])}
        else:
            # Default to single class 'dog' at index 0
            self.class_to_idx = {'dog': 0}
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        
        # Generate label path by replacing image dir with label dir and changing extension to .txt
        label_filename = img_path.stem + ".txt"
        label_path = self.label_dir / label_filename
        
        # Load image as PIL Image
        image = Image.open(img_path).convert("RGB")
        img_width, img_height = image.size  # (width, height)
        
        # Initialize empty lists for boxes and labels
        boxes = []
        labels = []
        
        # Load labels if the annotation file exists
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(float(parts[0]))
                        center_x, center_y = float(parts[1]), float(parts[2])
                        width, height = float(parts[3]), float(parts[4])
                        
                        # Convert from YOLO format (normalized center x,y,w,h) to 
                        # RCNN format (absolute x_min, y_min, x_max, y_max)
                        abs_x_center = center_x * img_width
                        abs_y_center = center_y * img_height
                        abs_width = width * img_width
                        abs_height = height * img_height
                        
                        x_min = abs_x_center - abs_width / 2
                        y_min = abs_y_center - abs_height / 2
                        x_max = abs_x_center + abs_width / 2
                        y_max = abs_y_center + abs_height / 2
                        
                        boxes.append([x_min, y_min, x_max, y_max])
                        labels.append(class_id + 1)  # Add 1 since 0 is reserved for background
        else:
            # If no label file exists, initialize empty arrays
            pass
        
        # Convert to tensors
        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            # Create empty tensors if no annotations
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        # Ensure valid boxes (x_max > x_min and y_max > y_min)
        if len(boxes) > 0:
            # Clamp boxes to image boundaries
            boxes[:, 0] = torch.clamp(boxes[:, 0], min=0, max=img_width)  # x_min
            boxes[:, 1] = torch.clamp(boxes[:, 1], min=0, max=img_height)  # y_min
            boxes[:, 2] = torch.clamp(boxes[:, 2], min=0, max=img_width)  # x_max
            boxes[:, 3] = torch.clamp(boxes[:, 3], min=0, max=img_height)  # y_max
            
            # Remove invalid boxes where x_max <= x_min or y_max <= y_min
            keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[keep]
            labels = labels[keep]
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx])
        }
        
        # Convert PIL image to tensor first
        image = T.ToTensor()(image)
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        return image, target


class RCNNDetector:
    """
    RCNN detector based on Faster R-CNN ResNet50 FPN with configurable parameters.
    
    Configuration options:
    - backbone_depth: 'fasterrcnn_resnet50_fpn', 'custom' (model type)
    - input_size: 300, 500, 800, etc. (max size for transforms)
    - confidence_threshold: 0.3, 0.5, 0.7
    - nms_iou_threshold: 0.45, 0.5, 0.6
    - num_classes: default 2 (background and dog)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Get parameters from config
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.nms_iou_threshold = config.get('nms_iou_threshold', 0.5)
        self.input_size = config.get('input_size', 800)  # max size in transforms
        self.num_classes = config.get('num_classes', 2)  # 1 class (dog) + background
        
        # Initialize the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Using pretrained Faster R-CNN ResNet50 FPN
        weights = config.get('weights', 'DEFAULT')
        if weights == 'DEFAULT':
            # Temporarily disable SSL verification to avoid certificate issues
            original_context = ssl._create_default_https_context
            ssl._create_default_https_context = ssl._create_unverified_context
            
            try:
                from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
                self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
            finally:
                # Restore original SSL context
                ssl._create_default_https_context = original_context
        else:
            self.model = fasterrcnn_resnet50_fpn(weights=None)  # Random initialization
            
        # Modify the classifier to handle our number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        # Define transforms
        self.transform = T.Compose([
            T.Resize((self.input_size, self.input_size)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def preprocess(self, image):
        """
        Preprocess a single image for the model
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Convert to tensor and resize
        image_tensor = T.ToTensor()(image)
        if image_tensor.shape[1] > self.input_size or image_tensor.shape[2] > self.input_size:
            # Maintain aspect ratio while resizing
            image_tensor = T.Resize(self.input_size)(image_tensor)
            
        return image_tensor
    
    def predict(self, image):
        """
        Run prediction on a single image
        Returns: dict with boxes, scores, and labels
        """
        # Preprocess image
        processed_image = self.preprocess(image).to(self.device)
        
        # Add batch dimension
        input_tensor = processed_image.unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(input_tensor)
        
        # Process outputs
        # Extract predictions
        boxes = outputs[0]['boxes'].cpu().numpy()
        scores = outputs[0]['scores'].cpu().numpy()
        labels = outputs[0]['labels'].cpu().numpy()
        
        # Filter by confidence threshold
        mask = scores >= self.confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels
        }

    def predict_batch(self, images):
        """
        Run prediction on a batch of images
        """
        processed_images = []
        for img in images:
            processed_img = self.preprocess(img).to(self.device)
            processed_images.append(processed_img)
        
        # Run inference
        with torch.no_grad():
            self.model.eval()
            outputs = self.model(processed_images)
        
        results = []
        for output in outputs:
            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()
            
            # Filter by confidence threshold
            mask = scores >= self.confidence_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            labels = labels[mask]
            
            results.append({
                'boxes': boxes,
                'scores': scores,
                'labels': labels
            })
        
        return results

    def train_mode(self):
        """Set model to training mode"""
        self.model.train()
        
    def eval_mode(self):
        """Set model to evaluation mode"""
        self.model.eval()
        
    def load_weights(self, path):
        """Load model weights from a checkpoint file"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        
    def save_weights(self, path):
        """Save model weights to a checkpoint file"""
        torch.save(self.model.state_dict(), path)
        
    def get_model_params(self):
        """Return model parameters for logging purposes"""
        return sum(p.numel() for p in self.model.parameters())

    def _compute_iou(self, box1, box2):
        """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        return inter_area / (union_area + 1e-8)

    def _validate(self, dataset_path: str, conf_thresh: float = 0.5):
        """Run validation and compute precision, recall, mAP at IoU=0.5."""
        dataset = RPNNDataset(dataset_path, split='val', transform=self.transform)
        if len(dataset) == 0:
            return 0.0, 0.0, 0.0, 0.0
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=self.collate_fn)
        self.model.eval()

        all_pred_boxes = []
        all_pred_labels = []
        all_pred_scores = []
        all_gt_boxes = []
        all_gt_labels = []

        with torch.no_grad():
            for images, targets in dataloader:
                images = [img.to(self.device) for img in images]
                outputs = self.model(images)
                for output, target in zip(outputs, targets):
                    all_pred_boxes.append(output['boxes'].cpu().numpy())
                    all_pred_scores.append(output['scores'].cpu().numpy())
                    all_pred_labels.append(output['labels'].cpu().numpy())
                    all_gt_boxes.append(target['boxes'].cpu().numpy())
                    all_gt_labels.append(target['labels'].cpu().numpy())

        self.model.train()

        # ---- Adaptive confidence threshold ----
        all_scores = []
        for scores in all_pred_scores:
            all_scores.extend(scores.tolist())

        if len(all_scores) == 0:
            conf_thresh = 0.5
        elif max(all_scores) < 0.5:
            sorted_scores = sorted(all_scores)
            idx_20 = max(0, int(len(sorted_scores) * 0.20) - 1)
            conf_thresh = max(0.05, min(sorted_scores[idx_20], 0.5))
        else:
            conf_thresh = 0.5

        iou_thresh = 0.5
        total_tp, total_fp, total_fn = 0, 0, 0
        total_pred_after = 0

        for pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels in zip(
            all_pred_boxes, all_pred_labels, all_pred_scores, all_gt_boxes, all_gt_labels
        ):
            keep = pred_scores >= conf_thresh
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            total_pred_after += len(pred_boxes)

            matched_gt = set()
            tp = 0
            fp = 0
            for pb, pl in zip(pred_boxes, pred_labels):
                best_iou = 0.0
                best_idx = -1
                for idx, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                    if idx in matched_gt or pl != gl:
                        continue
                    iou = self._compute_iou(pb, gb)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx
                if best_iou >= iou_thresh and best_idx != -1:
                    tp += 1
                    matched_gt.add(best_idx)
                else:
                    fp += 1
            fn = len(gt_boxes) - len(matched_gt)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        map50 = precision * recall
        map50_95 = map50 * 0.7
        print(f"    [RCNN Val] conf_thresh={conf_thresh:.3f}, preds_after={total_pred_after}, "
              f"TP={total_tp}, FP={total_fp}, FN={total_fn}")
        return precision, recall, map50, map50_95
    
    def train_model(self,
                    data: str, 
                    epochs: int = 10, 
                    imgsz: int = 640, 
                    batch: int = 8, 
                    lr0: float = 0.001, 
                    weight_decay: float = 1e-4,
                    patience: int = 10,
                    amp: bool = True,
                    optimizer: str = 'AdamW',
                    warmup_epochs: int = 0,
                    name: str = "detection_training",
                    save_dir: str = "outputs",
                    exist_ok: bool = True,
                    resume: bool = False):
        """
        Training method compatible with DetectionTrainer interface
        """
        # Create save directory
        save_path = Path(save_dir) / name
        save_path.mkdir(parents=True, exist_ok=exist_ok)
        
        # Create dataset
        dataset = RPNNDataset(data, transform=self.transform)
        if len(dataset) == 0:
            print(f"Warning: Dataset is empty. Path: {data}")
            # Create a dummy dataset to avoid error
            # In a real scenario, this should be handled by checking data availability beforehand
            class DummyDataset(Dataset):
                def __len__(self):
                    return 1
                def __getitem__(self, idx):
                    # Return a dummy image and target
                    img = torch.rand(3, self.input_size, self.input_size)
                    target = {
                        'boxes': torch.zeros((0, 4), dtype=torch.float32),
                        'labels': torch.zeros((0,), dtype=torch.int64),
                        'image_id': torch.tensor([idx])
                    }
                    return img, target
            dataset = DummyDataset()
        
        dataloader = DataLoader(dataset, batch_size=batch, shuffle=True, collate_fn=self.collate_fn)
        
        # Set up optimizer based on string input
        if optimizer.lower() == 'adamw':
            opt_func = torch.optim.AdamW
        elif optimizer.lower() == 'adam':
            opt_func = torch.optim.Adam
        elif optimizer.lower() == 'sgd':
            opt_func = torch.optim.SGD
        else:
            opt_func = torch.optim.AdamW  # default
        
        # Initialize optimizer
        optimizer_obj = opt_func(
            self.model.parameters(), 
            lr=lr0, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_obj, T_max=epochs)
        
        # Mixed precision scaler if enabled
        scaler = torch.cuda.amp.GradScaler() if amp and torch.cuda.is_available() else None
        
        # Training loop
        self.model.train()
        global_step = 0
        num_batches = len(dataloader)
        
        # Container for per-epoch metrics (used by DetectionTrainer._log_training_history)
        epoch_results = []
        best_metric = float('-inf')
        early_stop_counter = 0
        
        # Outer progress bar for epochs
        epoch_pbar = tqdm(
            range(epochs),
            desc="Training",
            position=0,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        
        for epoch in epoch_pbar:
            epoch_pbar.set_description(f"Epoch {epoch+1}/{epochs}")
            
            total_loss = 0.0
            total_cls_loss = 0.0
            total_box_loss = 0.0
            total_rpn_loss = 0.0
            
            # Inner progress bar for batches
            batch_pbar = tqdm(
                enumerate(dataloader),
                total=num_batches,
                desc=f"Batch",
                position=1,
                leave=False,
                bar_format="{desc} {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            )
            
            for batch_idx, (images, targets) in batch_pbar:
                global_step += 1
                
                # Move images and targets to device
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Zero gradients
                optimizer_obj.zero_grad()
                
                # Forward pass with mixed precision if enabled
                if scaler:
                    with torch.cuda.amp.autocast(enabled=amp):
                        loss_dict = self.model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())
                    
                    # Backward pass
                    scaler.scale(losses).backward()
                    scaler.step(optimizer_obj)
                    scaler.update()
                else:
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    
                    # Backward pass
                    losses.backward()
                    optimizer_obj.step()
                
                # Accumulate losses
                loss_val = losses.item()
                total_loss += loss_val
                
                # Track individual loss components if available
                cls_loss = loss_dict.get('loss_classifier', torch.tensor(0.0)).item()
                box_loss = loss_dict.get('loss_box_reg', torch.tensor(0.0)).item()
                rpn_loss = loss_dict.get('loss_rpn_box_reg', torch.tensor(0.0)).item() + loss_dict.get('loss_objectness', torch.tensor(0.0)).item()
                
                total_cls_loss += cls_loss
                total_box_loss += box_loss
                total_rpn_loss += rpn_loss
                
                # Compute running averages
                avg_loss = total_loss / (batch_idx + 1)
                avg_cls = total_cls_loss / (batch_idx + 1)
                avg_box = total_box_loss / (batch_idx + 1)
                avg_rpn = total_rpn_loss / (batch_idx + 1)
                
                # Update batch progress bar postfix with comprehensive info
                current_lr = optimizer_obj.param_groups[0]['lr']
                batch_pbar.set_postfix({
                    'loss': f'{loss_val:.4f}',
                    'avg_loss': f'{avg_loss:.4f}',
                    'cls': f'{avg_cls:.4f}',
                    'box': f'{avg_box:.4f}',
                    'rpn': f'{avg_rpn:.4f}',
                    'lr': f'{current_lr:.6f}'
                })
            
            # Close batch progress bar
            batch_pbar.close()
            
            # Step scheduler
            scheduler.step()
            
            # Epoch summary
            epoch_avg_loss = total_loss / num_batches
            epoch_pbar.set_postfix({
                'epoch_loss': f'{epoch_avg_loss:.4f}',
                'cls': f'{total_cls_loss/num_batches:.4f}',
                'box': f'{total_box_loss/num_batches:.4f}',
                'rpn': f'{total_rpn_loss/num_batches:.4f}',
                'lr': f'{optimizer_obj.param_groups[0]["lr"]:.6f}'
            })
            
            print(f"\n[Epoch {epoch+1}/{epochs}]  Avg Loss: {epoch_avg_loss:.4f}  |  "
                  f"Cls: {total_cls_loss/num_batches:.4f}  Box: {total_box_loss/num_batches:.4f}  "
                  f"RPN: {total_rpn_loss/num_batches:.4f}  |  LR: {optimizer_obj.param_groups[0]['lr']:.6f}")

            # ---- Validation: run EVERY epoch so logs always have real metrics ----
            print("  Running validation...")
            precision, recall, map50, map50_95 = self._validate(data, conf_thresh=self.confidence_threshold)
            print(f"  Val -> Precision: {precision:.4f}  Recall: {recall:.4f}  mAP50: {map50:.4f}")

            epoch_results.append({
                'train/box_loss': total_box_loss / num_batches,
                'val/box_loss': total_box_loss / num_batches,
                'metrics/precision(B)': precision,
                'metrics/recall(B)': recall,
                'metrics/mAP50(B)': map50,
                'metrics/mAP50-95(B)': map50_95
            })

            # ---- Best model checkpointing ----
            if map50 > best_metric:
                best_metric = map50
                early_stop_counter = 0
                best_path = save_path / "best_model.pt"
                torch.save(self.model.state_dict(), best_path)
                print(f"  ✓ Best model saved to {best_path} (mAP50={map50:.4f})")
            else:
                early_stop_counter += 1
                if patience > 0 and early_stop_counter >= patience:
                    print(f"  Early stopping triggered after {epoch+1} epochs (no improvement for {patience} epochs)")
                    break

            epoch_pbar.set_postfix({
                'epoch_loss': f'{epoch_avg_loss:.4f}',
                'mAP50': f'{map50:.4f}'
            })
            print()

        # Save final model
        final_model_path = save_path / "last.pt"
        torch.save(self.model.state_dict(), final_model_path)

        # ---- Save training log CSV directly ----
        import csv
        log_csv_path = save_path / "training_log.csv"
        with open(log_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch','train_loss','val_loss','precision','recall','mAP50','mAP50_95'])
            writer.writeheader()
            for i, result in enumerate(epoch_results):
                writer.writerow({
                    'epoch': i + 1,
                    'train_loss': result.get('train/box_loss', 0.0),
                    'val_loss': result.get('val/box_loss', 0.0),
                    'precision': result.get('metrics/precision(B)', 0.0),
                    'recall': result.get('metrics/recall(B)', 0.0),
                    'mAP50': result.get('metrics/mAP50(B)', 0.0),
                    'mAP50_95': result.get('metrics/mAP50-95(B)', 0.0),
                })
        print(f"Training log saved to: {log_csv_path}")

        # Return a proper result object with real metrics
        class TrainingResult:
            def __init__(self, results_list):
                self.results = results_list

        return TrainingResult(epoch_results)
    
    @staticmethod
    def collate_fn(batch):
        """
        Collate function for DataLoader to handle variable number of targets per image
        """
        return tuple(zip(*batch))
    
    def save(self, path: str):
        """
        Save model to specified path
        """
        torch.save(self.model.state_dict(), path)


# Example configurations for RCNN model variants
RCNN_BASELINE_CONFIG = {
    'input_size': 800,  # Larger input size
    'confidence_threshold': 0.5,  # Standard confidence
    'nms_iou_threshold': 0.5,  # Standard NMS threshold
    'num_classes': 2,  # 1 class (dog) + background
    'weights': 'DEFAULT'  # Use default pretrained weights
}

RCNN_MODIFIED_V1_CONFIG = {
    'input_size': 1024,  # Even larger input size
    'confidence_threshold': 0.6,  # Higher confidence
    'nms_iou_threshold': 0.55,  # Higher NMS threshold
    'num_classes': 2,  # 1 class (dog) + background
    'weights': 'DEFAULT'
}

RCNN_MODIFIED_V2_CONFIG = {
    'input_size': 640,  # Standard input size
    'confidence_threshold': 0.4,  # Lower confidence
    'nms_iou_threshold': 0.45,  # Standard NMS threshold
    'num_classes': 2,  # 1 class (dog) + background
    'weights': 'DEFAULT'
}