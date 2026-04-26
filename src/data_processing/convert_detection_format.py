"""
Detection Dataset Format Converter

Converts YOLO format detection dataset to COCO JSON and VOC XML formats
for use with Faster R-CNN and SSD models respectively.

Input Format (YOLO):
    data/processed/detection/
    ├── images/{train,val,test}/
    └── labels/{train,val,test}/*.txt  # class x_center y_center width height

Output Formats:
    COCO JSON → data/processed/detection_coco/
    VOC XML   → data/processed/detection_voc/
"""

import json
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
import yaml
from PIL import Image
from typing import Dict, List, Tuple
import shutil


class DetectionFormatConverter:
    """Convert YOLO format detection dataset to COCO and VOC formats."""
    
    def __init__(self, source_dir: str, output_base_dir: str = None):
        """
        Initialize converter.
        
        Args:
            source_dir: Path to YOLO format dataset directory
            output_base_dir: Base directory for converted datasets (default: parent of source_dir)
        """
        self.source_dir = Path(source_dir)
        self.output_base_dir = Path(output_base_dir) if output_base_dir else self.source_dir.parent
        
        # Load dataset config
        dataset_yaml = self.source_dir / "dataset.yaml"
        if not dataset_yaml.exists():
            raise FileNotFoundError(f"Dataset config not found: {dataset_yaml}")
        
        with open(dataset_yaml, 'r') as f:
            self.dataset_config = yaml.safe_load(f)
        
        self.class_names = self.dataset_config['names']
        self.num_classes = self.dataset_config['nc']
        
        print(f"Loaded dataset config:")
        print(f"  Classes: {self.class_names}")
        print(f"  Number of classes: {self.num_classes}")
    
    def _read_yolo_label(self, label_path: Path) -> List[Dict]:
        """
        Read YOLO format label file.
        
        Returns:
            List of annotations: [{'class': int, 'x_center': float, 'y_center': float, 'width': float, 'height': float}]
        """
        annotations = []
        if not label_path.exists():
            return annotations
        
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id, x_center, y_center, width, height = map(float, parts)
                    annotations.append({
                        'class': int(class_id),
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    })
        
        return annotations
    
    def _get_image_dimensions(self, image_path: Path) -> Tuple[int, int]:
        """Get image width and height."""
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    
    def convert_to_coco(self):
        """
        Convert YOLO dataset to COCO JSON format.
        
        Output structure:
            data/processed/detection_coco/
            ├── images/{train,val,test}/  (symlinks or copies)
            ├── annotations/
            │   ├── instances_train.json
            │   ├── instances_val.json
            │   └── instances_test.json
            └── dataset.yaml
        """
        print("\n" + "=" * 80)
        print("Converting to COCO format...")
        print("=" * 80)
        
        coco_dir = self.output_base_dir / "detection_coco"
        coco_dir.mkdir(parents=True, exist_ok=True)
        
        # Create images directory (copy from source)
        images_dir = coco_dir / "images"
        if images_dir.exists():
            shutil.rmtree(images_dir)
        shutil.copytree(self.source_dir / "images", images_dir)
        
        # Create annotations directory
        annotations_dir = coco_dir / "annotations"
        annotations_dir.mkdir(exist_ok=True)
        
        # Process each split
        for split in ['train', 'val', 'test']:
            print(f"\nProcessing {split} split...")
            
            images_split_dir = self.source_dir / "images" / split
            labels_split_dir = self.source_dir / "labels" / split
            
            if not images_split_dir.exists():
                print(f"  Warning: {images_split_dir} not found, skipping...")
                continue
            
            # Build COCO structure
            coco_data = {
                'info': {
                    'description': 'Dog Face Detection Dataset',
                    'version': '1.0',
                    'year': 2026
                },
                'licenses': [],
                'images': [],
                'annotations': [],
                'categories': [
                    {'id': i, 'name': name, 'supercategory': 'dog'}
                    for i, name in enumerate(self.class_names)
                ]
            }
            
            image_id = 0
            annotation_id = 0
            
            # Process all images in this split
            image_files = list(images_split_dir.glob('*.jpg')) + list(images_split_dir.glob('*.png'))
            
            for img_path in image_files:
                # Get image dimensions
                img_width, img_height = self._get_image_dimensions(img_path)
                
                # Add image entry
                coco_data['images'].append({
                    'id': image_id,
                    'file_name': img_path.name,
                    'width': img_width,
                    'height': img_height
                })
                
                # Read corresponding label
                label_path = labels_split_dir / f"{img_path.stem}.txt"
                annotations = self._read_yolo_label(label_path)
                
                # Convert YOLO annotations to COCO format
                for ann in annotations:
                    # Convert normalized coordinates to pixel coordinates
                    x_center_px = ann['x_center'] * img_width
                    y_center_px = ann['y_center'] * img_height
                    width_px = ann['width'] * img_width
                    height_px = ann['height'] * img_height
                    
                    # COCO uses top-left corner
                    x_min = x_center_px - width_px / 2
                    y_min = y_center_px - height_px / 2
                    
                    coco_data['annotations'].append({
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': ann['class'],
                        'bbox': [x_min, y_min, width_px, height_px],
                        'area': width_px * height_px,
                        'iscrowd': 0
                    })
                    annotation_id += 1
                
                image_id += 1
            
            # Save COCO JSON
            output_file = annotations_dir / f"instances_{split}.json"
            with open(output_file, 'w') as f:
                json.dump(coco_data, f, indent=2)
            
            print(f"  ✓ Saved {output_file}")
            print(f"    Images: {len(coco_data['images'])}")
            print(f"    Annotations: {len(coco_data['annotations'])}")
        
        # Create dataset.yaml for COCO format
        # Use relative path from project root for consistency with YOLO format
        dataset_yaml_content = f"""path: data/processed/detection_coco
train: images/train
val: images/val
test: images/test

nc: {self.num_classes}
names: {self.class_names}
"""
        with open(coco_dir / "dataset.yaml", 'w') as f:
            f.write(dataset_yaml_content)
        
        print(f"\n✓ COCO conversion complete: {coco_dir}")
        return coco_dir
    
    def convert_to_voc(self):
        """
        Convert YOLO dataset to VOC XML format.
        
        Output structure:
            data/processed/detection_voc/
            ├── images/{train,val,test}/  (symlinks or copies)
            ├── annotations/{train,val,test}/  (XML files)
            └── dataset.yaml
        """
        print("\n" + "=" * 80)
        print("Converting to VOC format...")
        print("=" * 80)
        
        voc_dir = self.output_base_dir / "detection_voc"
        voc_dir.mkdir(parents=True, exist_ok=True)
        
        # Create images directory (copy from source)
        images_dir = voc_dir / "images"
        if images_dir.exists():
            shutil.rmtree(images_dir)
        shutil.copytree(self.source_dir / "images", images_dir)
        
        # Process each split
        for split in ['train', 'val', 'test']:
            print(f"\nProcessing {split} split...")
            
            images_split_dir = self.source_dir / "images" / split
            labels_split_dir = self.source_dir / "labels" / split
            
            if not images_split_dir.exists():
                print(f"  Warning: {images_split_dir} not found, skipping...")
                continue
            
            # Create annotations directory for this split
            annotations_split_dir = voc_dir / "annotations" / split
            annotations_split_dir.mkdir(parents=True, exist_ok=True)
            
            # Process all images in this split
            image_files = list(images_split_dir.glob('*.jpg')) + list(images_split_dir.glob('*.png'))
            count = 0
            
            for img_path in image_files:
                # Get image dimensions
                img_width, img_height = self._get_image_dimensions(img_path)
                
                # Read corresponding label
                label_path = labels_split_dir / f"{img_path.stem}.txt"
                annotations = self._read_yolo_label(label_path)
                
                # Create VOC XML
                root = ET.Element('annotation')
                
                # Folder
                folder = ET.SubElement(root, 'folder')
                folder.text = split
                
                # Filename
                filename = ET.SubElement(root, 'filename')
                filename.text = img_path.name
                
                # Path (optional)
                path = ET.SubElement(root, 'path')
                path.text = str(img_path)
                
                # Source
                source = ET.SubElement(root, 'source')
                database = ET.SubElement(source, 'database')
                database.text = 'Unknown'
                
                # Size
                size = ET.SubElement(root, 'size')
                width = ET.SubElement(size, 'width')
                width.text = str(img_width)
                height = ET.SubElement(size, 'height')
                height.text = str(img_height)
                depth = ET.SubElement(size, 'depth')
                depth.text = '3'
                
                # Segmented
                segmented = ET.SubElement(root, 'segmented')
                segmented.text = '0'
                
                # Objects
                for ann in annotations:
                    obj = ET.SubElement(root, 'object')
                    
                    name = ET.SubElement(obj, 'name')
                    name.text = self.class_names[ann['class']]
                    
                    pose = ET.SubElement(obj, 'pose')
                    pose.text = 'Unspecified'
                    
                    truncated = ET.SubElement(obj, 'truncated')
                    truncated.text = '0'
                    
                    difficult = ET.SubElement(obj, 'difficult')
                    difficult.text = '0'
                    
                    # Bounding box (VOC uses xmin, ymin, xmax, ymax)
                    bndbox = ET.SubElement(obj, 'bndbox')
                    
                    x_center_px = ann['x_center'] * img_width
                    y_center_px = ann['y_center'] * img_height
                    width_px = ann['width'] * img_width
                    height_px = ann['height'] * img_height
                    
                    xmin = ET.SubElement(bndbox, 'xmin')
                    xmin.text = str(int(x_center_px - width_px / 2))
                    
                    ymin = ET.SubElement(bndbox, 'ymin')
                    ymin.text = str(int(y_center_px - height_px / 2))
                    
                    xmax = ET.SubElement(bndbox, 'xmax')
                    xmax.text = str(int(x_center_px + width_px / 2))
                    
                    ymax = ET.SubElement(bndbox, 'ymax')
                    ymax.text = str(int(y_center_px + height_px / 2))
                
                # Write XML file
                xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
                xml_path = annotations_split_dir / f"{img_path.stem}.xml"
                with open(xml_path, 'w') as f:
                    f.write(xml_str)
                
                count += 1
            
            print(f"  ✓ Saved {count} XML files to {annotations_split_dir}")
        
        # Create dataset.yaml for VOC format
        # Use relative path from project root for consistency with YOLO format
        dataset_yaml_content = f"""path: data/processed/detection_voc
train: images/train
val: images/val
test: images/test

nc: {self.num_classes}
names: {self.class_names}
"""
        with open(voc_dir / "dataset.yaml", 'w') as f:
            f.write(dataset_yaml_content)
        
        print(f"\n✓ VOC conversion complete: {voc_dir}")
        return voc_dir


def main():
    """Main function to run format conversion."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert YOLO detection dataset to COCO and VOC formats')
    parser.add_argument('--source-dir', type=str, default='data/processed/detection',
                        help='Path to YOLO format dataset directory')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Base directory for converted datasets (default: parent of source_dir)')
    parser.add_argument('--format', type=str, choices=['coco', 'voc', 'both'], default='both',
                        help='Output format to generate')
    
    args = parser.parse_args()
    
    # Initialize converter
    converter = DetectionFormatConverter(args.source_dir, args.output_dir)
    
    # Perform conversion
    if args.format in ['coco', 'both']:
        converter.convert_to_coco()
    
    if args.format in ['voc', 'both']:
        converter.convert_to_voc()
    
    print("\n" + "=" * 80)
    print("Conversion completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
