#!/usr/bin/env python3
import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm

def coco_to_yolo_bbox(bbox, img_width, img_height):
    """
    Convert COCO bbox format [x, y, width, height] to YOLO format [x_center, y_center, width, height] normalized
    
    Args:
        bbox: COCO bbox [x, y, width, height]
        img_width, img_height: Image dimensions
    
    Returns:
        YOLO bbox [x_center_norm, y_center_norm, width_norm, height_norm]
    """
    x, y, w, h = bbox
    
    # Convert to center coordinates
    x_center = x + w / 2
    y_center = y + h / 2
    
    # Normalize by image dimensions
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    
    return [x_center_norm, y_center_norm, w_norm, h_norm]

def process_coco_to_yolo(input_json, output_label_dir):
    """
    Convert COCO JSON format to YOLO txt format
    
    Args:
        input_json: Path to input COCO JSON file
        output_label_dir: Directory to save YOLO txt files
    """
    print(f"Loading COCO JSON: {input_json}")
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']
    
    print(f"Found {len(images)} images and {len(annotations)} annotations")
    
    # Create output directory
    os.makedirs(output_label_dir, exist_ok=True)
    
    # Create mapping from image_id to image info
    img_id_to_info = {img['id']: img for img in images}
    
    # Group annotations by image_id
    img_id_to_anns = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)
    
    print("Converting to YOLO format...")
    
    # Process each image
    for img_info in tqdm(images):
        img_id = img_info['id']
        file_name = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Get annotations for this image
        img_annotations = img_id_to_anns.get(img_id, [])
        
        # Create output txt file
        base_name = os.path.splitext(file_name)[0]
        output_txt_path = os.path.join(output_label_dir, base_name + '.txt')
        
        with open(output_txt_path, 'w') as f:
            for ann in img_annotations:
                category_id = ann['category_id']
                bbox = ann['bbox']
                
                # Convert to YOLO format
                yolo_bbox = coco_to_yolo_bbox(bbox, img_width, img_height)
                
                # Write to file
                f.write(f"{category_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}\n")
    
    print(f"Conversion complete! YOLO labels saved to: {output_label_dir}")
    
    # Print statistics
    label_count = len([f for f in os.listdir(output_label_dir) if f.endswith('.txt')])
    print(f"Created {label_count} YOLO label files")

def main():
    parser = argparse.ArgumentParser(description='Convert COCO JSON format to YOLO txt format')
    parser.add_argument('--input-json', required=True, help='Input COCO JSON file')
    parser.add_argument('--output-label-dir', required=True, help='Output directory for YOLO txt files')
    
    args = parser.parse_args()
    
    process_coco_to_yolo(args.input_json, args.output_label_dir)

if __name__ == "__main__":
    main()
