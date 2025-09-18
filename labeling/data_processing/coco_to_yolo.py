#!/usr/bin/env python3
import json
import os
from pathlib import Path
import shutil
from tqdm import tqdm
import argparse
import time

def convert_coco_to_yolo(coco_file, output_dir, img_dir=None, copy_images=False, verbose=False):
    """
    Convert COCO format annotations to YOLOv8 format.
    
    Args:
        coco_file: Path to the COCO format JSON file
        output_dir: Directory to save YOLO format annotations
        img_dir: Directory containing source images (if copy_images=True)
        copy_images: Whether to copy images to the output directory
        verbose: Whether to print detailed information
    """
    start_time = time.time()
    
    # Create output directories
    labels_dir = os.path.join(output_dir)
    os.makedirs(labels_dir, exist_ok=True)
    
    if copy_images and img_dir:
        images_dir = os.path.join(output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
    
    print(f"Reading COCO file: {coco_file}")
    # Read the COCO JSON file
    with open(coco_file, 'r') as f:
        data = json.load(f)
    
    # Extract images, categories, and annotations
    images = {img['id']: img for img in data.get('images', [])}
    categories = {cat['id']: cat for cat in data.get('categories', [])}
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in data.get('annotations', []):
        image_id = ann.get('image_id')
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # Print statistics
    print(f"Found {len(images)} images, {len(categories)} categories, and {len(data.get('annotations', []))} annotations")
    print(f"Categories: {[f'{cat['id']}:{cat['name']}' for cat in categories.values()]}")
    
    # Process each image and its annotations
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    desc = "Converting annotations"
    if verbose:
        pbar = tqdm(total=len(images), desc=desc)
    else:
        pbar = tqdm(images.items(), desc=desc)
        
    for img_id, img_info in (pbar if not verbose else images.items()):
        if verbose:
            pbar.update(1)
            
        file_name = img_info.get('file_name')
        width = img_info.get('width')
        height = img_info.get('height')
        
        # Skip if dimensions are missing
        if not width or not height:
            if verbose:
                print(f"Warning: Skipping {file_name} due to missing dimensions")
            skipped_count += 1
            continue
        
        # Get annotations for this image
        anns = annotations_by_image.get(img_id, [])
        
        try:
            # Create a YOLOv8 format label file (basename.txt)
            base_name = os.path.splitext(file_name)[0]
            label_file = os.path.join(labels_dir, f"{base_name}.txt")
            
            with open(label_file, 'w') as f:
                for ann in anns:
                    # Get category id and bbox
                    cat_id = ann.get('category_id')
                    bbox = ann.get('bbox')  # [x_min, y_min, width, height] in COCO format
                    
                    if not bbox:
                        continue
                    
                    # Convert COCO bbox to YOLO format: [x_center, y_center, width, height] normalized
                    x_min, y_min, w, h = bbox
                    
                    # Ensure bbox coordinates are valid
                    x_min = max(0, min(x_min, width))
                    y_min = max(0, min(y_min, height))
                    w = max(0, min(w, width - x_min))
                    h = max(0, min(h, height - y_min))
                    
                    # Calculate normalized center coordinates and dimensions
                    x_center = (x_min + w / 2) / width
                    y_center = (y_min + h / 2) / height
                    w_norm = w / width
                    h_norm = h / height
                    
                    # Write to file in YOLOv8 format: class_id center_x center_y width height
                    f.write(f"{cat_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
            
            # Copy the image if requested
            if copy_images and img_dir:
                src_img = os.path.join(img_dir, file_name)
                if os.path.exists(src_img):
                    dst_img = os.path.join(images_dir, file_name)
                    shutil.copy2(src_img, dst_img)
                elif verbose:
                    print(f"Warning: Source image not found: {src_img}")
            
            success_count += 1
            
        except Exception as e:
            if verbose:
                print(f"Error processing {file_name}: {e}")
            error_count += 1
    
    elapsed_time = time.time() - start_time
    
    # Print summary
    print(f"\nConversion complete in {elapsed_time:.2f} seconds!")
    print(f"Successfully processed: {success_count} images")
    print(f"Errors: {error_count} images")
    print(f"Skipped: {skipped_count} images")
    print(f"Labels saved to: {labels_dir}")
    
    # Check output directory
    label_files = len(os.listdir(labels_dir))
    print(f"Number of label files created: {label_files}")
    
    if copy_images and img_dir:
        images_count = len(os.listdir(images_dir)) if os.path.exists(images_dir) else 0
        print(f"Images copied to: {images_dir} ({images_count} files)")
    
    return success_count, error_count, skipped_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert COCO format annotations to YOLOv8 format")
    parser.add_argument("--coco", type=str, default="/home/vnptai/duytv/projects/AICITY2025_track4/labeling/data_processing/syn_labels/merged_train.json",
                        help="Path to COCO format JSON file")
    parser.add_argument("--output", type=str, default="/home/vnptai/duytv/projects/AICITY2025_track4/labeling/data_processing/visdrone-train/yolov8",
                        help="Output directory for YOLO format annotations")
    parser.add_argument("--img-dir", type=str, help="Directory containing source images (optional)")
    parser.add_argument("--copy-images", action="store_true", help="Copy images to output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed information")
    
    args = parser.parse_args()
    
    convert_coco_to_yolo(
        coco_file=args.coco,
        output_dir=args.output,
        img_dir=args.img_dir,
        copy_images=args.copy_images,
        verbose=args.verbose
    )