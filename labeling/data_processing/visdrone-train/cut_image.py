#!/usr/bin/env python3
import json
import os
import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

def clip(val, min_v, max_v):
    """Clip value val to range [min_v, max_v]"""
    return max(min(val, max_v), min_v)

def process_single_image_and_annotations(img_path, annotations, output_img_dir):
    """
    Cut image into 2 parts and adjust annotations according to original cut_image.py logic
    
    Args:
        img_path: Path to original image
        annotations: List of annotations for this image
        output_img_dir: Output directory for images
    
    Returns:
        List of tuples (new_image_info, new_annotations, output_path)
    """
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    
    H, W = img.shape[:2]  # OpenCV: height, width
    base_name = os.path.splitext(os.path.basename(img_path))[0]

    # Only process if W > H and W < 2H (same as original logic)
    if not (W > H and W < 2 * H):
        print(f"⚠️ Image {img_path} does not meet W>H and W<2H condition. Skipping.")
        return []

    # Convert COCO bbox [x, y, w, h] to absolute [x1, y1, x2, y2]
    abs_boxes = []
    for ann in annotations:
        bbox = ann['bbox']  # [x, y, width, height]
        x, y, w, h = bbox
        x1, y1, x2, y2 = x, y, x + w, y + h
        abs_boxes.append((ann, x1, y1, x2, y2))

    # --- Prepare two sub-images ---
    # Image 1 (left): crop (0,0) → (H,H)
    img_left = img[0:H, 0:H]  # [y1:y2, x1:x2]
    
    # Image 2 (right): crop (0, W-H) → (H, W)
    img_right = img[0:H, (W-H):W]

    # Lists to hold new annotations for each image
    left_annotations = []
    right_annotations = []

    # Process each original annotation
    for ann, x1, y1, x2, y2 in abs_boxes:
        # Original area
        area_orig = max(0, x2 - x1) * max(0, y2 - y1)
        if area_orig <= 0:
            continue

        # === PROCESSING FOR IMAGE 1 (left, crop x ∈ [0, H]) ===
        # Case 1.A: bbox completely on left (x2 ≤ H) → keep as is
        if x2 <= H:
            new_bbox = [x1, y1, x2 - x1, y2 - y1]
            new_ann = ann.copy()
            new_ann['bbox'] = new_bbox
            new_ann['area'] = (x2 - x1) * (y2 - y1)
            left_annotations.append(new_ann)

        # Case 1.B: bbox crosses right border of image 1 (x1 < H < x2)
        elif x1 < H < x2:
            # Calculate intersection with [0, H] for image 1
            lx1 = clip(x1, 0, H)
            ly1 = clip(y1, 0, H) 
            lx2 = clip(x2, 0, H)
            ly2 = clip(y2, 0, H)
            w_left = max(0, lx2 - lx1)
            h_left = max(0, ly2 - ly1)
            area_left = w_left * h_left

            # If area of this part ≥ 40% of original area → keep and clamp
            if area_left / area_orig >= 0.4:
                new_bbox = [lx1, ly1, w_left, h_left]
                new_ann = ann.copy()
                new_ann['bbox'] = new_bbox
                new_ann['area'] = area_left
                left_annotations.append(new_ann)

        # === PROCESSING FOR IMAGE 2 (right, crop x ∈ [W-H, W]) ===
        # For image 2, we shift origin coordinate: x' = x - (W - H)

        # Case 2.A: bbox completely on right (x1 ≥ W - H) → keep as is (only shift)
        if x1 >= (W - H):
            new_x1 = x1 - (W - H)
            new_x2 = x2 - (W - H)
            new_bbox = [new_x1, y1, new_x2 - new_x1, y2 - y1]
            new_ann = ann.copy()
            new_ann['bbox'] = new_bbox
            new_ann['area'] = (new_x2 - new_x1) * (y2 - y1)
            right_annotations.append(new_ann)

        # Case 2.B: bbox crosses left border of image 2 (x1 < W - H < x2)
        elif x1 < (W - H) < x2:
            # Calculate intersection with [W - H, W] for image 2, then shift to [0, H]
            rx1 = clip(x1 - (W - H), 0, H)
            ry1 = clip(y1, 0, H)
            rx2 = clip(x2 - (W - H), 0, H)
            ry2 = clip(y2, 0, H)
            w_right = max(0, rx2 - rx1)
            h_right = max(0, ry2 - ry1)
            area_right = w_right * h_right

            # If area of this part ≥ 40% of original area → keep and clamp
            if area_right / area_orig >= 0.4:
                new_bbox = [rx1, ry1, w_right, h_right]
                new_ann = ann.copy()
                new_ann['bbox'] = new_bbox
                new_ann['area'] = area_right
                right_annotations.append(new_ann)

    # Create file names and output paths same as original code
    left_name = f"{base_name}_1.png"
    right_name = f"{base_name}_2.png"
    
    left_path = os.path.join(output_img_dir, left_name)
    right_path = os.path.join(output_img_dir, right_name)
    
    # Save images
    cv2.imwrite(left_path, img_left)
    cv2.imwrite(right_path, img_right)

    # Create new image info
    left_image_info = {
        'id': None,  # Will be set by caller
        'file_name': left_name,
        'width': H,
        'height': H
    }
    
    right_image_info = {
        'id': None,  # Will be set by caller  
        'file_name': right_name,
        'width': H,
        'height': H
    }

    return [
        (left_image_info, left_annotations),
        (right_image_info, right_annotations)
    ]

def process_coco_json(input_json, input_img_dir, output_json, output_img_dir):
    """
    Process COCO JSON by cutting images into two parts and adjusting annotations
    """
    print(f"Loading COCO JSON: {input_json}")
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']
    
    print(f"Found {len(images)} images and {len(annotations)} annotations")
    
    # Create output directory
    os.makedirs(output_img_dir, exist_ok=True)
    
    # Group annotations by image_id
    img_id_to_anns = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)
    
    new_images = []
    new_annotations = []
    new_img_id = 1
    new_ann_id = 1
    
    processed_images = 0
    skipped_images = 0
    
    print("Processing images...")
    for img_info in tqdm(images):
        img_id = img_info['id']
        file_name = img_info['file_name']
        
        # Input path
        input_img_path = os.path.join(input_img_dir, file_name)
        
        if not os.path.exists(input_img_path):
            print(f"Warning: Image not found: {input_img_path}")
            skipped_images += 1
            continue
        
        # Get annotations for this image
        img_annotations = img_id_to_anns.get(img_id, [])
        
        try:
            # Process image and annotations
            results = process_single_image_and_annotations(
                input_img_path, img_annotations, output_img_dir
            )
            
            if not results:
                skipped_images += 1
                continue
                
            # Add results to new dataset
            for img_info_new, anns_new in results:
                # Set image ID
                img_info_new['id'] = new_img_id
                new_images.append(img_info_new)
                
                # Add adjusted annotations with new image_id and annotation_id
                for ann in anns_new:
                    ann['id'] = new_ann_id
                    ann['image_id'] = new_img_id
                    new_annotations.append(ann)
                    new_ann_id += 1
                
                new_img_id += 1
                
            processed_images += 1
                
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            skipped_images += 1
            continue
    
    # Create output data
    output_data = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': categories
    }
    
    print(f"Saving processed COCO JSON: {output_json}")
    with open(output_json, 'w') as f:
        json.dump(output_data, f)
    print(f"  - Total input images: {len(images)}")
    print(f"  - Successfully processed: {processed_images}")  
    print(f"  - Skipped: {skipped_images}")
    print(f"  - Output images: {len(new_images)}")
    print(f"  - Output annotations: {len(new_annotations)}")
    
    return output_data

def main():
    parser = argparse.ArgumentParser(description='Cut images into two parts and adjust COCO annotations')
    parser.add_argument('--input-json', required=True, help='Input COCO JSON file')
    parser.add_argument('--input-img-dir', required=True, help='Input image directory')
    parser.add_argument('--output-json', required=True, help='Output COCO JSON file')
    parser.add_argument('--output-img-dir', required=True, help='Output image directory')
    
    args = parser.parse_args()
    
    process_coco_json(
        args.input_json,
        args.input_img_dir, 
        args.output_json,
        args.output_img_dir
    )

if __name__ == "__main__":
    main()