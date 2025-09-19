#!/usr/bin/env python3
import json
import os
import cv2
import numpy as np
import math
import argparse
from pathlib import Path
from tqdm import tqdm

def rect_to_fisheye_point(x_src, y_src, S, f_fisheye):
    """
    Convert a point (x_src, y_src) on a square rectilinear image of size S×S
    to fisheye coordinates
    """
    C = S / 2.0
    f_rect = S / 2.0
    
    # Coordinate relative to the center of the rectilinear image
    dx = x_src - C
    dy = y_src - C
    # In the rectilinear system, the camera vector is not normalized:
    X_cam = dx
    Y_cam = dy
    Z_cam = f_rect
    
    # Normalize to unit vector (Xn, Yn, Zn)
    norm = math.sqrt(X_cam**2 + Y_cam**2 + Z_cam**2)
    if norm == 0:
        Xn = 0.0
        Yn = 0.0
        Zn = 1.0
    else:
        Xn = X_cam / norm
        Yn = Y_cam / norm
        Zn = Z_cam / norm
    
    # Calculate theta = arccos(Zn)
    theta = math.acos(max(-1.0, min(1.0, Zn)))
    # Calculate phi (azimuth angle)
    phi = math.atan2(Yn, Xn)
    
    # Theta_max = angle between the vector to the origin image and the optical axis
    theta_max = math.acos(1.0 / math.sqrt(3.0))  # ≈0.95532 rad
    if theta > theta_max:
        theta = theta_max
    
    # Equidistant: r_out = f_fisheye * theta (distance from the center of the fisheye image)
    r_out = f_fisheye * theta
    
    # Calculate (x_out, y_out) relative to the center of the fisheye image
    x_out = C + r_out * math.cos(phi)
    y_out = C + r_out * math.sin(phi)
    
    return x_out, y_out

def convert_bbox_to_fisheye(bbox, img_width, img_height, k=1.0):
    """
    Convert a COCO bbox to fisheye format
    
    Args:
        bbox: COCO bbox [x, y, width, height]
        img_width, img_height: Original image dimensions
        k: Fisheye scale factor
    
    Returns:
        Transformed bbox in fisheye coordinates
    """
    # Resize to square image S×S, with S = min(h_orig, w_orig)
    S = min(img_height, img_width)
    
    # Calculate scale factors
    scale_x = S / img_width
    scale_y = S / img_height
    
    # Scale bbox to square image
    x, y, w, h = bbox
    x_sq = x * scale_x
    y_sq = y * scale_y
    w_sq = w * scale_x
    h_sq = h * scale_y
    
    # Calculate fisheye focal length
    theta_max = math.acos(1.0 / math.sqrt(3.0))
    R_out = S / 2.0
    f_fisheye = (R_out / theta_max) * k
    
    # Calculate 4 corners of the bbox
    corners = [
        (x_sq, y_sq),                    # top-left
        (x_sq, y_sq + h_sq),            # bottom-left
        (x_sq + w_sq, y_sq),            # top-right
        (x_sq + w_sq, y_sq + h_sq),     # bottom-right
    ]
    
    # Map each corner to fisheye
    fisheye_corners = [rect_to_fisheye_point(x, y, S, f_fisheye) for (x, y) in corners]
    xs_f = [pt[0] for pt in fisheye_corners]
    ys_f = [pt[1] for pt in fisheye_corners]
    
    # Calculate bbox on fisheye
    x_min_f = min(xs_f)
    x_max_f = max(xs_f)
    y_min_f = min(ys_f)
    y_max_f = max(ys_f)
    
    # Clip to [0..S]
    x_min_f = max(0.0, min(S, x_min_f))
    x_max_f = max(0.0, min(S, x_max_f))
    y_min_f = max(0.0, min(S, y_min_f))
    y_max_f = max(0.0, min(S, y_max_f))
    
    # Convert to COCO format
    w_f = x_max_f - x_min_f
    h_f = y_max_f - y_min_f
    
    return [x_min_f, y_min_f, w_f, h_f], S

def crop_and_adjust_bbox(bbox, S, crop_ratio=0.827):
    """
    Adjust bbox after cropping the fisheye image
    """
    x, y, w, h = bbox
    
    # Calculate new image size and crop offset
    new_size = int(S * crop_ratio)
    start = (S - new_size) // 2
    
    # Adjust coordinates due to crop
    x_new = x - start
    y_new = y - start
    
    # Calculate the boundary points
    x_min_new = max(0, x_new)
    x_max_new = min(new_size, x_new + w)
    y_min_new = max(0, y_new)
    y_max_new = min(new_size, y_new + h)
    
    # Check if box still exists after cropping
    if x_min_new >= x_max_new or y_min_new >= y_max_new:
        return None
    
    # Calculate new dimensions
    w_new = x_max_new - x_min_new
    h_new = y_max_new - y_min_new
    
    return [x_min_new, y_min_new, w_new, h_new]

def create_fisheye_image(img, f_fisheye, interp=cv2.INTER_CUBIC):
    """Create fisheye image from square input image"""
    S = img.shape[0]
    C = S / 2.0
    
    # Create map_x, map_y
    map_x = np.zeros((S, S), dtype=np.float32)
    map_y = np.zeros((S, S), dtype=np.float32)
    
    theta_max = math.acos(1.0 / math.sqrt(3.0))
    
    for y_out in range(S):
        for x_out in range(S):
            dx = x_out - C
            dy = y_out - C
            r_out = math.hypot(dx, dy)
            
            # Calculate theta corresponding
            theta = r_out / f_fisheye
            if theta > theta_max:
                map_x[y_out, x_out] = -1
                map_y[y_out, x_out] = -1
                continue
            
            # Calculate phi
            if r_out == 0:
                phi = 0.0
            else:
                phi = math.atan2(dy, dx)
            
            # Calculate 3D vector on the sphere
            sin_t = math.sin(theta)
            X_cam = sin_t * math.cos(phi)
            Y_cam = sin_t * math.sin(phi)
            Z_cam = math.cos(theta)
            
            # Project back to the rectilinear image
            if abs(Z_cam) < 1e-6:
                map_x[y_out, x_out] = -1
                map_y[y_out, x_out] = -1
            else:
                x_src = (X_cam / Z_cam) * (S/2.0) + C
                y_src = (Y_cam / Z_cam) * (S/2.0) + C
                # Check if outside the boundary
                if x_src < 0 or x_src >= (S-1) or y_src < 0 or y_src >= (S-1):
                    map_x[y_out, x_out] = -1
                    map_y[y_out, x_out] = -1
                else:
                    map_x[y_out, x_out] = x_src
                    map_y[y_out, x_out] = y_src
    
    dst_fisheye = cv2.remap(
        img,
        map_x, map_y,
        interpolation=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    return dst_fisheye

def process_image_and_annotations(img_path, annotations, output_img_path, k=1.0, crop_ratio=0.827):
    """
    Process a single image with fisheye transformation and adjust annotations
    """
    # Read original image
    img_orig = cv2.imread(img_path)
    if img_orig is None:
        raise ValueError(f"Could not read image: {img_path}")
    
    h_orig, w_orig = img_orig.shape[:2]
    
    # Resize to square
    S = min(h_orig, w_orig)
    img_square = cv2.resize(img_orig, (S, S))
    
    # Calculate fisheye focal length
    theta_max = math.acos(1.0 / math.sqrt(3.0))
    R_out = S / 2.0
    f_fisheye = (R_out / theta_max) * k
    
    # Create fisheye image
    fisheye_img = create_fisheye_image(img_square, f_fisheye)
    
    # Crop fisheye image
    new_size = int(S * crop_ratio)
    start = (S - new_size) // 2
    end = start + new_size
    fisheye_cropped = fisheye_img[start:end, start:end]
    
    # Save cropped fisheye image
    cv2.imwrite(output_img_path, fisheye_cropped)
    
    # Transform annotations
    transformed_annotations = []
    for ann in annotations:
        try:
            # Convert bbox to fisheye
            fisheye_bbox, _ = convert_bbox_to_fisheye(ann['bbox'], w_orig, h_orig, k)
            
            # Adjust for cropping
            cropped_bbox = crop_and_adjust_bbox(fisheye_bbox, S, crop_ratio)
            
            if cropped_bbox is not None:
                new_ann = ann.copy()
                new_ann['bbox'] = cropped_bbox
                new_ann['area'] = cropped_bbox[2] * cropped_bbox[3]
                transformed_annotations.append(new_ann)
                
        except Exception as e:
            print(f"Error transforming annotation: {e}")
            continue
    
    # Create new image info
    new_image_info = {
        'id': None,  # Will be set by caller
        'file_name': os.path.basename(output_img_path),
        'width': new_size,
        'height': new_size
    }
    
    return new_image_info, transformed_annotations

def process_coco_json(input_json, input_img_dir, output_json, output_img_dir, k=1.0, crop_ratio=0.827):
    """
    Process COCO JSON with fisheye augmentation
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
    new_ann_id = 1
    
    print("Processing images with fisheye augmentation...")
    for img_info in tqdm(images):
        img_id = img_info['id']
        file_name = img_info['file_name']
        
        # Input and output paths
        input_img_path = os.path.join(input_img_dir, file_name)
        output_img_path = os.path.join(output_img_dir, file_name)
        
        if not os.path.exists(input_img_path):
            print(f"Warning: Image not found: {input_img_path}")
            continue
        
        # Get annotations for this image
        img_annotations = img_id_to_anns.get(img_id, [])
        
        try:
            # Process image with fisheye and adjust annotations
            new_img_info, adj_annotations = process_image_and_annotations(
                input_img_path, img_annotations, output_img_path, k, crop_ratio
            )
            
            # Set image id
            new_img_info['id'] = img_id
            new_images.append(new_img_info)
            
            # Add adjusted annotations
            for ann in adj_annotations:
                ann['id'] = new_ann_id
                new_annotations.append(ann)
                new_ann_id += 1
                
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue
    
    # Create output data
    output_data = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': categories
    }
    
    print(f"Saving fisheye augmented COCO JSON: {output_json}")
    with open(output_json, 'w') as f:
        json.dump(output_data, f)
    
    print(f"Processed {len(new_images)} images with {len(new_annotations)} annotations")
    
    return output_data

def main():
    parser = argparse.ArgumentParser(description='Apply fisheye augmentation to COCO dataset')
    parser.add_argument('--input-json', required=True, help='Input COCO JSON file')
    parser.add_argument('--input-img-dir', required=True, help='Input image directory')
    parser.add_argument('--output-json', required=True, help='Output COCO JSON file')
    parser.add_argument('--output-img-dir', required=True, help='Output image directory')
    parser.add_argument('--k', type=float, default=1.0, help='Fisheye scale factor')
    parser.add_argument('--crop-ratio', type=float, default=0.827, help='Crop ratio for fisheye images')
    
    args = parser.parse_args()
    
    process_coco_json(
        args.input_json,
        args.input_img_dir,
        args.output_json,
        args.output_img_dir,
        args.k,
        args.crop_ratio
    )

if __name__ == "__main__":
    main()
