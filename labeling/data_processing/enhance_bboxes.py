import os
import argparse
import glob
import shutil
import random
from pathlib import Path
import cv2
import numpy as np

def parse_yolo_file(file_path):
    """Parse YOLO format txt file and return list of bboxes [class_id, x, y, w, h]"""
    bboxes = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # YOLO format: class_id x_center y_center width height
                bboxes.append([int(parts[0]), float(parts[1]), float(parts[2]), 
                              float(parts[3]), float(parts[4])])
    return bboxes

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes in YOLO format
    box format: [class_id, x_center, y_center, width, height]
    """
    # Convert from YOLO format (center x, center y, width, height) to (x1, y1, x2, y2)
    b1_x1 = box1[1] - box1[3] / 2
    b1_y1 = box1[2] - box1[4] / 2
    b1_x2 = box1[1] + box1[3] / 2
    b1_y2 = box1[2] + box1[4] / 2
    
    b2_x1 = box2[1] - box2[3] / 2
    b2_y1 = box2[2] - box2[4] / 2
    b2_x2 = box2[1] + box2[3] / 2
    b2_y2 = box2[2] + box2[4] / 2
    
    # Get the coordinates of the intersection rectangle
    x_left = max(b1_x1, b2_x1)
    y_top = max(b1_y1, b2_y1)
    x_right = min(b1_x2, b2_x2)
    y_bottom = min(b1_y2, b2_y2)
    
    # No intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = b1_area + b2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return iou

def should_add_bbox(pred_box, gt_boxes):
    """Determine if prediction bbox should be added to groundtruth
    
    For each predict bbox, we check if it overlaps significantly with any groundtruth bbox.
    - If it overlaps with any groundtruth bbox with IoU > 0.5, discard it
    - Otherwise, add it
    """
    # If no groundtruth boxes, we should add this box
    if not gt_boxes:
        return True
    
    # Check overlap with all groundtruth boxes
    for gt_box in gt_boxes:
        iou = calculate_iou(pred_box, gt_box)
        if iou > 0.5:
            return False
    
    # If no significant overlap with any groundtruth box, add it
    return True

def process_files(gt_dir, pred_dir, output_dir, img_dir=None):
    """Process prediction files and update groundtruth files in a new directory"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all prediction files
    pred_files = glob.glob(os.path.join(pred_dir, "*.txt"))
    
    # Get all ground truth files
    gt_files = glob.glob(os.path.join(gt_dir, "*.txt"))
    
    added_boxes_count = 0
    processed_files_count = 0
    modified_files = []
    
    # Process files that have predictions
    for pred_file in pred_files:
        # Get corresponding groundtruth file name
        file_name = os.path.basename(pred_file)
        gt_file = os.path.join(gt_dir, file_name)
        output_file = os.path.join(output_dir, file_name)
        
        # Skip if groundtruth file doesn't exist
        if not os.path.exists(gt_file):
            continue
        
        # Parse files
        pred_boxes = parse_yolo_file(pred_file)
        gt_boxes = parse_yolo_file(gt_file)
        
        # Check each prediction box
        boxes_to_add = []
        for pred_box in pred_boxes:
            if should_add_bbox(pred_box, gt_boxes):
                boxes_to_add.append(pred_box)
        
        # Create new output file
        if boxes_to_add:
            # Copy original groundtruth content and append new boxes
            with open(gt_file, 'r') as src, open(output_file, 'w') as dest:
                dest.write(src.read())
                for box in boxes_to_add:
                    box_str = f"{box[0]} {box[1]} {box[2]} {box[3]} {box[4]}\n"
                    dest.write(box_str)
            
            added_boxes_count += len(boxes_to_add)
            modified_files.append((file_name, gt_boxes, boxes_to_add))
        else:
            # Just copy the original file
            shutil.copy2(gt_file, output_file)
        
        processed_files_count += 1
    
    # Process ground truth files that don't have corresponding prediction files
    for gt_file in gt_files:
        file_name = os.path.basename(gt_file)
        pred_file = os.path.join(pred_dir, file_name)
        output_file = os.path.join(output_dir, file_name)
        
        # If prediction file doesn't exist and output file doesn't exist yet
        if not os.path.exists(pred_file) and not os.path.exists(output_file):
            # Copy the ground truth file directly
            shutil.copy2(gt_file, output_file)
            processed_files_count += 1
    
    return processed_files_count, added_boxes_count, modified_files

def visualize_samples(img_dir, gt_dir, output_dir, vis_dir, modified_files, num_samples=20):
    """Visualize samples of before and after adding bounding boxes"""
    
    # Create visualization directory if it doesn't exist
    os.makedirs(vis_dir, exist_ok=True)
    
    # Select random samples from modified files
    samples = random.sample(modified_files, min(num_samples, len(modified_files)))
    
    for idx, (file_name, original_boxes, added_boxes) in enumerate(samples):
        # Get image file path (assuming same name with different extension)
        img_base = os.path.splitext(file_name)[0]
        
        # Try common image extensions
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            potential_path = os.path.join(img_dir, img_base + ext)
            if os.path.exists(potential_path):
                img_path = potential_path
                break
        
        if img_path is None:
            print(f"Warning: Image file for {file_name} not found, skipping visualization")
            continue
        
        # Read the image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}, skipping visualization")
            continue
        
        height, width = img.shape[:2]
        
        # Create copies for before and after
        img_before = img.copy()
        img_after = img.copy()
        
        # Color for original ground truth: green
        gt_color = (0, 255, 0)
        # Color for new added boxes: red
        new_color = (0, 0, 255)
        
        # Draw original (ground truth) boxes on both images
        for box in original_boxes:
            class_id, x_center, y_center, w, h = box
            x1 = int((x_center - w/2) * width)
            y1 = int((y_center - h/2) * height)
            x2 = int((x_center + w/2) * width)
            y2 = int((y_center + h/2) * height)
            
            cv2.rectangle(img_before, (x1, y1), (x2, y2), gt_color, 1)
            cv2.rectangle(img_after,  (x1, y1), (x2, y2), gt_color, 1)
            
            # Add class label
            cv2.putText(img_before, f"{class_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, gt_color, 1)
            cv2.putText(img_after,  f"{class_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, gt_color, 2)
        
        # Draw added boxes only on the "after" image
        for box in added_boxes:
            class_id, x_center, y_center, w, h = box
            x1 = int((x_center - w/2) * width)
            y1 = int((y_center - h/2) * height)
            x2 = int((x_center + w/2) * width)
            y2 = int((y_center + h/2) * height)
            
            cv2.rectangle(img_after, (x1, y1), (x2, y2), new_color, 1)
            cv2.putText(img_after, f"{class_id}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, new_color, 2)
        
        # Create side-by-side comparison
        combined = np.hstack((img_before, img_after))
        
        # Add titles
        cv2.putText(combined, "Before", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.putText(combined, "After", (width + 10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        
        # Save visualization
        vis_path = os.path.join(vis_dir, f"sample_{idx+1}_{img_base}.jpg")
        cv2.imwrite(vis_path, combined)
    
    print(f"Created {len(samples)} visualization samples in {vis_dir}")


def main():
    parser = argparse.ArgumentParser(description='Merge prediction bboxes into groundtruth files')
    parser.add_argument('--gt-dir', default='fisheye_labels', help='Directory containing groundtruth txt files')
    parser.add_argument('--pred-dir', default='predicted_fisheye', help='Directory containing prediction txt files')
    parser.add_argument('--output-dir', default='synthesis_fisheye_labels', help='Directory to save new groundtruth files')
    parser.add_argument('--img-dir', default='path_to_image_directory', help='Directory containing image files (for visualization)')
    parser.add_argument('--vis-dir', default='visualized', help='Directory to save visualization samples')
    parser.add_argument('--num-vis', type=int, default=100, help='Number of samples to visualize (default: 20)')
    args = parser.parse_args()
    
    # Ensure directories exist
    if not os.path.isdir(args.gt_dir):
        print(f"Error: Groundtruth directory '{args.gt_dir}' does not exist")
        return
    
    if not os.path.isdir(args.pred_dir):
        print(f"Error: Prediction directory '{args.pred_dir}' does not exist")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    processed_files, added_boxes, modified_files = process_files(
        args.gt_dir, args.pred_dir, args.output_dir, args.img_dir
    )
    
    print(f"Processed {processed_files} files")
    print(f"Added {added_boxes} new bounding boxes to groundtruth files")
    print(f"Modified {len(modified_files)} files")
    print(f"All files saved to {args.output_dir}")
    
    # Visualize samples if requested
    if args.vis_dir and args.img_dir:
        visualize_samples(
            args.img_dir, args.gt_dir, args.output_dir, 
            args.vis_dir, modified_files, args.num_vis
        )
    elif args.vis_dir and not args.img_dir:
        print("Warning: Image directory (--img-dir) not provided, skipping visualization")
    
    print("20s")

if __name__ == "__main__":
    main() 