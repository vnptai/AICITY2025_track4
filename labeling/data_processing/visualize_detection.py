#!/usr/bin/env python3
import json
import os
import cv2
import numpy as np
import argparse
import random
from pathlib import Path
from tqdm import tqdm

# Define colors for different classes (BGR format for OpenCV)
COLORS = [
    (255, 0, 0),      # Blue
    (0, 255, 0),      # Green  
    (0, 0, 255),      # Red
    (255, 255, 0),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Yellow
    (128, 0, 128),    # Purple
    (255, 165, 0),    # Orange
    (0, 128, 255),    # Light Blue
    (255, 192, 203)   # Pink
]

def get_color(class_id):
    """Get color for a specific class"""
    return COLORS[class_id % len(COLORS)]

def draw_bbox(img, bbox, class_id, class_name, confidence=None, color=None):
    """
    Draw bounding box on image
    
    Args:
        img: OpenCV image
        bbox: [x, y, width, height] in COCO format
        class_id: Class ID
        class_name: Class name string
        confidence: Optional confidence score
        color: Optional color override
    """
    if color is None:
        color = get_color(class_id)
    
    x, y, w, h = bbox
    x1, y1 = int(x), int(y)
    x2, y2 = int(x + w), int(y + h)
    
    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    
    # Prepare label text
    if confidence is not None:
        label = f"{class_name} ({confidence:.2f})"
    else:
        label = f"{class_name}"
    
    # Calculate text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    
    # Draw background for text
    cv2.rectangle(img, (x1, y1 - text_h - baseline - 5), (x1 + text_w, y1), color, -1)
    
    # Draw text
    cv2.putText(img, label, (x1, y1 - baseline - 2), font, font_scale, (255, 255, 255), thickness)

def load_coco_data(json_file):
    """Load COCO format JSON file"""
    print(f"Loading COCO data from: {json_file}")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    images = data.get('images', [])
    annotations = data.get('annotations', [])
    categories = data.get('categories', [])
    
    print(f"Found {len(images)} images, {len(annotations)} annotations, {len(categories)} categories")
    
    # Create mappings
    img_id_to_info = {img['id']: img for img in images}
    cat_id_to_name = {cat['id']: cat['name'] for cat in categories}
    
    # Group annotations by image_id
    img_id_to_anns = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)
    
    return img_id_to_info, img_id_to_anns, cat_id_to_name

def visualize_sample_images(img_dir, json_file, output_dir, num_samples=20, save_individual=True):
    """
    Visualize detection results on sample images
    
    Args:
        img_dir: Directory containing images
        json_file: COCO format JSON file
        output_dir: Directory to save visualization results
        num_samples: Number of sample images to visualize
        save_individual: Whether to save individual images
    """
    # Load COCO data
    img_id_to_info, img_id_to_anns, cat_id_to_name = load_coco_data(json_file)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of images with annotations
    images_with_anns = [img_id for img_id, anns in img_id_to_anns.items() if len(anns) > 0]
    
    if not images_with_anns:
        print("No images with annotations found!")
        return
    
    # Sample images
    sample_img_ids = random.sample(images_with_anns, min(num_samples, len(images_with_anns)))
    
    print(f"Visualizing {len(sample_img_ids)} sample images...")
    
    # Statistics
    total_annotations = 0
    category_counts = {}
    
    for img_id in tqdm(sample_img_ids):
        img_info = img_id_to_info[img_id]
        img_filename = img_info['file_name']
        img_path = os.path.join(img_dir, img_filename)
        
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load image: {img_path}")
            continue
        
        # Get annotations for this image
        annotations = img_id_to_anns.get(img_id, [])
        total_annotations += len(annotations)
        
        # Draw bounding boxes
        for ann in annotations:
            bbox = ann['bbox']
            class_id = ann['category_id']
            class_name = cat_id_to_name.get(class_id, f"Class_{class_id}")
            confidence = ann.get('score')  # If available from detection results
            
            # Count categories
            category_counts[class_name] = category_counts.get(class_name, 0) + 1
            
            # Draw bbox
            draw_bbox(img, bbox, class_id, class_name, confidence)
        
        # Add image info
        info_text = f"Image: {img_filename} | Annotations: {len(annotations)}"
        cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        if save_individual:
            # Save individual image
            base_name = os.path.splitext(img_filename)[0]
            output_path = os.path.join(output_dir, f"vis_{base_name}.jpg")
            cv2.imwrite(output_path, img)
    
    # Print statistics
    print(f"\n=== Visualization Statistics ===")
    print(f"Total images visualized: {len(sample_img_ids)}")
    print(f"Total annotations: {total_annotations}")
    print(f"Average annotations per image: {total_annotations/len(sample_img_ids):.1f}")
    
    print(f"\n=== Category Distribution ===")
    for cat_name, count in sorted(category_counts.items()):
        print(f"{cat_name}: {count} annotations")
    
    # Create summary visualization
    create_category_legend(output_dir, cat_id_to_name, category_counts)

def create_category_legend(output_dir, cat_id_to_name, category_counts):
    """Create a legend showing categories and their colors"""
    img_height = 50 + len(cat_id_to_name) * 40
    img_width = 400
    legend_img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    
    # Title
    cv2.putText(legend_img, "Category Legend", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Draw categories
    y_offset = 60
    for cat_id, cat_name in cat_id_to_name.items():
        color = get_color(cat_id)
        count = category_counts.get(cat_name, 0)
        
        # Draw color box
        cv2.rectangle(legend_img, (10, y_offset - 15), (35, y_offset + 5), color, -1)
        cv2.rectangle(legend_img, (10, y_offset - 15), (35, y_offset + 5), (0, 0, 0), 1)
        
        # Draw text
        text = f"{cat_name} ({count} objects)"
        cv2.putText(legend_img, text, (45, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        y_offset += 30
    
    # Save legend
    legend_path = os.path.join(output_dir, "category_legend.jpg")
    cv2.imwrite(legend_path, legend_img)
    print(f"Category legend saved to: {legend_path}")

def create_grid_visualization(img_dir, json_file, output_dir, grid_size=(4, 4)):
    """
    Create a grid visualization with multiple images
    """
    # Load COCO data
    img_id_to_info, img_id_to_anns, cat_id_to_name = load_coco_data(json_file)
    
    # Get sample images
    images_with_anns = [img_id for img_id, anns in img_id_to_anns.items() if len(anns) > 0]
    num_samples = grid_size[0] * grid_size[1]
    sample_img_ids = random.sample(images_with_anns, min(num_samples, len(images_with_anns)))
    
    # Target size for each cell in the grid
    cell_width, cell_height = 300, 300
    
    # Create grid image
    grid_width = grid_size[1] * cell_width
    grid_height = grid_size[0] * cell_height
    grid_img = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    print(f"Creating {grid_size[0]}x{grid_size[1]} grid visualization...")
    
    for idx, img_id in enumerate(sample_img_ids):
        if idx >= num_samples:
            break
            
        row = idx // grid_size[1]
        col = idx % grid_size[1]
        
        img_info = img_id_to_info[img_id]
        img_filename = img_info['file_name']
        img_path = os.path.join(img_dir, img_filename)
        
        if not os.path.exists(img_path):
            continue
        
        # Load and resize image
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        img_resized = cv2.resize(img, (cell_width, cell_height))
        
        # Get annotations and scale them
        annotations = img_id_to_anns.get(img_id, [])
        scale_x = cell_width / img_info['width']
        scale_y = cell_height / img_info['height']
        
        # Draw bounding boxes
        for ann in annotations:
            bbox = ann['bbox']
            class_id = ann['category_id']
            class_name = cat_id_to_name.get(class_id, f"Class_{class_id}")
            
            # Scale bbox
            x, y, w, h = bbox
            scaled_bbox = [x * scale_x, y * scale_y, w * scale_x, h * scale_y]
            
            draw_bbox(img_resized, scaled_bbox, class_id, class_name)
        
        # Add to grid
        y_start = row * cell_height
        y_end = y_start + cell_height
        x_start = col * cell_width
        x_end = x_start + cell_width
        
        grid_img[y_start:y_end, x_start:x_end] = img_resized
    
    # Save grid
    os.makedirs(output_dir, exist_ok=True)
    grid_path = os.path.join(output_dir, f"detection_grid_{grid_size[0]}x{grid_size[1]}.jpg")
    cv2.imwrite(grid_path, grid_img)
    print(f"Grid visualization saved to: {grid_path}")

def main():
    parser = argparse.ArgumentParser(description='Visualize detection results from COCO JSON')
    parser.add_argument('--img-dir', required=True, help='Directory containing images')
    parser.add_argument('--json-file', required=True, help='COCO format JSON file')
    parser.add_argument('--output-dir', default='./visualization_results', help='Output directory for visualization')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of sample images to visualize')
    parser.add_argument('--grid-size', type=int, nargs=2, default=[2, 2], help='Grid size for grid visualization (rows cols)')
    parser.add_argument('--mode', choices=['individual', 'grid', 'both'], default='both', 
                        help='Visualization mode: individual images, grid, or both')
    
    args = parser.parse_args()
    
    # Check inputs
    if not os.path.exists(args.img_dir):
        print(f"Error: Image directory not found: {args.img_dir}")
        return
        
    if not os.path.exists(args.json_file):
        print(f"Error: JSON file not found: {args.json_file}")
        return
    
    print(f"=== Detection Visualization ===")
    print(f"Image directory: {args.img_dir}")
    print(f"JSON file: {args.json_file}")
    print(f"Output directory: {args.output_dir}")
    
    if args.mode in ['individual', 'both']:
        print(f"\n--- Creating Individual Visualizations ---")
        visualize_sample_images(
            args.img_dir, 
            args.json_file, 
            os.path.join(args.output_dir, 'individual'),
            args.num_samples
        )
    
    if args.mode in ['grid', 'both']:
        print(f"\n--- Creating Grid Visualization ---")
        create_grid_visualization(
            args.img_dir,
            args.json_file,
            os.path.join(args.output_dir, 'grid'),
            tuple(args.grid_size)
        )
    
    print(f"\n✅ Visualization complete! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
