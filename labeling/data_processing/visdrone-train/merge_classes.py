#!/usr/bin/env python3
import json
import os
import argparse
from pathlib import Path

def compute_iou(boxA, boxB):
    """Compute IoU between two boxes in [x, y, width, height] format"""
    # Convert to [x1, y1, x2, y2]
    x1A, y1A, x2A, y2A = boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3]
    x1B, y1B, x2B, y2B = boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3]
    
    xa = max(x1A, x1B); ya = max(y1A, y1B)
    xb = min(x2A, x2B); yb = min(y2A, y2B)
    inter_w = max(0, xb - xa); inter_h = max(0, yb - ya)
    inter = inter_w * inter_h
    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]
    union = areaA + areaB - inter
    return inter/union if union > 0 else 0

def compute_overlap_ratio(boxA, boxB):
    """Return overlap ratio of A with respect to area of A"""
    # Convert to [x1, y1, x2, y2]
    x1A, y1A, x2A, y2A = boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3]
    x1B, y1B, x2B, y2B = boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3]
    
    xa = max(x1A, x1B); ya = max(y1A, y1B)
    xb = min(x2A, x2B); yb = min(y2A, y2B)
    inter_w = max(0, xb - xa); inter_h = max(0, yb - ya)
    inter = inter_w * inter_h
    areaA = max(0, boxA[2] * boxA[3])
    return (inter / areaA) if areaA > 0 else 0.0

def union_boxes(boxA, boxB):
    """Return union of two boxes in [x, y, width, height] format"""
    x1A, y1A, x2A, y2A = boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3]
    x1B, y1B, x2B, y2B = boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3]
    
    x1 = min(x1A, x1B)
    y1 = min(y1A, y1B)
    x2 = max(x2A, x2B)
    y2 = max(y2A, y2B)
    
    return [x1, y1, x2 - x1, y2 - y1]

def remap_class(c):
    """Remap original VisDrone classes to target classes"""
    mapping = {
        0: 3,  # pedestrian -> Pedestrian
        1: 1,  # people -> Bike (this will be handled by merging logic)
        2: 1,  # bicycle -> Bike  
        3: 2,  # car -> Car
        4: 2,  # van -> Car
        5: 4,  # truck -> Truck
        # 6, 7 removed (tricycle, awning-tricycle)
        8: 0,  # bus -> Bus
        9: 1   # motor -> Bike
    }
    return mapping.get(c)

def create_target_categories():
    """Create target categories for the merged dataset"""
    return [
        {"id": 0, "name": "Bus", "supercategory": "vehicle"},
        {"id": 1, "name": "Bike", "supercategory": "vehicle"}, 
        {"id": 2, "name": "Car", "supercategory": "vehicle"},
        {"id": 3, "name": "Pedestrian", "supercategory": "person"},
        {"id": 4, "name": "Truck", "supercategory": "vehicle"}
    ]

def process_annotations_for_image(anns):
    """
    Process annotations for a single image following the merge logic
    """
    # Track which annotations have been used
    used = set()
    merged_anns = []
    new_ann_id_start = max([ann['id'] for ann in anns], default=0) + 1
    
    # Step 1: Handle person (class 1) merging
    person_anns = [ann for ann in anns if ann['category_id'] == 1]
    
    for person_ann in person_anns:
        if person_ann['id'] in used:
            continue
            
        # Check if person overlaps with classes 6 or 7 (should be removed)
        skip_person = False
        for other_ann in anns:
            if other_ann['category_id'] in {6, 7}:
                if compute_iou(person_ann['bbox'], other_ann['bbox']) > 0.0:
                    used.add(person_ann['id'])
                    skip_person = True
                    break
        
        if skip_person:
            continue
            
        # Check overlap with existing merged entries
        best_merge_idx = None
        best_overlap = 0.0
        for idx, merged_ann in enumerate(merged_anns):
            if merged_ann['category_id'] == 1:  # Only merge with other persons
                overlap_ratio = compute_overlap_ratio(person_ann['bbox'], merged_ann['bbox'])
                if overlap_ratio >= 0.2 and overlap_ratio > best_overlap:
                    best_overlap = overlap_ratio
                    best_merge_idx = idx
        
        if best_merge_idx is not None:
            # Merge with existing person
            merged_anns[best_merge_idx]['bbox'] = union_boxes(
                person_ann['bbox'], merged_anns[best_merge_idx]['bbox']
            )
            merged_anns[best_merge_idx]['area'] = (
                merged_anns[best_merge_idx]['bbox'][2] * merged_anns[best_merge_idx]['bbox'][3]
            )
            used.add(person_ann['id'])
            continue
            
        # Check merge with vehicles (classes 2, 9)
        best_vehicle = None
        best_iou = 0.0
        for other_ann in anns:
            if (other_ann['id'] not in used and 
                other_ann['category_id'] in {2, 9}):
                iou = compute_iou(person_ann['bbox'], other_ann['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_vehicle = other_ann
        
        if best_vehicle is not None and best_iou > 0.0:
            # Create merged annotation
            union_bbox = union_boxes(person_ann['bbox'], best_vehicle['bbox'])
            merged_ann = {
                'id': new_ann_id_start,
                'image_id': person_ann['image_id'],
                'category_id': 1,  # Keep as person/bike
                'bbox': union_bbox,
                'area': union_bbox[2] * union_bbox[3],
                'segmentation': [],
                'iscrowd': 0
            }
            merged_anns.append(merged_ann)
            used.add(person_ann['id'])
            used.add(best_vehicle['id'])
            new_ann_id_start += 1
        else:
            # Keep person as is but mark as used (effectively removing it if no overlap)
            used.add(person_ann['id'])
    
    # Step 2: Handle class 0 merging with vehicles
    zero_anns = [ann for ann in anns if ann['category_id'] == 0]
    
    for zero_ann in zero_anns:
        if zero_ann['id'] in used:
            continue
            
        merged_flag = False
        for other_ann in anns:
            if (other_ann['id'] not in used and 
                other_ann['category_id'] in {2, 9}):
                if compute_iou(zero_ann['bbox'], other_ann['bbox']) >= 0.1:
                    # Create merged annotation
                    union_bbox = union_boxes(zero_ann['bbox'], other_ann['bbox'])
                    merged_ann = {
                        'id': new_ann_id_start,
                        'image_id': zero_ann['image_id'],
                        'category_id': 3,  # Pedestrian merged with vehicle -> Pedestrian
                        'bbox': union_bbox,
                        'area': union_bbox[2] * union_bbox[3],
                        'segmentation': [],
                        'iscrowd': 0
                    }
                    merged_anns.append(merged_ann)
                    used.add(zero_ann['id'])
                    used.add(other_ann['id'])
                    new_ann_id_start += 1
                    merged_flag = True
                    break
        
        if not merged_flag:
            # Remap class 0 (pedestrian) to target class 3 (Pedestrian)
            new_ann = zero_ann.copy()
            new_ann['id'] = new_ann_id_start
            new_ann['category_id'] = 3  # Map to Pedestrian in target categories
            merged_anns.append(new_ann)
            used.add(zero_ann['id'])
            new_ann_id_start += 1
    
    # Step 3: Add remaining annotations with class remapping
    for ann in anns:
        if ann['id'] not in used and ann['category_id'] not in {6, 7}:
            new_cat_id = remap_class(ann['category_id'])
            if new_cat_id is not None:
                new_ann = ann.copy()
                new_ann['category_id'] = new_cat_id
                new_ann['id'] = new_ann_id_start
                merged_anns.append(new_ann)
                new_ann_id_start += 1
    
    return merged_anns

def process_coco_json(input_json, output_json):
    """Process COCO JSON file with class merging logic"""
    
    print(f"Loading COCO JSON: {input_json}")
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    images = data['images']
    annotations = data['annotations']
    
    print(f"Found {len(images)} images and {len(annotations)} annotations")
    
    # Group annotations by image_id
    img_id_to_anns = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)
    
    # Process each image
    new_annotations = []
    
    for img_id, anns in img_id_to_anns.items():
        processed_anns = process_annotations_for_image(anns)
        new_annotations.extend(processed_anns)
    
    # Create output data structure
    output_data = {
        'images': images,
        'annotations': new_annotations,
        'categories': create_target_categories()
    }
    
    print(f"Saving merged COCO JSON: {output_json}")
    with open(output_json, 'w') as f:
        json.dump(output_data, f)
    
    print(f"Merged dataset: {len(new_annotations)} annotations with {len(output_data['categories'])} categories")
    
    # Print category statistics
    cat_counts = {}
    for ann in new_annotations:
        cat_id = ann['category_id']
        cat_counts[cat_id] = cat_counts.get(cat_id, 0) + 1
    
    print("\nCategory distribution:")
    for cat in output_data['categories']:
        count = cat_counts.get(cat['id'], 0)
        print(f"  {cat['name']} (id={cat['id']}): {count} annotations")
    
    return output_data

def main():
    parser = argparse.ArgumentParser(description='Merge classes in VisDrone COCO JSON format')
    parser.add_argument('--input-json', required=True, help='Input COCO JSON file')
    parser.add_argument('--output-json', required=True, help='Output COCO JSON file')
    
    args = parser.parse_args()
    
    process_coco_json(args.input_json, args.output_json)

if __name__ == "__main__":
    main()