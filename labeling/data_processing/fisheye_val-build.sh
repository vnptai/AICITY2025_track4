#!/bin/bash

# Set up paths
COCO_FILE="labeling/data_processing/syn_labels/val_syn.json"
YOLO_OUTPUT_DIR="data/val/labels_syn"
IMAGE_DIR="data/val/images"
GT_DIR="data/val/labels"
FINAL_OUTPUT_DIR="data/val_syn"

echo "Starting fisheye training dataset build process..."

# Step 1: Convert COCO format to YOLO format
echo "Step 1: Converting COCO labels to YOLO format..."
mkdir -p $YOLO_OUTPUT_DIR

python3 labeling/data_processing/coco_to_yolo.py \
  --coco $COCO_FILE \
  --output $YOLO_OUTPUT_DIR \
  --verbose

# Step 2: Merge the ground truth and predicted labels using enhance_bboxes.py
echo "Step 2: Merging ground truth and synthetic labels..."
mkdir -p $FINAL_OUTPUT_DIR/labels
mkdir -p $FINAL_OUTPUT_DIR/images

python3 labeling/data_processing/enhance_bboxes.py \
  --gt-dir $GT_DIR \
  --pred-dir $YOLO_OUTPUT_DIR \
  --output-dir $FINAL_OUTPUT_DIR/labels \
  --img-dir $IMAGE_DIR \
  --vis-dir $FINAL_OUTPUT_DIR/visualizations

# Step 3: Copy images to the final directory
echo "Step 3: Copying images to the final directory..."
cp $IMAGE_DIR/* $FINAL_OUTPUT_DIR/images/

echo "Process completed successfully!"
echo "Final dataset available at: $FINAL_OUTPUT_DIR"
