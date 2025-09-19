# VisDrone Data Processing Pipeline



The pipeline converts VisDrone COCO JSON format through multiple processing steps to create an enhanced dataset suitable for fisheye camera training.

## Pipeline Steps

### 1. **Merge Classes** (`merge_classes.py`)
- **Input**: `data/visdrone/subset.json` (COCO format)
- **Output**: `data/visdrone_enhance/merged/merged.json`
- **Purpose**: Consolidates 10 VisDrone classes into 5 target classes:
  - `0: Bus` (from original bus)
  - `1: Bike` (from bicycle, motor, people merged with vehicles)
  - `2: Car` (from car, van)
  - `3: Pedestrian` (from pedestrian, people)
  - `4: Truck` (from truck)
- **Logic**: Uses IoU thresholds for merging overlapping annotations

### 2. **Cut Images** (`cut_image.py`)
- **Input**: Merged JSON + `data/visdrone/images/`
- **Output**: `data/visdrone_enhance/cut/cut.json` + `data/visdrone_enhance/cut/images/`
- **Purpose**: Splits wide images into two square parts for better fisheye processing
- **Condition**: Only processes images where `W > H` and `W < 2H`
- **Output**: Creates `{basename}_1.png` and `{basename}_2.png` with adjusted annotations
- **Overlap threshold**: 40% minimum area retention for boundary-crossing objects

### 3. **Fisheye Augmentation** (`aug_visdrone.py`)
- **Input**: Cut JSON + cut images
- **Output**: `data/visdrone_enhance/visdrone_enhance.json` + `data/visdrone_enhance/images/`
- **Purpose**: Applies barrel distortion transformation to simulate fisheye camera
- **Method**: Equidistant fisheye projection with configurable focal length
- **Bbox transformation**: Converts rectangular boxes to fisheye coordinates

### 4. **Convert to YOLO** (`coco_to_yolo.py`)
- **Input**: Fisheye JSON
- **Output**: `data/visdrone_enhance/labels/` (YOLO txt format)
- **Purpose**: Converts COCO format to YOLO format for training
- **Format**: `class_id x_center y_center width height` (normalized)

## Usage

```bash
# Run the complete pipeline
bash labeling/data_processing/visdrone_build.sh
```

## Input Requirements

- **JSON file**: `data/visdrone/subset.json` (COCO format)
- **Images**: `data/visdrone/images/` (corresponding image files)
- **Categories**: Standard VisDrone 10-class format

## Output Structure

```
data/visdrone_enhance/
├── visdrone_enhance.json     # Final COCO JSON
├── images/                   # Fisheye augmented images
├── labels/                   # YOLO format labels
├── merged/                   # Intermediate: merged classes
└── cut/                      # Intermediate: cut images
```
