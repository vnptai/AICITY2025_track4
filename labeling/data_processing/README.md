# VisDrone Data Processing Pipeline

This directory contains scripts for processing VisDrone dataset through a comprehensive pipeline.

## Quick Start

```bash
cd labeling/data_processing
./visdrone_build.sh
```

## Workflow

1. **Process VisDrone Dataset**
   - Merge classes → Cut images → Fisheye augmentation

2. **Convert Synthetic Labels**
   - COCO to YOLO format

3. **Enhance Labels with Synthetic Data**
   - Add synthetic annotations to ground truth labels

## Input/Output

**Input:**
- `data/visdrone/images`: Original images
- `data/visdrone/labels`: Original labels
- `syn_labels/visdrone_syn.json`: Synthetic labels (COCO format)

**Output:**
- `data/visdrone/images-aug`: Fisheye images
- `data/visdrone/labels-aug`: Fisheye labels
- `data/visdrone_syn_enhance`: Final enhanced dataset

## Class Mapping in Visdrone

The class mapping used in merge_classes.py:

- 0 (ignored region) → 3
- 1 (pedestrian) → 1
- 2 (people) → 1
- 3 (bicycle) → 2
- 4 (car) → 2
- 5 (van) → 4
- 6, 7 (truck, tricycle) → removed
- 8 (awning-tricycle) → 0
- 9 (bus) → 1