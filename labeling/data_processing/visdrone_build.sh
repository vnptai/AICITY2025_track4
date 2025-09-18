#!/bin/bash

# visdrone_build.sh - Script to process VisDrone images through the entire pipeline
# Author: Claude AI
# Date: September 18, 2023

set -e  # Exit immediately if a command exits with a non-zero status

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Define paths
BASE_DIR=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(realpath "${BASE_DIR}/../..")
INPUT_DIR="${PROJECT_ROOT}/data/visdrone"
MERGED_DIR="${INPUT_DIR}/merged"
CUT_DIR="${INPUT_DIR}/cut"
AUG_DIR="${INPUT_DIR}/augmented"
FINAL_DIR="${INPUT_DIR}"
SYN_LABELS_DIR="${PROJECT_ROOT}/syn_labels"
SYN_OUTPUT_DIR="${INPUT_DIR}/labels-aug-syn"
FINAL_ENHANCE_DIR="${PROJECT_ROOT}/data/visdrone_syn_enhance"

# Utility functions
print_header() {
    echo -e "\n${BLUE}=======================================================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${BLUE}=======================================================================${NC}\n"
}

print_step() {
    echo -e "${YELLOW}===> $1${NC}"
}

check_directory() {
    if [ ! -d "$1" ]; then
        echo -e "${RED}ERROR: Directory $1 does not exist!${NC}"
        exit 1
    fi
}

# Check if input directory exists
check_directory "${INPUT_DIR}/images"
check_directory "${INPUT_DIR}/labels"

# Create necessary directories
mkdir -p "${MERGED_DIR}/images"
mkdir -p "${MERGED_DIR}/merged_labels"
mkdir -p "${CUT_DIR}/images"
mkdir -p "${CUT_DIR}/labels"
mkdir -p "${AUG_DIR}/images"
mkdir -p "${AUG_DIR}/labels"
mkdir -p "${SYN_OUTPUT_DIR}"
mkdir -p "${FINAL_ENHANCE_DIR}/images"
mkdir -p "${FINAL_ENHANCE_DIR}/labels"

# ===== STEP 1: Process VisDrone Dataset =====
print_header "STEP 1: MERGE CLASSES, CUT IMAGES, AND AUGMENT WITH FISHEYE"

# 1.1: Merge Classes
print_step "Running merge_classes.py"
python "${BASE_DIR}/visdrone-train/merge_classes.py" \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${MERGED_DIR}"

# 1.2: Cut Images
print_step "Running cut_image.py"
python "${BASE_DIR}/visdrone-train/cut_image.py" \
    --input-dir "${MERGED_DIR}" \
    --output-dir "${CUT_DIR}"

# 1.3: Augment with Fisheye
print_step "Running aug_visdrone.py"
python "${BASE_DIR}/visdrone-train/aug_visdrone.py" \
    --input-dir "${MERGED_DIR}" \
    --output-dir "${AUG_DIR}"

# Rename augmented outputs to final locations
print_step "Moving augmented output to final locations"
cp -r "${AUG_DIR}/images/"* "${FINAL_DIR}/images-aug/"
cp -r "${AUG_DIR}/labels/"* "${FINAL_DIR}/labels-aug/"

# ===== STEP 2: Convert Synthetic Labels =====
print_header "STEP 2: CONVERT SYNTHETIC LABELS FROM COCO TO YOLO FORMAT"
print_step "Running coco_to_yolov.py"

# Check if synthetic labels file exists
SYN_JSON="${SYN_LABELS_DIR}/visdrone_syn.json"
if [ ! -f "${SYN_JSON}" ]; then
    echo -e "${RED}ERROR: Synthetic labels file ${SYN_JSON} not found!${NC}"
    exit 1
fi

python "${BASE_DIR}/coco_to_yolov.py" \
    --coco "${SYN_JSON}" \
    --output "${SYN_OUTPUT_DIR}"

# ===== STEP 3: Enhance Labels with Synthetic Data =====
print_header "STEP 3: ENHANCE LABELS WITH SYNTHETIC DATA"
print_step "Running enhance_bboxes.py"

python "${BASE_DIR}/enhance_bboxes.py" \
    --gt-dir "${FINAL_DIR}/labels-aug" \
    --pred-dir "${SYN_OUTPUT_DIR}/labels" \
    --output-dir "${FINAL_ENHANCE_DIR}/labels" \
    --img-dir "${FINAL_DIR}/images-aug" \
    --vis-dir "${FINAL_ENHANCE_DIR}/visualized"

# Copy the original images to the enhanced directory
print_step "Copying images to final enhanced directory"
cp -r "${FINAL_DIR}/images-aug/"* "${FINAL_ENHANCE_DIR}/images/"

print_header "PROCESSING COMPLETE!"
echo "Output directories:"
echo "- Images: ${FINAL_ENHANCE_DIR}/images"
echo "- Labels: ${FINAL_ENHANCE_DIR}/labels"

# Print statistics
IMG_COUNT=$(find "${FINAL_ENHANCE_DIR}/images" -type f | wc -l)
LBL_COUNT=$(find "${FINAL_ENHANCE_DIR}/labels" -type f | wc -l)
echo -e "\nStatistics:"
echo "- Final images: ${IMG_COUNT}"
echo "- Final labels: ${LBL_COUNT}"

exit 0
