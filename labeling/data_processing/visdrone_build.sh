#!/bin/bash
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
INPUT_JSON="${PROJECT_ROOT}/data/visdrone/visdrone-train.json"
INPUT_IMG_DIR="${PROJECT_ROOT}/data/visdrone/images"
WORK_DIR="${PROJECT_ROOT}/data/visdrone_enhance"
FINAL_JSON="${WORK_DIR}/visdrone_enhance.json"
FINAL_YOLO_DIR="${WORK_DIR}/labels"

# Utility functions
print_header() {
    echo -e "\n${BLUE}=======================================================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${BLUE}=======================================================================${NC}\n"
}

print_step() {
    echo -e "${YELLOW}===> $1${NC}"
}

check_file() {
    if [ ! -f "$1" ]; then
        echo -e "${RED}ERROR: File $1 does not exist!${NC}"
        exit 1
    fi
}

check_directory() {
    if [ ! -d "$1" ]; then
        echo -e "${RED}ERROR: Directory $1 does not exist!${NC}"
        exit 1
    fi
}

# Check if input files exist
check_file "${INPUT_JSON}"
check_directory "${INPUT_IMG_DIR}"

# Create working directories
mkdir -p "${WORK_DIR}/images"
mkdir -p "${WORK_DIR}/merged"
mkdir -p "${WORK_DIR}/cut"
mkdir -p "${FINAL_YOLO_DIR}"

# ===== STEP 1: MERGE CLASSES =====
print_header "STEP 1: MERGE CLASSES"
print_step "Running merge_classes.py"

python "${BASE_DIR}/visdrone-train/merge_classes.py" \
    --input-json "${INPUT_JSON}" \
    --output-json "${WORK_DIR}/merged/merged.json"

# ===== STEP 2: CUT IMAGES =====  
print_header "STEP 2: CUT IMAGES"
print_step "Running cut_image.py"

python "${BASE_DIR}/visdrone-train/cut_image.py" \
    --input-json "${WORK_DIR}/merged/merged.json" \
    --input-img-dir "${INPUT_IMG_DIR}" \
    --output-json "${WORK_DIR}/cut/cut.json" \
    --output-img-dir "${WORK_DIR}/cut/images"

# ===== STEP 3: AUGMENT WITH FISHEYE =====
print_header "STEP 3: AUGMENT WITH FISHEYE"
print_step "Running aug_visdrone.py"

python "${BASE_DIR}/visdrone-train/aug_visdrone.py" \
    --input-json "${WORK_DIR}/cut/cut.json" \
    --input-img-dir "${WORK_DIR}/cut/images" \
    --output-json "${FINAL_JSON}" \
    --output-img-dir "${WORK_DIR}/images"

# ===== STEP 4: CONVERT TO YOLO FORMAT =====
print_header "STEP 4: CONVERT TO YOLO FORMAT"
print_step "Running coco_to_yolo.py"

python "${BASE_DIR}/visdrone-train/coco_to_yolo.py" \
    --input-json "${FINAL_JSON}" \
    --output-label-dir "${FINAL_YOLO_DIR}"

print_header "PROCESSING COMPLETE!"

# Print statistics
if [ -f "${FINAL_JSON}" ]; then
    JSON_SIZE=$(du -h "${FINAL_JSON}" | cut -f1)
    echo "Final JSON: ${FINAL_JSON} (${JSON_SIZE})"
    
    # Get statistics from JSON
    JSON_STATS=$(python -c "
import json
with open('${FINAL_JSON}', 'r') as f:
    data = json.load(f)
print(f'Images: {len(data.get(\"images\", []))}')
print(f'Annotations: {len(data.get(\"annotations\", []))}')
print(f'Categories: {len(data.get(\"categories\", []))}')
")
    echo "${JSON_STATS}"
else
    echo -e "${RED}ERROR: Final JSON file was not created!${NC}"
fi

if [ -d "${FINAL_YOLO_DIR}" ]; then
    YOLO_COUNT=$(find "${FINAL_YOLO_DIR}" -name "*.txt" | wc -l)
    echo "YOLO labels: ${YOLO_COUNT} files"
else
    echo -e "${RED}ERROR: YOLO label directory was not created!${NC}"
fi

if [ -d "${WORK_DIR}/images" ]; then
    IMG_COUNT=$(find "${WORK_DIR}/images" -type f | wc -l)
    echo "Final images: ${IMG_COUNT} files"
else
    echo -e "${RED}ERROR: Final image directory was not created!${NC}"
fi

echo -e "\nOutput locations:"
echo "- Final JSON: ${FINAL_JSON}"
echo "- Final images: ${WORK_DIR}/images"
echo "- YOLO labels: ${FINAL_YOLO_DIR}"

exit 0