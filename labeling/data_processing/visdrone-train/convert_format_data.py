import os
from PIL import Image
import argparse
from pathlib import Path
import tqdm

def convert_visdrone_to_yolo(annotations_dir, images_dir, labels_dir):
    """Convert VisDrone annotation format to YOLO format
    
    Args:
        annotations_dir: Directory containing VisDrone annotations
        images_dir: Directory containing images
        labels_dir: Directory to save YOLO format labels
    """
    os.makedirs(labels_dir, exist_ok=True)
    
    print(f"Converting annotations from {annotations_dir}")
    print(f"Using images from {images_dir}")
    print(f"Output will be saved to {labels_dir}")
    
    files = [f for f in os.listdir(annotations_dir) if f.endswith('.txt')]
    print(f"Found {len(files)} annotation files")
    
    success_count = 0
    error_count = 0

    for filename in tqdm.tqdm(files, desc="Converting annotations"):

        annotation_path = os.path.join(annotations_dir, filename)
        image_filename = filename.replace('.txt', '.jpg')
        image_path = os.path.join(images_dir, image_filename)
        label_output_path = os.path.join(labels_dir, filename)

        if not os.path.exists(image_path):
            print(f"Image does not exist: {image_path}")
            error_count += 1
            continue

        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            error_count += 1
            continue

        yolo_lines = []
        with open(annotation_path, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue

                try:
                    bbox_left = float(parts[0])
                    bbox_top = float(parts[1])
                    bbox_width = float(parts[2])
                    bbox_height = float(parts[3])
                    score = int(parts[4])
                    category = int(parts[5])
                except ValueError:
                    continue

                if score == 0 or category == 0:
                    continue  # Bỏ qua các vùng bị bỏ qua

                class_id = category - 1  # Chuyển từ 1–10 sang 0–9

                x_center = (bbox_left + bbox_width / 2) / img_width
                y_center = (bbox_top + bbox_height / 2) / img_height
                width = bbox_width / img_width
                height = bbox_height / img_height

                yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                yolo_lines.append(yolo_line)

        with open(label_output_path, 'w') as out_file:
            out_file.write('\n'.join(yolo_lines))
            
        success_count += 1

    print(f"\nConversion completed:")
    print(f"  - Total files: {len(files)}")
    print(f"  - Successfully converted: {success_count}")
    print(f"  - Errors: {error_count}")
    
    return success_count, error_count

def parse_args():
    parser = argparse.ArgumentParser(description='Convert VisDrone annotations to YOLO format')
    parser.add_argument('--input-dir', type=str, default='dataset/visdrone',
                       help='Path to the VisDrone dataset directory')
    parser.add_argument('--output-dir', type=str, default='dataset/processed_visdrone',
                       help='Path to save the output YOLO format labels')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Set up paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    annotations_dir = input_dir / 'annotations' if (input_dir / 'annotations').exists() else input_dir / 'labels'
    images_dir = input_dir / 'images'
    labels_dir = output_dir / 'labels'
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir / 'images', exist_ok=True)
    
    # Copy images to output directory for consistency
    from shutil import copy2
    print(f"Copying images from {images_dir} to {output_dir / 'images'}")
    
    img_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for img_file in tqdm.tqdm(img_files, desc="Copying images"):
        copy2(os.path.join(images_dir, img_file), os.path.join(output_dir / 'images', img_file))
    
    # Convert annotations
    convert_visdrone_to_yolo(
        annotations_dir=annotations_dir,
        images_dir=images_dir,
        labels_dir=labels_dir
    )