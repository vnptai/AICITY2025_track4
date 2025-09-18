import cv2
import numpy as np
import math
import os
import glob
from pathlib import Path
import argparse
import tqdm

def rect_to_fisheye_point(x_src, y_src, S, f_fisheye):
    """
    Convert a point (x_src, y_src) on a square rectilinear image of size S×S
    (center C = S/2, focal length of rectilinear = S/2) to the corresponding coordinates
    on an equidistant fisheye image of the same size S×S (center C = S/2, fisheye focal length = f_fisheye).

    Returns (x_out, y_out). If the point is outside the circle (theta > theta_max),
    clamp theta to theta_max to map it to the edge of the circle (do not assign –1).
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

def convert_yolo_bbox_to_fisheye(
    txt_path,     # path to the YOLO txt file
    img_orig,     # numpy array of the original image
    k=1.0         # hệ số scale fisheye
):
    h_orig, w_orig = img_orig.shape[:2]
    # 1) Resize the original image to a square S×S, with S = min(h_orig, w_orig)
    S = min(h_orig, w_orig)
    img_square = cv2.resize(img_orig, (S, S))
    
    # 2) Calculate focal_fisheye
    theta_max = math.acos(1.0 / math.sqrt(3.0))
    R_out = S / 2.0
    f_fisheye = (R_out / theta_max) * k
    
    # 3) Read the YOLO txt file if it exists
    bboxes = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    cls_id = int(parts[0])
                    x_c_norm = float(parts[1])
                    y_c_norm = float(parts[2])
                    w_norm = float(parts[3])
                    h_norm = float(parts[4])
                except:
                    continue
                
                # Calculate bbox pixel on the original image
                x_c_abs = x_c_norm * w_orig
                y_c_abs = y_c_norm * h_orig
                w_abs   = w_norm   * w_orig
                h_abs   = h_norm   * h_orig
                
                # 4 corners on the original image: 
                x_min_orig = x_c_abs - w_abs/2.0
                x_max_orig = x_c_abs + w_abs/2.0
                y_min_orig = y_c_abs - h_abs/2.0
                y_max_orig = y_c_abs + h_abs/2.0
                
                # 4) Scale to the square image S×S:
                scale_x = S / w_orig
                scale_y = S / h_orig
                x_min_sq = x_min_orig * scale_x
                x_max_sq = x_max_orig * scale_x
                y_min_sq = y_min_orig * scale_y
                y_max_sq = y_max_orig * scale_y
                
                # 5) Calculate 4 corners
                corners = [
                    (x_min_sq, y_min_sq),
                    (x_min_sq, y_max_sq),
                    (x_max_sq, y_min_sq),
                    (x_max_sq, y_max_sq),
                ]
                # 6) Map each corner to fisheye
                fisheye_corners = [rect_to_fisheye_point(x, y, S, f_fisheye) for (x, y) in corners]
                xs_f = [pt[0] for pt in fisheye_corners]
                ys_f = [pt[1] for pt in fisheye_corners]
                
                # 7) Calculate bbox on fisheye
                x_min_f = min(xs_f)
                x_max_f = max(xs_f)
                y_min_f = min(ys_f)
                y_max_f = max(ys_f) 
                
                # 8) Clip to [0..S]
                x_min_f = max(0.0, min(S, x_min_f))
                x_max_f = max(0.0, min(S, x_max_f))
                y_min_f = max(0.0, min(S, y_min_f))
                y_max_f = max(0.0, min(S, y_max_f))
                
                # 9) Convert to YOLO normalized format
                w_f = x_max_f - x_min_f
                h_f = y_max_f - y_min_f
                x_c_f = x_min_f + w_f/2.0
                y_c_f = y_min_f + h_f/2.0
                
                # Normalized
                x_c_f_norm = x_c_f / S
                y_c_f_norm = y_c_f / S
                w_f_norm   = w_f   / S
                h_f_norm   = h_f   / S
                
                bboxes.append((cls_id, x_c_f_norm, y_c_f_norm, w_f_norm, h_f_norm))
    
    return img_square, f_fisheye, bboxes


def create_fisheye_image(img_square, f_fisheye, interp=cv2.INTER_CUBIC):
    S = img_square.shape[0]
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
        img_square,
        map_x, map_y,
        interpolation=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    return dst_fisheye

def crop_and_adjust_bboxes(fisheye_img, bboxes, crop_ratio=0.827):
    S = fisheye_img.shape[0]
    new_size = int(S * crop_ratio)
    start = (S - new_size) // 2
    end = start + new_size
    
    # Crop image
    cropped_img = fisheye_img[start:end, start:end]
    
    # Adjust bounding boxes
    new_bboxes = []
    for cls_id, x_c_norm, y_c_norm, w_norm, h_norm in bboxes:
        # Convert normalized to pixel (S×S)
        x_c = x_c_norm * S
        y_c = y_c_norm * S
        w = w_norm * S
        h = h_norm * S
        
        # Adjust coordinates due to crop
        x_c_new = x_c - start
        y_c_new = y_c - start
        
        # Calculate the boundary points
        x_min_new = max(0, x_c_new - w/2)
        x_max_new = min(new_size, x_c_new + w/2)
        y_min_new = max(0, y_c_new - h/2)
        y_max_new = min(new_size, y_c_new + h/2)
        
        w_new = x_max_new - x_min_new
        h_new = y_max_new - y_min_new
        
        if w_new > 0 and h_new > 0:
            x_c_new = (x_min_new + x_max_new) / 2
            y_c_new = (y_min_new + y_max_new) / 2
            
            # Convert to normalized so với kích thước mới
            x_c_norm_new = x_c_new / new_size
            y_c_norm_new = y_c_new / new_size
            w_norm_new = w_new / new_size
            h_norm_new = h_new / new_size
            
            new_bboxes.append((cls_id, x_c_norm_new, y_c_norm_new, w_norm_new, h_norm_new))
    
    return cropped_img, new_bboxes

def process_folder(
    img_folder, 
    label_folder, 
    output_img_folder, 
    output_label_folder,
    crop_ratio=0.827
):
    # Create output directory if it doesn't exist
    os.makedirs(output_img_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)
    
    print(f"Processing images from {img_folder}")
    print(f"Using labels from {label_folder}")
    print(f"Output images will be saved to {output_img_folder}")
    print(f"Output labels will be saved to {output_label_folder}")
    print(f"Using crop ratio: {crop_ratio}")
    
    # Get list of images in the directory
    img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    img_paths = []
    for ext in img_extensions:
        img_paths.extend(glob.glob(os.path.join(img_folder, ext)))
    
    total = len(img_paths)
    processed = 0
    errors = []
    
    print(f"Starting to process {total} images...")
    
    # Use tqdm for progress bar
    for img_path in tqdm.tqdm(img_paths, desc="Processing fisheye images"):
        try:
            # Build label path corresponding
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(label_folder, base_name + ".txt")
            
            # Read image
            img_orig = cv2.imread(img_path)
            if img_orig is None:
                errors.append(f"Failed to read image: {img_path}")
                continue
                
            # Process fisheye
            img_square, f_fisheye, bboxes_fisheye = convert_yolo_bbox_to_fisheye(txt_path, img_orig)
            fisheye_img = create_fisheye_image(img_square, f_fisheye)
            
            # Crop and adjust bbox
            fisheye_cropped, bboxes_cropped = crop_and_adjust_bboxes(
                fisheye_img, bboxes_fisheye, crop_ratio
            )
            
            # Save image
            output_img_path = os.path.join(output_img_folder, base_name + ".jpg")
            cv2.imwrite(output_img_path, fisheye_cropped)
            
            # Save label
            output_txt_path = os.path.join(output_label_folder, base_name + ".txt")
            with open(output_txt_path, 'w') as f:
                for bbox in bboxes_cropped:
                    cls_id, xcn, ycn, wn, hn = bbox
                    f.write(f"{cls_id} {xcn:.6f} {ycn:.6f} {wn:.6f} {hn:.6f}\n")
            
            processed += 1
                
        except Exception as e:
                errors.append(f"Error processing {img_path}: {str(e)}")
    
    # Report result
    print("\nProcessing complete!")
    print(f"Successfully processed: {processed}/{total} images")
    
    if errors:
        print("\nErrors encountered:")
        with open(os.path.join(output_dir, "errors.log"), "w") as error_file:
            for error in errors:
                print(f" - {error}")
                error_file.write(f"{error}\n")
        print(f"Error details saved to {os.path.join(output_dir, 'errors.log')}")
    else:
        print("No errors occurred")

def parse_args():
    parser = argparse.ArgumentParser(description='Create fisheye augmented images from VisDrone dataset')
    parser.add_argument('--input-dir', type=str, default='dataset/processed_visdrone',
                        help='Path to the processed VisDrone dataset directory')
    parser.add_argument('--output-dir', type=str, default='dataset/aug_visdrone',
                        help='Path to save the output augmented images and labels')
    parser.add_argument('--crop-ratio', type=float, default=0.827,
                        help='Crop ratio for fisheye images')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Set up paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    img_folder = input_dir / 'images'
    label_folder = input_dir / 'merged_labels'
    output_img_folder = output_dir / 'images'
    output_label_folder = output_dir / 'labels'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all images
    process_folder(
        img_folder=img_folder,
        label_folder=label_folder,
        output_img_folder=output_img_folder,
        output_label_folder=output_label_folder,
        crop_ratio=args.crop_ratio
    )
    