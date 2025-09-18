import os
from PIL import Image
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Cut images into two parts and adjust annotations')
    parser.add_argument('--input-dir', type=str, default='dataset/processed_visdrone',
                        help='Path to the processed VisDrone dataset directory')
    parser.add_argument('--output-dir', type=str, default='dataset/cut_visdrone',
                        help='Path to save the output cut images and labels')
    return parser.parse_args()

# ==========================
# HÀM HỖ TRỢ
# ==========================
def read_yolo_labels(label_path):
    """
    Đọc file label YOLO (.txt), trả về list các bbox dạng:
    [(class_id, x_center_norm, y_center_norm, w_norm, h_norm), ...]
    """
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            boxes.append((cls, x_center, y_center, w, h))
    return boxes

def write_yolo_labels(label_path, boxes):
    """
    Ghi file label YOLO (.txt) theo định dạng chuẩn
    boxes: list các tuple (class_id, x_center_norm, y_center_norm, w_norm, h_norm)
    """
    with open(label_path, "w") as f:
        for (cls, xc, yc, w, h) in boxes:
            line = f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n"
            f.write(line)

def clip(val, min_v, max_v):
    """Clip giá trị val về khoảng [min_v, max_v]"""
    return max(min(val, max_v), min_v)

def process_single_image(image_path, label_path):
    """
    Xử lý một ảnh + label YOLO tương ứng:
    - Cắt ra 2 ảnh H×H
    - Chỉ tính diện tích khi bbox cắt qua biên:
        + Ảnh 1 (bên trái): đường biên x = H
        + Ảnh 2 (bên phải): đường biên x = W-H
      Nếu bbox nằm hoàn toàn bên trong ảnh con → giữ nguyên.
      Nếu bbox cắt qua biên → kiểm tra area phần nằm trong crop ≥ 0.70 area gốc mới giữ (và clamp tọa độ).
    - Trả về danh sách:
        [
           (left_image, left_boxes, left_name),
           (right_image, right_boxes, right_name)
        ]
    """
    img    = Image.open(image_path)
    W, H   = img.size
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Chỉ xử lý nếu W > H và W < 2H
    if not (W > H and W < 2 * H):
        print(f"⚠️ Ảnh {image_path} không thỏa điều kiện W>H và W<2H. Bỏ qua.")
        return []

    # Đọc label gốc (YOLO)
    orig_boxes = read_yolo_labels(label_path)

    # Chuyển mỗi bbox từ normalized (xc, yc, w, h) → pixel (x1, y1, x2, y2)
    abs_boxes = []
    for (cls, xc, yc, w_norm, h_norm) in orig_boxes:
        box_w = w_norm * W
        box_h = h_norm * H
        cx = xc * W
        cy = yc * H
        x1 = cx - box_w / 2
        y1 = cy - box_h / 2
        x2 = cx + box_w / 2
        y2 = cy + box_h / 2
        abs_boxes.append((cls, x1, y1, x2, y2))

    # --- Chuẩn bị hai ảnh con ---
    # Ảnh 1 (bên trái): crop (0,0) → (H,H)
    box_left  = (0, 0, H, H)
    img_left  = img.crop(box_left)

    # Ảnh 2 (bên phải): crop (W-H,0) → (W,H)
    box_right = (W - H, 0, W, H)
    img_right = img.crop(box_right)

    # Danh sách sẽ chứa các bbox đã được chuyển lại về YOLO-normalized trên ảnh H×H
    left_boxes_new  = []
    right_boxes_new = []

    # Duyệt từng bbox gốc
    for (cls, x1, y1, x2, y2) in abs_boxes:
        # Diện tích gốc
        area_orig = max(0, x2 - x1) * max(0, y2 - y1)
        if area_orig <= 0:
            continue

        # === XỬ LÝ CHO ẢNH 1 (bên trái, crop x ∈ [0, H]) ===
        # Trường hợp 1.A: bbox nằm hoàn toàn bên trái (x2 ≤ H) → giữ nguyên
        if x2 <= H:
            new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2
            cx_n = ((new_x1 + new_x2) / 2) / H
            cy_n = ((new_y1 + new_y2) / 2) / H
            w_n  = (new_x2 - new_x1) / H
            h_n  = (new_y2 - new_y1) / H
            left_boxes_new.append((cls, cx_n, cy_n, w_n, h_n))

        # Trường hợp 1.B: bbox cắt qua biên phải của ảnh 1 (x1 < H < x2)
        elif x1 < H < x2:
            # Tính phần giao với [0, H] cho ảnh 1
            lx1 = clip(x1, 0, H)
            ly1 = clip(y1, 0, H)
            lx2 = clip(x2, 0, H)
            ly2 = clip(y2, 0, H)
            w_left  = max(0, lx2 - lx1)
            h_left  = max(0, ly2 - ly1)
            area_left = w_left * h_left

            # Nếu diện tích phần này ≥ 70% area gốc → giữ và clamp
            if area_left / area_orig >= 0.4:
                new_x1, new_y1 = lx1, ly1
                new_x2, new_y2 = lx2, ly2
                cx_n = ((new_x1 + new_x2) / 2) / H
                cy_n = ((new_y1 + new_y2) / 2) / H
                w_n  = (new_x2 - new_x1) / H
                h_n  = (new_y2 - new_y1) / H
                left_boxes_new.append((cls, cx_n, cy_n, w_n, h_n))
            # Nếu <70% thì bỏ hẳn (không ghi vào ảnh 1)

        # Ngược lại (x1 ≥ H) → không thuộc ảnh 1

        # === XỬ LÝ CHO ẢNH 2 (bên phải, crop x ∈ [W-H, W]) ===
        # Với ảnh 2, ta shift toạ độ gốc: x' = x - (W - H)
        # Và chỉ quan tâm phần x' ∈ [0, H]

        # Trường hợp 2.A: bbox nằm hoàn toàn bên phải (x1 ≥ W - H) → giữ nguyên (chỉ shift)
        if x1 >= (W - H):
            new_x1 = x1 - (W - H)
            new_x2 = x2 - (W - H)
            new_y1 = y1
            new_y2 = y2
            cx_n = ((new_x1 + new_x2) / 2) / H
            cy_n = ((new_y1 + new_y2) / 2) / H
            w_n  = (new_x2 - new_x1) / H
            h_n  = (new_y2 - new_y1) / H
            right_boxes_new.append((cls, cx_n, cy_n, w_n, h_n))

        # Trường hợp 2.B: bbox cắt qua biên trái của ảnh 2 (x1 < W - H < x2)
        elif x1 < (W - H) < x2:
            # Tính phần giao với [W - H, W] cho ảnh 2, sau đó shift về [0, H]
            rx1 = clip(x1 - (W - H), 0, H)
            ry1 = clip(y1, 0, H)
            rx2 = clip(x2 - (W - H), 0, H)
            ry2 = clip(y2, 0, H)
            w_right = max(0, rx2 - rx1)
            h_right = max(0, ry2 - ry1)
            area_right = w_right * h_right

            # Nếu diện tích phần này ≥ 70% area gốc → giữ và clamp
            if area_right / area_orig >= 0.4:
                new_x1, new_y1 = rx1, ry1
                new_x2, new_y2 = rx2, ry2
                cx_n = ((new_x1 + new_x2) / 2) / H
                cy_n = ((new_y1 + new_y2) / 2) / H
                w_n  = (new_x2 - new_x1) / H
                h_n  = (new_y2 - new_y1) / H
                right_boxes_new.append((cls, cx_n, cy_n, w_n, h_n))
            # Nếu <70% thì bỏ hẳn (không ghi vào ảnh 2)

        # Ngược lại (x2 ≤ W - H) → không thuộc ảnh 2

    # Đặt tên file mới
    left_name  = f"{base_name}_1.png"
    right_name = f"{base_name}_2.png"

    return [
        (img_left, left_boxes_new, left_name),
        (img_right, right_boxes_new, right_name)
    ]

def process_folder(images_dir, labels_dir, output_images_dir, output_labels_dir):
    """Process all images in the specified folder"""
    # Tạo thư mục đầu ra nếu chưa có
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    
    print(f"Processing images from {images_dir}")
    print(f"Using labels from {labels_dir}")
    print(f"Output images will be saved to {output_images_dir}")
    print(f"Output labels will be saved to {output_labels_dir}")
    
    total_images = 0
    processed_images = 0
    skipped_images = 0
    
    for fname in os.listdir(images_dir):
        if not (fname.lower().endswith(".png") or 
                fname.lower().endswith(".jpg") or 
                fname.lower().endswith(".jpeg")):
            continue
            
        total_images += 1
        image_path = os.path.join(images_dir, fname)
        base_name = os.path.splitext(fname)[0]
        label_path = os.path.join(labels_dir, base_name + ".txt")

        # Nếu không tìm thấy file label tương ứng → bỏ qua
        if not os.path.exists(label_path):
            print(f"⚠️ Missing label for image {fname}, skipping.")
            skipped_images += 1
            continue

        # Xử lý 1 ảnh
        results = process_single_image(image_path, label_path)
        if not results:
            skipped_images += 1
            continue
            
        for (img_cropped, new_boxes, new_fname) in results:
            # 1. Lưu ảnh
            save_path = os.path.join(output_images_dir, new_fname)
            img_cropped.save(save_path)

            # 2. Lưu label
            save_lbl_path = os.path.join(
                output_labels_dir, 
                os.path.splitext(new_fname)[0] + ".txt"
            )
            write_yolo_labels(save_lbl_path, new_boxes)

        processed_images += 1
        if processed_images % 10 == 0:
            print(f"Progress: {processed_images}/{total_images} images processed")

    print(f"\n✅ Processing complete!")
    print(f"  - Total images: {total_images}")
    print(f"  - Successfully processed: {processed_images}")
    print(f"  - Skipped: {skipped_images}")


if __name__ == "__main__":
    args = parse_args()
    
    # Set up paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    images_dir = input_dir / 'images'
    labels_dir = input_dir / 'merged_labels'
    output_images_dir = output_dir / 'images'
    output_labels_dir = output_dir / 'labels'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all images
    process_folder(images_dir, labels_dir, output_images_dir, output_labels_dir)
