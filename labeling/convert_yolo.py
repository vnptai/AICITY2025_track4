import os
import json
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def convert_yolo(coco_json_path, output_folder, float_fmt=".6f"):
    """
    Chuyển COCO annotations -> YOLO txt.
    - Input: coco_json_path (instances_*.json)
    - Output: output_folder chứa các file <image_stem>.txt
    """
    # Load COCO JSON
    with open(coco_json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # Tạo thư mục output
    os.makedirs(output_folder, exist_ok=True)

    # Map category_id -> index (0..N-1) theo thứ tự xuất hiện trong 'categories'
    if "categories" not in coco or not coco["categories"]:
        raise ValueError("COCO JSON không có trường 'categories' hợp lệ.")
    cat_id_to_idx = {cat["id"]: idx for idx, cat in enumerate(coco["categories"])}

    # Gom annotations theo image_id để truy cập nhanh
    anns_by_img = defaultdict(list)
    for ann in coco.get("annotations", []):
        # Bỏ qua ann không có bbox
        if "bbox" not in ann or not ann["bbox"]:
            continue
        anns_by_img[ann["image_id"]].append(ann)

    images = coco.get("images", [])
    if not images:
        raise ValueError("COCO JSON không có trường 'images' hợp lệ.")

    for img in tqdm(images, desc="Đang tạo nhãn YOLO"):
        img_id = img["id"]
        img_w = img.get("width")
        img_h = img.get("height")
        img_name = img.get("file_name")
        if img_w is None or img_h is None or not img_name:
            # Cần width/height để chuẩn hoá
            raise ValueError(f"Thiếu width/height/file_name cho image_id={img_id}")

        label_path = os.path.join(output_folder, f"{Path(img_name).stem}.txt")
        lines = []

        for ann in anns_by_img.get(img_id, []):
            cat_id = ann["category_id"]
            bbox = ann["bbox"]  # [x_min, y_min, width, height] (COCO)
            if len(bbox) != 4:
                continue

            x_min, y_min, bw, bh = bbox
            # Chuyển COCO -> YOLO: (x_center, y_center, w, h) chuẩn hoá [0,1]
            x_c = (x_min + bw / 2.0) / img_w
            y_c = (y_min + bh / 2.0) / img_h
            w_n = bw / img_w
            h_n = bh / img_h

            # (tuỳ chọn) kẹp về [0,1] để tránh lệch do làm tròn/annotation lố ảnh
            x_c = max(0.0, min(1.0, x_c))
            y_c = max(0.0, min(1.0, y_c))
            w_n = max(0.0, min(1.0, w_n))
            h_n = max(0.0, min(1.0, h_n))

            cls_idx = cat_id_to_idx[cat_id]
            lines.append(
                f"{cls_idx} {x_c:{float_fmt}} {y_c:{float_fmt}} {w_n:{float_fmt}} {h_n:{float_fmt}}"
            )

        # Ghi file (có thể rỗng nếu không có bbox)
        with open(label_path, "w", encoding="utf-8") as lf:
            lf.write("\n".join(lines))

    print(f"✅ Hoàn tất. Đã tạo nhãn YOLO tại: {output_folder}")
    print(f"🔢 Số lớp: {len(cat_id_to_idx)} (theo thứ tự trong 'categories').")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert COCO JSON -> YOLO .txt (chỉ tạo nhãn, không cần thư mục ảnh)."
    )
    parser.add_argument(
        "--coco_json",
        type=str,
        required=True,
        help="Đường dẫn file COCO JSON (instances_*.json)",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Thư mục đầu ra chứa các .txt",
    )
    parser.add_argument(
        "--float_digits",
        type=int,
        default=6,
        help="Số chữ số thập phân cho toạ độ (mặc định 6).",
    )
    args = parser.parse_args()

    convert_yolo(
        args.coco_json,
        args.output_folder,
        float_fmt=f".{args.float_digits}f",
    )
