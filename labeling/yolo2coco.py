#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from PIL import Image
from typing import List, Dict, Tuple, Optional

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Fixed categories as requested
CATEGORIES = [
    {"id": 0, "name": "Bus"},
    {"id": 1, "name": "Bike"},
    {"id": 2, "name": "Car"},
    {"id": 3, "name": "Pedestrian"},
    {"id": 4, "name": "Truck"},
]
VALID_CLASS_IDS = {c["id"] for c in CATEGORIES}

def yolo_to_coco_bbox(cx, cy, w, h, img_w, img_h) -> List[float]:
    x = (cx - w / 2.0) * img_w
    y = (cy - h / 2.0) * img_h
    bw = w * img_w
    bh = h * img_h
    # clamp to image bounds
    x = max(0.0, min(x, img_w))
    y = max(0.0, min(y, img_h))
    bw = max(0.0, min(bw, img_w - x))
    bh = max(0.0, min(bh, img_h - y))
    return [x, y, bw, bh]

def parse_yolo_line(line: str) -> Optional[Tuple[int, float, float, float, float]]:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    try:
        cls = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        return cls, cx, cy, w, h
    except Exception:
        return None

def find_label_for_image(img_path: Path, labels_dir: Path) -> Path:
    return labels_dir / (img_path.stem + ".txt")

def collect_images(images_dir: Path) -> List[Path]:
    return [p for p in sorted(images_dir.rglob("*"))
            if p.suffix.lower() in IMG_EXTS and p.is_file()]

def main():
    ap = argparse.ArgumentParser(description="Convert YOLO txt labels to COCO JSON.")
    ap.add_argument("--images", required=True, type=Path, help="Folder ảnh (VD: /data/images)")
    ap.add_argument("--labels", required=True, type=Path, help="Folder nhãn YOLO (VD: /data/labels)")
    ap.add_argument("--out", required=True, type=Path, help="Đường dẫn JSON xuất ra (VD: /data/label.json)")
    args = ap.parse_args()

    if not args.images.exists() or not args.images.is_dir():
        print(f"[ERROR] --images không hợp lệ: {args.images}", file=sys.stderr); sys.exit(1)
    if not args.labels.exists() or not args.labels.is_dir():
        print(f"[ERROR] --labels không hợp lệ: {args.labels}", file=sys.stderr); sys.exit(1)

    image_paths = collect_images(args.images)
    if not image_paths:
        print(f"[ERROR] Không tìm thấy ảnh trong: {args.images}", file=sys.stderr); sys.exit(1)

    coco = {
        "info": {"description": "Dataset chuyển từ YOLO sang COCO", "version": "1.0", "year": 2025},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": CATEGORIES,
    }

    images = []
    annotations = []
    image_id = 1
    ann_id = 1

    # Để output file_name đẹp (tương đối từ gốc /data)
    dataset_root = args.images.parent if args.images.parent.exists() else args.images

    for img_path in image_paths:
        try:
            with Image.open(img_path) as im:
                width, height = im.size
        except Exception as e:
            print(f"[WARN] Không mở được ảnh {img_path}: {e}", file=sys.stderr)
            continue

        try:
            rel_name = str(img_path.relative_to(dataset_root))
        except Exception:
            rel_name = img_path.name

        images.append({
            "id": image_id,
            "file_name": rel_name,
            "width": width,
            "height": height
        })

        label_path = find_label_for_image(img_path, args.labels)
        if label_path.exists():
            with label_path.open("r", encoding="utf-8") as f:
                for ln_no, ln in enumerate(f, start=1):
                    parsed = parse_yolo_line(ln)
                    if parsed is None:
                        print(f"[WARN] Bỏ qua dòng không hợp lệ {label_path}:{ln_no}: {ln.strip()}", file=sys.stderr)
                        continue
                    cls, cx, cy, w, h = parsed
                    if cls not in VALID_CLASS_IDS:
                        print(f"[WARN] Bỏ qua class_id ngoài tập [0..4] tại {label_path}:{ln_no} (class={cls})", file=sys.stderr)
                        continue
                    bbox = yolo_to_coco_bbox(cx, cy, w, h, width, height)
                    area = bbox[2] * bbox[3]
                    annotations.append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": cls,
                        "iscrowd": 0,
                        "bbox": [round(v, 2) for v in bbox],
                        "area": round(area, 2),
                        "segmentation": []  # YOLO bbox không có polygon
                    })
                    ann_id += 1

        image_id += 1

    coco["images"] = images
    coco["annotations"] = annotations

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    print(f"✅ Đã tạo COCO JSON: {args.out}")
    print(f"- Số ảnh: {len(images)}")
    print(f"- Số annotations: {len(annotations)}")
    print(f"- Categories cố định: {', '.join([c['name'] for c in CATEGORIES])}")

if __name__ == "__main__":
    main()
