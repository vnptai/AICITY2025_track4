import os
from PIL import Image
import math
import argparse
from pathlib import Path

def xywhn_to_xyxy(xc, yc, w, h, W, H):
    x1 = (xc - w/2) * W
    y1 = (yc - h/2) * H
    x2 = (xc + w/2) * W
    y2 = (yc + h/2) * H
    return x1, y1, x2, y2

def xyxy_to_xywhn(x1, y1, x2, y2, W, H):
    xc = ((x1 + x2) / 2) / W
    yc = ((y1 + y2) / 2) / H
    w  = (x2 - x1) / W
    h  = (y2 - y1) / H
    return xc, yc, w, h

def compute_iou(boxA, boxB):
    xa = max(boxA[0], boxB[0]); ya = max(boxA[1], boxB[1])
    xb = min(boxA[2], boxB[2]); yb = min(boxA[3], boxB[3])
    inter_w = max(0, xb - xa); inter_h = max(0, yb - ya)
    inter = inter_w * inter_h
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter/union if union>0 else 0

def compute_overlap_ratio(boxA, boxB):
    """
    Trả về tỉ lệ diện tích giao của A so với diện tích A.
    """
    xa = max(boxA[0], boxB[0]); ya = max(boxA[1], boxB[1])
    xb = min(boxA[2], boxB[2]); yb = min(boxA[3], boxB[3])
    inter_w = max(0, xb - xa); inter_h = max(0, yb - ya)
    inter = inter_w * inter_h
    areaA = max(0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    return (inter / areaA) if areaA > 0 else 0.0

def merge_person_with_vehicles(objs, used, vehicle_ids=(2,9), remove_ids=(6,7)):
    """
    Với mỗi person (cls=1) chưa dùng:
      0) Nếu person có IoU > 0 với bất kỳ object cls in remove_ids (6 hoặc 7),
         thì bỏ luôn person đó (đánh dấu used[i] = True) và không thêm vào merged.
      1) Ngược lại, quét qua tất cả entry đã merge cũ:
         - Tính overlap_ratio (area(person ∩ entry) / area(person)).
         - Nếu ≥ 0.8, merge vào entry đó (union box), đánh dấu used[i] = True.
      2) Nếu không merge ở bước 1, quét qua các vehicle (cls in vehicle_ids) chưa dùng:
         - Tìm vehicle có IoU lớn nhất với person.
         - Nếu IoU > 0, merge person+vehicle thành entry mới cls=1, đánh dấu used[i], used[j].
         - Nếu IoU = 0 với tất cả, giữ nguyên person (bbox gốc), đánh dấu used[i].
    Trả về danh sách các entry mới tạo (mỗi {'cls': 1, 'box': (...)})
    """
    merged = []
    for i, o1 in enumerate(objs):
        if used[i] or o1['cls'] != 1:
            continue

        # ----- Bước 0: Nếu overlap với bất kỳ cls=6 hoặc cls=7, bỏ luôn person đó -----
        skip_person = False
        for j, o2 in enumerate(objs):
            if o2['cls'] in remove_ids:
                if compute_iou(o1['box'], o2['box']) > 0.0:
                    # Bỏ person
                    used[i] = True
                    skip_person = True
                    break
        if skip_person:
            continue

        # ----- Bước 1: Check overlap_ratio ≥ 0.2 với merged entries cũ -----
        best_merge_idx = None
        best_overlap = 0.0
        for idx_m, entry in enumerate(merged):
            overlap_ratio = compute_overlap_ratio(o1['box'], entry['box'])
            # Note: This uses 0.2 threshold, not 0.8 as mentioned in the comments
            if overlap_ratio >= 0.2 and overlap_ratio > best_overlap:
                best_overlap = overlap_ratio
                best_merge_idx = idx_m

        if best_merge_idx is not None:
            # Merge person vào entry đã có sẵn
            ex_box = merged[best_merge_idx]['box']
            x1 = min(o1['box'][0], ex_box[0])
            y1 = min(o1['box'][1], ex_box[1])
            x2 = max(o1['box'][2], ex_box[2])
            y2 = max(o1['box'][3], ex_box[3])
            merged[best_merge_idx]['box'] = (x1, y1, x2, y2)
            merged[best_merge_idx]['cls'] = 1
            used[i] = True
            continue

        # ----- Bước 2: Nếu không merge với entry cũ, check với vehicle (cls=2 hoặc 9) -----
        best_j = None
        best_iou = 0.0
        for j, o2 in enumerate(objs):
            if used[j] or o2['cls'] not in vehicle_ids:
                continue
            iou = compute_iou(o1['box'], o2['box'])
            if iou > best_iou:
                best_iou = iou
                best_j = j

        if best_j is not None and best_iou > 0.0:
            o2 = objs[best_j]
            x1 = min(o1['box'][0], o2['box'][0])
            y1 = min(o1['box'][1], o2['box'][1])
            x2 = max(o1['box'][2], o2['box'][2])
            y2 = max(o1['box'][3], o2['box'][3])
            merged.append({'cls': 1, 'box': (x1, y1, x2, y2)})
            used[i] = True
            used[best_j] = True
        else:
            # Person không overlap với bất kỳ vehicle nào (IoU=0), giữ nguyên
            # merged.append({'cls': 1, 'box': o1['box']})
            used[i] = True

    return merged

def merge_zero_with_vehicles(objs, used, iou_thresh=0.1, vehicle_ids=(2,9)):
    merged = []
    for i, o0 in enumerate(objs):
        if used[i] or o0['cls'] != 0:
            continue
        merged_flag = False
        for j, o2 in enumerate(objs):
            if used[j] or o2['cls'] not in vehicle_ids:
                continue
            if compute_iou(o0['box'], o2['box']) >= iou_thresh:
                x1 = min(o0['box'][0], o2['box'][0])
                y1 = min(o0['box'][1], o2['box'][1])
                x2 = max(o0['box'][2], o2['box'][2])
                y2 = max(o0['box'][3], o2['box'][3])
                merged.append({'cls': 0, 'box': (x1, y1, x2, y2)})
                used[i] = True
                used[j] = True
                merged_flag = True
                break
        if not merged_flag:
            merged.append({'cls': 0, 'box': o0['box']})
            used[i] = True
    return merged

def remap_class(c):
    mapping = {
        0: 3,
        1: 1,
        2: 1,
        3: 2,
        4: 2,
        5: 4,
        # 6,7 removed
        8: 0,
        9: 1
    }
    return mapping.get(c)

def process_annotations(anns_dir, imgs_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Processing annotations from {anns_dir}")
    print(f"Using images from {imgs_dir}")
    print(f"Output will be saved to {out_dir}")

    for fn in os.listdir(anns_dir):
        if not fn.endswith('.txt'):
            continue
        img_path = os.path.join(imgs_dir, fn.replace('.txt', '.jpg'))
        ann_path = os.path.join(anns_dir, fn)
        out_path = os.path.join(out_dir, fn)

        if not os.path.exists(img_path):
            print(f"[WARN] Missing image: {img_path}")
            continue

        W, H = Image.open(img_path).size

        # Đọc YOLO bbox ban đầu
        objs = []
        with open(ann_path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) != 5: continue
                c, xc, yc, w, h_ = parts
                c = int(c)
                xc, yc, w_, h_ = map(float, (xc, yc, w, h_))
                x1, y1, x2, y2 = xywhn_to_xyxy(xc, yc, w_, h_, W, H)
                objs.append({'cls': c, 'box': (x1, y1, x2, y2)})

        used = [False]*len(objs)
        merged = []

        # 1) Merge person (cls=1) với logic mới (check cls 6/7, rồi overlap_ratio với merged, rồi vehicle)
        merged += merge_person_with_vehicles(objs, used, vehicle_ids=(2,9), remove_ids=(6,7))

        # 2) Merge zero (cls=0) với vehicle (cls=2 hoặc 9)
        merged += merge_zero_with_vehicles(objs, used, iou_thresh=0.1, vehicle_ids=(2,9))

        # 3) Thêm các object còn lại chưa dùng (ngoài cls=0,1,2,9)
        for idx, o in enumerate(objs):
            if used[idx]:
                continue
            if o['cls'] == 1:
                # Nếu person chưa dùng, lẽ ra đã add ở bước trên
                continue
            merged.append(o)
            used[idx] = True

        # 4) Remap class và ghi file YOLO
        lines_out = []
        for obj in merged:
            new_c = remap_class(obj['cls'])
            if new_c is None:
                continue
            x1, y1, x2, y2 = obj['box']
            xc, yc, w_, h_ = xyxy_to_xywhn(x1, y1, x2, y2, W, H)
            lines_out.append(f"{new_c} {xc:.6f} {yc:.6f} {w_:.6f} {h_:.6f}")

        with open(out_path, 'w') as fw:
            fw.write("\n".join(lines_out))

    print("\nCompleted annotation processing with the following steps:")
print("1. Merged Person class (removed if overlapping with cls 6/7)")
print("2. Merged Person with existing entries (overlap_ratio ≥ 0.2)")
print("3. Merged Person with vehicles when applicable")
print("4. Merged Zero class with vehicles (IoU ≥ 0.1)")
print("5. Remapped all classes to the target schema")


def parse_args():
    parser = argparse.ArgumentParser(description='Merge classes in VisDrone annotations')
    parser.add_argument('--input-dir', type=str, default='dataset/visdrone',
                        help='Path to the VisDrone dataset directory containing images and labels folders')
    parser.add_argument('--output-dir', type=str, default='dataset/processed_visdrone',
                        help='Path to the output directory for merged labels')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Set up paths
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    anns_dir = input_dir / 'labels'
    imgs_dir = input_dir / 'images'
    out_dir = output_dir / 'merged_labels'
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir / 'images', exist_ok=True)
    
    # Copy images to output directory for consistency
    from shutil import copy2
    print(f"Copying images from {imgs_dir} to {output_dir / 'images'}")
    for img_file in os.listdir(imgs_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            copy2(os.path.join(imgs_dir, img_file), os.path.join(output_dir / 'images', img_file))
    
    # Process annotations
    process_annotations(
        anns_dir=anns_dir,
        imgs_dir=imgs_dir,
        out_dir=out_dir
    )
    
    print(f"\nProcessing complete!")
    print(f"Merged labels saved to: {out_dir}")
    print(f"Images copied to: {output_dir / 'images'}")
