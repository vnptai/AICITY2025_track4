import json
import argparse
from pathlib import Path

def iou(boxA, boxB):
    """
    box format: [x, y, w, h]
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0:
        return 0.0
    return interArea / unionArea

def merge_annotations(gt_file, pseudo_file, output_file, iou_thr=0.5, score_thr=0.4):
    # Load ground truth COCO format
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)

    # Load pseudo labels
    with open(pseudo_file, 'r', encoding='utf-8') as f:
        pseudo_data = json.load(f)

    # Copy ground truth annotations
    new_data = gt_data.copy()
    annotations = new_data.get("annotations", [])

    # Index GT annotations theo image_id
    gt_by_image = {}
    for ann in gt_data["annotations"]:
        gt_by_image.setdefault(ann["image_id"], []).append(ann)

    # Lấy ID lớn nhất hiện có để tiếp tục đánh số
    next_ann_id = max([ann["id"] for ann in annotations], default=0) + 1

    added, skipped = 0, 0

    for pred in pseudo_data:
        image_id = pred["image_id"]
        pred_box = pred["bbox"]
        pred_score = pred["score"]
        pred_cat = pred["category_id"]

        if pred_score < score_thr:
            skipped += 1
            continue

        matched = False
        if image_id in gt_by_image:
            for gt_ann in gt_by_image[image_id]:
                iou_val = iou(pred_box, gt_ann["bbox"])
                if iou_val > iou_thr:
                    matched = True
                    break

        if matched:
            new_ann = {
                "id": next_ann_id,
                "image_id": image_id,
                "category_id": pred_cat,
                "bbox": pred_box,
                "area": pred_box[2] * pred_box[3],
                "iscrowd": 0,
                "score": pred_score
            }
            annotations.append(new_ann)
            next_ann_id += 1
            added += 1
        else:
            skipped += 1

    new_data["annotations"] = annotations

    # Save output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Merge completed: {added} bbox added, {skipped} bbox skipped")

def main():
    parser = argparse.ArgumentParser(description="Merge COCO ground truth and pseudo labels")
    parser.add_argument("--gt", type=Path, required=True, help="Path to ground truth COCO json file")
    parser.add_argument("--pseudo", type=Path, required=True, help="Path to pseudo label json file")
    parser.add_argument("--out", type=Path, required=True, help="Path to output merged json file")
    parser.add_argument("--iou_thr", type=float, default=0.5, help="IoU threshold")
    parser.add_argument("--score_thr", type=float, default=0.4, help="Score threshold")

    args = parser.parse_args()

    merge_annotations(args.gt, args.pseudo, args.out, args.iou_thr, args.score_thr)

if __name__ == "__main__":
    main()
