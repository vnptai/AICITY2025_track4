
# gen pseudo labels
cd ../labeling
cd mmdetection
#python tools/test.py projects/CO-DETR/configs/train_set_gen.py ../../weights/CO_DETR.pth
python tools/test.py projects/CO-DETR/configs/codino/val_set_gen.py ../../weights/CO_DETR.pth


# merge pseudo labels with ground truth labels
python merge_pseudo.py --gt ../data/train/train.json --pseudo ../data/train/train_pseudo.json --out ../data/train/train_merge.json  --iou_thr 0.5 --score_thr 0.4

# convert to yolo format
python convert_yolo.py --coco_json ../data/val/test.json --output_folder ../data/val/labels
python convert_yolo.py --coco_json ../data/train/train_merge.json --output_folder ../data/train/labels
python convert_yolo.py --coco_json ../data/test/test.json --output_folder ../data/test/labels
python convert_yolo.py --coco_json ../data/train_syn/train_syn.json --output_folder ../data/train_syn/labels
python convert_yolo.py --coco_json ../data/val_syn/val_syn.json --output_folder ../data/val_syn/labels
python convert_yolo.py --coco_json ../data/visdrone_syn_enhance/vis_syn.json  --output_folder ../data/visdrone_syn_enhance/labels
python convert_yolo.py --coco_json ../data/other_dataset/other.json --output_folder ../data/other_dataset/labels