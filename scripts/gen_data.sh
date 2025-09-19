
# gen pseudo labels
cd ../labeling
cd mmdetection
python tools/test.py projects/CO-DETR/configs/codino/train_set_gen.py ../../weights/CO_DETR.pth
python tools/test.py projects/CO-DETR/configs/codino/val_set_gen.py ../../weights/CO_DETR.pth
python tools/test.py projects/CO-DETR/configs/codino/train_syn_set_gen.py ../../weights/CO_DETR.pth
python tools/test.py projects/CO-DETR/configs/codino/val_syn_set_gen.py ../../weights/CO_DETR.pth
python tools/test.py projects/CO-DETR/configs/codino/other_dataset_gen.py ../../weights/CO_DETR.pth
python tools/test.py projects/CO-DETR/configs/codino/visdrone_syn_enhance_gen.py ../../weights/CO_DETR.pth

# merge pseudo labels with ground truth labels
cd ../
python merge_pseudo.py --gt ../data/train/train.json --pseudo ../data/train/train_gen.json --out ../data/train/train_merge.json  --iou_thr 0.5 --score_thr 0.4
python merge_pseudo.py --gt ../data/val/test.json --pseudo ../data/val/val_gen.json --out ../data/val/val_merge.json  --iou_thr 0.5 --score_thr 0.4
python merge_pseudo.py --gt ../data/train_syn/train_syn.json --pseudo ../data/train_syn/train_syn_gen.json --out ../data/val/train_syn_merge.json  --iou_thr 0.5 --score_thr 0.4
python merge_pseudo.py --gt ../data/val_syn/val_syn.json --pseudo ../data/val_syn/val_syn_gen.json --out ../data/val/val_syn_merge.json  --iou_thr 0.5 --score_thr 0.4
python merge_pseudo.py --gt ../data/other_dataset/other_dataset.json --pseudo ../data/other_dataset/other_dataset_gen.json --out ../data/val/other_dataset_merge.json  --iou_thr 0.5 --score_thr 0.4
python merge_pseudo.py --gt ../data/visdrone_syn_enhance/visdrone_syn_enhance.json --pseudo ../data/visdrone_syn_enhance/visdrone_syn_enhance_gen.json --out ../data/val/visdrone_syn_enhance_merge.json  --iou_thr 0.5 --score_thr 0.4

# convert to yolo format
python convert_yolo.py --coco_json ../data/val/val_merge.json --output_folder ../data/val/labels
python convert_yolo.py --coco_json ../data/train/train_merge.json --output_folder ../data/train/labels
python convert_yolo.py --coco_json ../data/test/test.json --output_folder ../data/test/labels
python convert_yolo.py --coco_json ../data/train_syn/train_syn_merge.json --output_folder ../data/train_syn/labels
python convert_yolo.py --coco_json ../data/val_syn/val_syn_merge.json --output_folder ../data/val_syn/labels
python convert_yolo.py --coco_json ../data/visdrone_syn_enhance/visdrone_syn_enhance_merge.json  --output_folder ../data/visdrone_syn_enhance/labels
python convert_yolo.py --coco_json ../data/other_dataset/other_dataset_merge.json --output_folder ../data/other_dataset/labels
