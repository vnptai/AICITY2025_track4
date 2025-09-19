cd ../inference/yolo11
python gen_wts.py -w ../../weights/yolo11m.pt -o yolo11m.wts
cd ../
bash infer.sh