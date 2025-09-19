# AICITY2025 Track 4

End-to-end pipeline for training and converting the model used in AICITY2025 Track 4.

## 1. Download Data
Download the dataset archive(s) from:
<DATA_DOWNLOAD_LINK>

Extract them into:
```
AICITY2025_track4/data/
```
Expected (example):

```
AICITY2025_track4/
  data/                 
    other_datasets/
        images/
        labels/
    test/
        images/
        labels/
    train/
        images/
        labels/
    train_syn/
        images/
        labels/
    val/
        images/
        labels/
    val_syn/
        images/
        labels/
    visdrone_syn_enhanced/
        images/
        labels/
 
```

## 2. Download Pretrained Model
Download from:
[weight link](https://1drv.ms/u/c/3f64f78089759ca1/EXtAbIvc1N5FtYTKH18T5bcBoLbjzCYnXK7fGAMbM6PXPg?e=eHNeAC)

Extract into:
```
AICITY2025_track4/weights/
```
Example:
```
weights/
  CO_DETR.pth
  yolo11m-obj365.pt   # (object365 pretrained weights)
  yolo11m.engine
```

## 3. (Optional) Regenerate Labels 
If you need to regenerate training-ready annotations:
```
cd scripts
bash gen_data.sh
```

## 4. (Optional) Pretrain on Objects365 
If you want to create a pretrained checkpoint before final training:
```
cd training
python train_object365.py
```

## 5. Final Training
Uses either the freshly produced object365 pretrained weights or a backbone.
```
cd training
python train.py
```

## 6. Convert and Inference on Jetson
After training:
```
cd scripts
bash convert_model.sh
```



