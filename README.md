# AICITY2025 Track 4

End-to-end pipeline for training and converting the model used in AICITY2025 Track 4.


## Installation
```
git clone <this_repo_url>
cd AICITY2025_track4
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # If provided
```
(If `requirements.txt` not yet created, install deps referenced inside training scripts.)

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
<MODEL_DOWNLOAD_LINK>

Extract into:
```
AICITY2025_track4/weights/
```
Example:
```
weights/
  backbone.pth
  object365_pretrained.pth   # (will be produced later if you pretrain)
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

## 6. Convert / Export Model
After training:
```
cd scripts
bash convert_model.sh
```
Outputs might include ONNX / TorchScript versions (depends on script implementation).

## 7. Inference (Example)
(Adjust once an inference script exists.)
```
python inference.py \
  --weights model/final_model.pth \
  --source data/images/sample.jpg \
  --out outputs/
```


