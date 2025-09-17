# AICITY2025 Track 4

End-to-end pipeline for training and converting the model used in AICITY2025 Track 4.

## Features
- Unified data preparation script
- Optional pretraining on Objects365 (object365) style data
- Final task-specific training
- Model export / conversion script
- Reproducible bash-based automation

## Repository Structure (simplified)
```
AICITY2025_track4/
  data/                 # Place extracted datasets here
  model/                # Place downloaded / converted weights here
  scripts/
    gen_data.sh
    convert_model.sh
  training/
    train_object365.py
    train.py
```

## Requirements
- Python 3.9+
- Recommended: Linux (bash). On Windows use WSL or adapt commands.
- GPU with sufficient VRAM
- Installed CUDA + compatible PyTorch

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
data/
  images/
  annotations/
  splits/
```
Adjust paths if your layout differs.

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

## 3. (Optional) Regenerate Labels / Intermediate Data
If you need to regenerate training-ready annotations:
```
cd scripts
bash gen_data.sh
```
Outputs (example):
```
data/processed/
  train.json
  val.json
```

## 4. (Optional) Pretrain on Objects365 
If you want to create a pretrained checkpoint before final training:
```
cd training
python train_object365.py \
  --data ../data/processed/object365.json \
  --output ../model/object365_pretrained.pth
```
Arguments may vary; check the script's parser (`--help`).

## 5. Final Training
Uses either the freshly produced object365 pretrained weights or a backbone.
```
cd training
python train.py \
  --data ../data/processed/train.json \
  --val  ../data/processed/val.json \
  --pretrained ../model/object365_pretrained.pth \
  --output ../model/final_model.pth
```

## 6. Convert / Export Model
After training:
```
cd scripts
bash convert_model.sh \
  ../model/final_model.pth \
  ../model/export/
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


