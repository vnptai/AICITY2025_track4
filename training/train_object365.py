from ultralytics import YOLO

# Load a model
model = YOLO("yolo11m.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="datasets/Objects365/data.yaml", imgsz=640, batch=64, conf=0.25, iou=0.6, plots=True)