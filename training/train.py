from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11x.pt")

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="coco8.yaml", epochs=1, imgsz=640)
