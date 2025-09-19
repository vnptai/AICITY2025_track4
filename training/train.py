from ultralytics import YOLO

model = YOLO("../weights/yolo11m-obj365.pt")

train_results = model.train(
    data="../data/full-pipeline.yaml",  # Path to dataset configuration file
    epochs=100,  # Number of` training epochs
    imgsz=960,  # Image size for training
    device=[0,1],  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    batch=16,
    seed=0,
    val=True,
    project="track4",
    name="final",
    mosaic=1,
    translate=0,
    flipud=0,
    fliplr=0.5,
    warmup_epochs=0,
    workers=8,
    replace=0.0,
    save_period=5,
)