from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # load pretrained model
results = model.predict("test_image.jpg", show=True)