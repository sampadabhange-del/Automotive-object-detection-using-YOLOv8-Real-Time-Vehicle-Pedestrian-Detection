# main.py
# Real-Time Automotive Object Detection using YOLOv8

from ultralytics import YOLO
import cv2

# Load YOLO model (small version for fast performance)
model = YOLO("yolov8s.pt")  # Automatically downloads the model if not found

# Select input: 0 = webcam. Replace '0' with 'traffic.mp4' for video
cap = cv2.VideoCapture(0)  # Webcam input
# cap = cv2.VideoCapture("traffic.mp4")  # Uncomment to use a video file

if not cap.isOpened():
    print("Error: Could not open video source")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video stream ended or error.")
        break

    # Run YOLO inference
    results = model(frame)

    # Annotate frame with results (bounding boxes, labels, confidence)
    annotated_frame = results[0].plot()

    # Display output
    cv2.imshow("Automotive Object Detection (YOLOv8)", annotated_frame)

    # Press 'q' to quit window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
