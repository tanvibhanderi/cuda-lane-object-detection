from ultralytics import YOLO
import cv2
import time
import torch

model = YOLO("yolov8n.pt")

if not torch.cuda.is_available():
    print("GPU not available")
    exit()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not opened")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()

    results = model(frame, device=0)

    end = time.time()
    fps = 1 / (end - start)

    annotated = results[0].plot()

    cv2.putText(annotated, f"GPU FPS: {fps:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("GPU Detection", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
