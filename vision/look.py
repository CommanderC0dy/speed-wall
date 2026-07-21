"""Live object recognition through the webcam.

Opens your camera, runs YOLO on every frame, and draws labeled boxes
around everything it recognizes. Press Q to quit.

Run with:  .venv/bin/python look.py
"""

import cv2
from ultralytics import YOLO

# OpenVINO export of the nano model, tuned for Intel CPUs (~10 fps here
# vs 0.6 fps for the plain PyTorch version). Recreate it with:
#   .venv/bin/yolo export model=yolo11n.pt format=openvino imgsz=320
model = YOLO("yolo11n_openvino_model/", task="detect")

cap = cv2.VideoCapture(0)  # /dev/video0 — change to 1 for the other camera
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always show the freshest frame, no lag
if not cap.isOpened():
    raise SystemExit("Could not open camera 0 — try VideoCapture(1)")

print("Camera open. Press Q in the video window to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # Run detection; results[0].plot() returns the frame with boxes drawn on
    results = model(frame, imgsz=320, verbose=False)
    annotated = results[0].plot()

    cv2.imshow("What I see - press Q to quit", annotated)
    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
        break

cap.release()
cv2.destroyAllWindows()
