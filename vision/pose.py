"""Pose-tracking speedrun timer prototype.

Draws a live skeleton on anyone in view and tracks their wrists.
Click once to place the START hold, click again to place the FINISH hold.

Timer logic, same as a real speed wall:
  - put a hand on the start hold  -> READY
  - hand leaves the start hold    -> timer starts
  - hand touches the finish hold  -> timer stops

Keys:  R = reset holds and timer,  Q = quit
Run with:  .venv/bin/python pose.py
"""

import time

import cv2
from ultralytics import YOLO

# COCO keypoint indices for wrists
L_WRIST, R_WRIST = 9, 10
HOLD_RADIUS = 45  # px — how close a wrist must be to count as "on the hold"

model = YOLO("yolo11n-pose_openvino_model/", task="pose")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Could not open camera 0 — try VideoCapture(1)")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

holds = []  # [(x, y) start, (x, y) finish]
state = "PLACE HOLDS"  # -> WAITING -> READY -> CLIMBING -> FINISHED
t_start = t_final = 0.0


def on_mouse(event, x, y, flags, param):
    global state
    if event == cv2.EVENT_LBUTTONDOWN and len(holds) < 2:
        holds.append((x, y))
        if len(holds) == 2:
            state = "WAITING"


WIN = "Speedrun timer - click start hold, then finish hold"
cv2.namedWindow(WIN)
cv2.setMouseCallback(WIN, on_mouse)
print("Click the START hold, then the FINISH hold. R resets, Q quits.")


def wrists_of(result):
    """Confidently-seen wrist positions of everyone in frame, as (x, y)."""
    pts = []
    kp = result.keypoints
    if kp is not None and kp.xy is not None:
        for pi, person in enumerate(kp.xy):
            for idx in (L_WRIST, R_WRIST):
                if kp.conf is not None and float(kp.conf[pi][idx]) < 0.5:
                    continue  # occluded/guessed keypoint
                x, y = person[idx].tolist()
                if x > 0 or y > 0:  # (0,0) means "not visible"
                    pts.append((x, y))
    return pts


def near(pt, hold):
    return (pt[0] - hold[0]) ** 2 + (pt[1] - hold[1]) ** 2 < HOLD_RADIUS**2


while True:
    ok, frame = cap.read()
    if not ok:
        break

    result = model(frame, imgsz=320, verbose=False)[0]
    frame = result.plot(boxes=False)  # skeleton only, no bounding boxes
    wrists = wrists_of(result)

    # --- timer state machine ---
    if state == "WAITING" and any(near(w, holds[0]) for w in wrists):
        state = "READY"
    elif state == "READY" and not any(near(w, holds[0]) for w in wrists):
        state, t_start = "CLIMBING", time.perf_counter()
    elif state == "CLIMBING" and any(near(w, holds[1]) for w in wrists):
        state, t_final = "FINISHED", time.perf_counter() - t_start

    # --- drawing ---
    for i, hold in enumerate(holds):
        color = (0, 255, 0) if i == 0 else (0, 0, 255)  # green start, red finish
        cv2.circle(frame, hold, HOLD_RADIUS, color, 3)
        cv2.putText(frame, "START" if i == 0 else "FINISH",
                    (hold[0] - 40, hold[1] - HOLD_RADIUS - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    for w in wrists:
        cv2.circle(frame, (int(w[0]), int(w[1])), 8, (255, 255, 0), -1)

    if state == "CLIMBING":
        clock = f"{time.perf_counter() - t_start:6.2f}"
    elif state == "FINISHED":
        clock = f"{t_final:6.2f}"
    else:
        clock = "  0.00"
    cv2.putText(frame, clock, (frame.shape[1] - 220, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3)
    cv2.putText(frame, state, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    cv2.imshow(WIN, frame)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), ord("Q")):
        break
    if key in (ord("r"), ord("R")):
        holds, state = [], "PLACE HOLDS"

cap.release()
cv2.destroyAllWindows()
