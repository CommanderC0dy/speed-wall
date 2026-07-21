"""Teach the judge about a boulder: mark its start and finish hold once.

Usage:  .venv/bin/python mark.py wall_photo.jpg

Click the START hold, then the FINISH hold, then press S to save.
Writes wall_photo.holds.json next to the photo — that file plus the photo
is everything judge.py needs to time any submitted video of this boulder.

Keys:  S = save,  R = reset clicks,  Q = quit without saving
"""

import json
import sys
from pathlib import Path

import cv2

if len(sys.argv) != 2:
    raise SystemExit(__doc__)

photo = Path(sys.argv[1])
img = cv2.imread(str(photo))
if img is None:
    raise SystemExit(f"Could not read image: {photo}")

# Radius of the hold zone, in reference-photo pixels (scales automatically
# to each submitted video via the homography).
RADIUS = max(40, img.shape[1] // 25)

holds = []


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(holds) < 2:
        holds.append((x, y))


WIN = "Mark holds: click START, then FINISH, then press S"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WIN, on_mouse)

while True:
    view = img.copy()
    for i, h in enumerate(holds):
        color = (0, 255, 0) if i == 0 else (0, 0, 255)
        cv2.circle(view, h, RADIUS, color, 3)
        cv2.putText(view, "START" if i == 0 else "FINISH",
                    (h[0] - 50, h[1] - RADIUS - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    cv2.imshow(WIN, view)
    key = cv2.waitKey(30) & 0xFF
    if key in (ord("r"), ord("R")):
        holds.clear()
    elif key in (ord("q"), ord("Q")):
        break
    elif key in (ord("s"), ord("S")) and len(holds) == 2:
        out = photo.with_suffix(".holds.json")
        out.write_text(json.dumps({
            "image": photo.name,
            "start": list(holds[0]),
            "finish": list(holds[1]),
            "radius": RADIUS,
        }, indent=2))
        print(f"Saved {out}")
        break

cv2.destroyAllWindows()
