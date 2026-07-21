# Vision tools

AI timing tools for the speed wall. One-time setup:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv ultralytics opencv-python openvino
.venv/bin/yolo export model=yolo11n.pt format=openvino imgsz=320
.venv/bin/yolo export model=yolo11n-pose.pt format=openvino imgsz=320
.venv/bin/yolo export model=yolo11n-pose.pt format=openvino imgsz=640
mv yolo11n-pose_openvino_model yolo11n-pose-640_openvino_model   # after the 640 export
```

(the 320 pose export must end up named `yolo11n-pose_openvino_model`,
the 640 one `yolo11n-pose-640_openvino_model` — export twice, rename between)

## Tools

- `look.py` — webcam object recognition demo. `.venv/bin/python look.py`
- `pose.py` — live speedrun timer with skeleton tracking; click start/finish holds.
- `mark.py` — teach the judge a boulder: `mark.py wall_photo.jpg`, click the
  start and finish hold, press S. Writes `wall_photo.holds.json`.
- `judge.py` — time a submitted video: `judge.py wall_photo.holds.json video.mp4`.
  Prints the verdict and writes `judged_<video>.mp4` as proof.
- `ghost.py` — ghost racing: `ghost.py wall.holds.json run.mp4 ghost.mp4`.
  Overlays the ghost video's climber (magenta skeleton, warped through the
  shared wall from any camera angle) onto the run video, time-aligned at the
  moment each hand leaves the start hold. `--align frame0` for trimmed clips.

The phone version of the live timer is `/timer.html` on the site
(browser pose tracking via MediaPipe, needs HTTPS for camera access).
