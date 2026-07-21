"""Time a submitted climbing video against a marked reference photo.

Usage:  .venv/bin/python judge.py wall_photo.holds.json submitted_video.mp4

How it works, per frame:
  1. SIFT feature matching finds the reference wall in the frame and computes
     a homography (works from any camera angle — the wall is roughly flat).
  2. The marked start/finish holds are projected into the frame through it.
  3. YOLO pose tracks every person's wrists.
  4. Same state machine as the live timer, but on video timestamps:
     hand on start -> READY, hand leaves -> clock runs, hand on finish -> stop.

Prints the verdict and writes judged_<video>.mp4 with everything drawn on,
so you can verify the call before putting it on the leaderboard.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

L_WRIST, R_WRIST = 9, 10
MATCH_WIDTH = 960       # frames are downscaled to this width for SIFT matching
MIN_INLIERS = 15        # fewer matched points than this = wall not found
REMATCH_EVERY = 5       # recompute the homography every N frames


class WallLocator:
    """Finds the reference wall in arbitrary video frames."""

    def __init__(self, ref_img):
        self.sift = cv2.SIFT_create(nfeatures=3000)
        self.matcher = cv2.BFMatcher()
        scale = min(1.0, MATCH_WIDTH / ref_img.shape[1])
        self.ref_scale = scale
        small = cv2.resize(ref_img, None, fx=scale, fy=scale)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        self.ref_kp, self.ref_des = self.sift.detectAndCompute(gray, None)

    def find(self, frame):
        """Homography mapping reference-photo pixels -> frame pixels, or None."""
        scale = min(1.0, MATCH_WIDTH / frame.shape[1])
        small = cv2.resize(frame, None, fx=scale, fy=scale)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(gray, None)
        if des is None or len(kp) < MIN_INLIERS:
            return None
        good = [m for m, n in self.matcher.knnMatch(self.ref_des, des, k=2)
                if m.distance < 0.75 * n.distance]
        if len(good) < MIN_INLIERS:
            return None
        src = np.float32([self.ref_kp[m.queryIdx].pt for m in good])
        dst = np.float32([kp[m.trainIdx].pt for m in good])
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None or mask.sum() < MIN_INLIERS:
            return None
        # compose: ref full-res -> ref small -> frame small -> frame full-res
        pre = np.diag([self.ref_scale, self.ref_scale, 1.0])
        post = np.diag([1 / scale, 1 / scale, 1.0])
        return post @ H @ pre


def project(H, pt):
    p = H @ np.array([pt[0], pt[1], 1.0])
    return (float(p[0] / p[2]), float(p[1] / p[2]))


MIN_KP_CONF = 0.5  # keypoints below this are guesses (e.g. body parts out of frame)


def wrists_of(result):
    pts = []
    kp = result.keypoints
    if kp is not None and kp.xy is not None:
        for pi, person in enumerate(kp.xy):
            for idx in (L_WRIST, R_WRIST):
                if kp.conf is not None and float(kp.conf[pi][idx]) < MIN_KP_CONF:
                    continue
                x, y = person[idx].tolist()
                if x > 0 or y > 0:
                    pts.append((x, y))
    return pts


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    ref_path, vid_path = Path(sys.argv[1]), Path(sys.argv[2])

    cfg = json.loads(ref_path.read_text())
    ref_img = cv2.imread(str(ref_path.parent / cfg["image"]))
    if ref_img is None:
        raise SystemExit(f"Could not read reference image {cfg['image']}")
    locator = WallLocator(ref_img)

    model = YOLO("yolo11n-pose-640_openvino_model/", task="pose")

    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video {vid_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = vid_path.parent / f"judged_{vid_path.stem}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))

    state = "WAITING"          # -> READY -> CLIMBING -> FINISHED
    t_start = t_end = None
    H = None
    lost_frames = 0

    for i in range(n_frames if n_frames > 0 else 10**9):
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps

        if H is None or i % REMATCH_EVERY == 0:
            H_new = locator.find(frame)
            if H_new is not None:
                H = H_new
            else:
                lost_frames += 1

        result = model(frame, imgsz=640, verbose=False)[0]
        frame = result.plot(boxes=False)
        wrists = wrists_of(result)

        if H is not None:
            start = project(H, cfg["start"])
            finish = project(H, cfg["finish"])
            # hold radius in frame pixels: project a point radius-away and measure
            edge = project(H, (cfg["start"][0] + cfg["radius"], cfg["start"][1]))
            r = max(15.0, np.hypot(edge[0] - start[0], edge[1] - start[1]))

            def near(pt, hold):
                return (pt[0] - hold[0]) ** 2 + (pt[1] - hold[1]) ** 2 < r ** 2

            if state == "WAITING" and any(near(p, start) for p in wrists):
                state = "READY"
            elif state == "READY" and not any(near(p, start) for p in wrists):
                state, t_start = "CLIMBING", t
            elif state == "CLIMBING" and any(near(p, finish) for p in wrists):
                state, t_end = "FINISHED", t

            for hold, color, label in ((start, (0, 255, 0), "START"),
                                       (finish, (0, 0, 255), "FINISH")):
                c = (int(hold[0]), int(hold[1]))
                cv2.circle(frame, c, int(r), color, 3)
                cv2.putText(frame, label, (c[0] - 45, c[1] - int(r) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            cv2.putText(frame, "WALL NOT FOUND", (20, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        if state == "CLIMBING":
            clock = f"{t - t_start:6.2f}"
        elif state == "FINISHED":
            clock = f"{t_end - t_start:6.2f}"
        else:
            clock = "  0.00"
        cv2.putText(frame, clock, (w - 260, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)
        cv2.putText(frame, state, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        writer.write(frame)

        if i % 100 == 0 and n_frames > 0:
            print(f"  frame {i}/{n_frames} ({state})")

    cap.release()
    writer.release()

    print(f"\nAnnotated video: {out_path}")
    if state == "FINISHED":
        print(f"VERDICT: {t_end - t_start:.2f} s  "
              f"(start {t_start:.2f}s -> finish {t_end:.2f}s in the video)")
    elif state == "CLIMBING":
        print("VERDICT: started but never touched the finish hold — no time.")
    elif state == "READY":
        print("VERDICT: hand on start hold but never launched — no time.")
    else:
        print("VERDICT: never saw a hand on the start hold — no time. "
              f"(wall not found in {lost_frames} match attempts)")


if __name__ == "__main__":
    main()
