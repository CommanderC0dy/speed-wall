"""Ghost racing: overlay two runs of the same boulder into one video.

Usage:
  .venv/bin/python ghost.py wall.holds.json run.mp4 ghost.mp4 [--align frame0]

Renders from RUN's viewpoint (the new attempt). GHOST (e.g. the record
holder's video) is warped in through the shared reference wall as a
translucent magenta skeleton — like a ghost car in a racing game.

The two runs are time-aligned at the moment each climber's hand leaves the
start hold, so the ghost shows exactly where the other climber was at the
same race time, no matter when either video starts. Both videos can be
filmed from completely different angles and phones.

--align frame0 skips start detection and aligns both videos at frame 0
(useful for testing, or for manually trimmed clips).

Output: ghost_<run>_vs_<ghost>.mp4 with both clocks and the final gap.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from judge import MIN_KP_CONF, REMATCH_EVERY, WallLocator, project, wrists_of

# COCO-17 skeleton edges for drawing
EDGES = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (11, 12), (5, 11), (6, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]

RUN_COLOR = (80, 220, 80)      # green
GHOST_COLOR = (255, 80, 255)   # magenta
GHOST_ALPHA = 0.55


def analyze(vid_path, locator, model):
    """Pass 1: per frame, wall homography + main person's keypoints + wrists."""
    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video {vid_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames, H = [], None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if H is None or i % REMATCH_EVERY == 0:
            H_new = locator.find(frame)
            if H_new is not None:
                H = H_new
        res = model(frame, imgsz=640, verbose=False)[0]
        main = None
        if res.boxes is not None and len(res.boxes) and res.keypoints is not None:
            j = int(res.boxes.conf.argmax())
            main = np.array(res.keypoints.xy[j].tolist())  # 17x2, (0,0) = unseen
            if res.keypoints.conf is not None:  # drop guessed/occluded keypoints
                main[np.array(res.keypoints.conf[j].tolist()) < MIN_KP_CONF] = 0
        frames.append({"H": None if H is None else H.copy(),
                       "kpts": main, "wrists": wrists_of(res)})
        i += 1
        if i % 100 == 0:
            print(f"    frame {i}")
    cap.release()
    return fps, frames


def hold_geometry(H, cfg):
    """Projected (start, finish, radius) for a frame's homography."""
    start = project(H, cfg["start"])
    finish = project(H, cfg["finish"])
    edge = project(H, (cfg["start"][0] + cfg["radius"], cfg["start"][1]))
    r = max(15.0, float(np.hypot(edge[0] - start[0], edge[1] - start[1])))
    return start, finish, r


def find_run(frames, cfg):
    """Start/end frame indices via the usual state machine, or (None, None)."""
    state, start_i, end_i = "WAITING", None, None
    for i, f in enumerate(frames):
        if f["H"] is None:
            continue
        start, finish, r = hold_geometry(f["H"], cfg)

        def near(p, hold):
            return (p[0] - hold[0]) ** 2 + (p[1] - hold[1]) ** 2 < r * r

        if state == "WAITING" and any(near(p, start) for p in f["wrists"]):
            state = "READY"
        elif state == "READY" and not any(near(p, start) for p in f["wrists"]):
            state, start_i = "CLIMBING", i
        elif state == "CLIMBING" and any(near(p, finish) for p in f["wrists"]):
            return start_i, i
    return start_i, None


def draw_skeleton(img, kpts, color, thickness=3):
    def ok(p):
        return p[0] > 0 or p[1] > 0
    for a, b in EDGES:
        if ok(kpts[a]) and ok(kpts[b]):
            cv2.line(img, (int(kpts[a][0]), int(kpts[a][1])),
                     (int(kpts[b][0]), int(kpts[b][1])), color, thickness)
    for p in kpts:
        if ok(p):
            cv2.circle(img, (int(p[0]), int(p[1])), 5, color, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("run_video")
    ap.add_argument("ghost_video")
    ap.add_argument("--align", choices=["auto", "frame0"], default="auto")
    args = ap.parse_args()

    cfg = json.loads(Path(args.reference).read_text())
    ref_img = cv2.imread(str(Path(args.reference).parent / cfg["image"]))
    if ref_img is None:
        raise SystemExit(f"Could not read reference image {cfg['image']}")
    locator = WallLocator(ref_img)
    model = YOLO("yolo11n-pose-640_openvino_model/", task="pose")

    print("Pass 1/2: analyzing run video…")
    fps_a, A = analyze(args.run_video, locator, model)
    print("Pass 1/2: analyzing ghost video…")
    fps_b, B = analyze(args.ghost_video, locator, model)

    if args.align == "frame0":
        start_a = start_b = 0
        end_a = end_b = None
    else:
        start_a, end_a = find_run(A, cfg)
        start_b, end_b = find_run(B, cfg)
        for name, s in (("run", start_a), ("ghost", start_b)):
            if s is None:
                raise SystemExit(f"Could not detect the start in the {name} "
                                 "video — no hand seen leaving the start hold. "
                                 "Try --align frame0 with pre-trimmed clips.")

    t_a = None if (end_a is None or start_a is None) else (end_a - start_a) / fps_a
    t_b = None if (end_b is None or start_b is None) else (end_b - start_b) / fps_b

    print("Pass 2/2: rendering…")
    cap = cv2.VideoCapture(args.run_video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    run_p, ghost_p = Path(args.run_video), Path(args.ghost_video)
    out_path = run_p.parent / f"ghost_{run_p.stem}_vs_{ghost_p.stem}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps_a, (w, h))

    first = max(0, start_a - int(2 * fps_a))  # 2 s lead-in before the start
    for idx in range(len(A)):
        ok, frame = cap.read()
        if not ok:
            break
        if idx < first:
            continue
        fa = A[idx]

        # ghost frame at the same race time (aligned on the two start moments)
        gi = start_b + round((idx - start_a) * fps_b / fps_a)
        fb = B[gi] if 0 <= gi < len(B) else None

        if fa["H"] is not None:
            start, finish, r = hold_geometry(fa["H"], cfg)
            for hold, color in ((start, (0, 255, 0)), (finish, (0, 0, 255))):
                cv2.circle(frame, (int(hold[0]), int(hold[1])), int(r), color, 2)

            if fb is not None and fb["H"] is not None and fb["kpts"] is not None:
                try:
                    M = fa["H"] @ np.linalg.inv(fb["H"])
                except np.linalg.LinAlgError:
                    M = None
                if M is not None:
                    ghost_k = np.array([
                        project(M, p) if (p[0] > 0 or p[1] > 0) else (0, 0)
                        for p in fb["kpts"]])
                    overlay = frame.copy()
                    draw_skeleton(overlay, ghost_k, GHOST_COLOR, 4)
                    frame = cv2.addWeighted(overlay, GHOST_ALPHA, frame,
                                            1 - GHOST_ALPHA, 0)

        if fa["kpts"] is not None:
            draw_skeleton(frame, fa["kpts"], RUN_COLOR)

        # clocks: clamp at each climber's own finish
        def clock(idx_now, start_i, end_i, fps):
            stop = idx_now if end_i is None else min(idx_now, end_i)
            return max(0.0, (stop - start_i) / fps)

        ta = clock(idx, start_a, end_a, fps_a)
        tb = clock(gi, start_b, end_b, fps_b)
        cv2.putText(frame, f"RUN   {ta:6.2f}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, RUN_COLOR, 3)
        cv2.putText(frame, f"GHOST {tb:6.2f}", (20, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, GHOST_COLOR, 3)
        if t_a is not None and t_b is not None and idx >= max(end_a, start_a):
            gap = t_a - t_b
            cv2.putText(frame, f"{'+' if gap >= 0 else ''}{gap:.2f} s vs ghost",
                        (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        (255, 255, 255), 3)
        writer.write(frame)

    cap.release()
    writer.release()
    print(f"\nWrote {out_path}")
    if t_a is not None:
        print(f"Run:   {t_a:.2f} s")
    if t_b is not None:
        print(f"Ghost: {t_b:.2f} s")


if __name__ == "__main__":
    main()
