"""
Re-render just the redacted output videos from an already-logged
detections_log.csv, without re-running any detection model.

Box coordinates in the CSV are already dilated (that step happened once,
at detection time), so this only needs to re-apply the pixelation function
to fresh frames from the source video - useful for fixing a rendering bug
without paying the multi-minute cost of re-running SCRFD/YOLOX-s/YOLOv11n-face.
"""
import csv
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(__file__))
from run_test import pixelate_region  # reuses the fixed function directly

HERE = os.path.dirname(__file__)


def rerender(clip_dir_name, input_video):
    out_dir = os.path.join(HERE, "..", "output", clip_dir_name)
    csv_path = os.path.join(out_dir, "detections_log.csv")

    rows = list(csv.DictReader(open(csv_path)))
    by_pairing_frame = {"A": {}, "B": {}}
    for r in rows:
        f = int(r["frame"])
        box = (float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]), float(r["score"]))
        by_pairing_frame[r["pairing"]].setdefault(f, []).append(box)

    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = {
        "A": cv2.VideoWriter(os.path.join(out_dir, "pairing_a_scrfd_yolox_redacted.mp4"), fourcc, fps, (w, h)),
        "B": cv2.VideoWriter(os.path.join(out_dir, "pairing_b_scrfd_yoloface_redacted.mp4"), fourcc, fps, (w, h)),
    }

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        for pairing in ("A", "B"):
            out_frame = frame.copy()
            for box in by_pairing_frame[pairing].get(frame_idx, []):
                pixelate_region(out_frame, box)
            writers[pairing].write(out_frame)
        frame_idx += 1
        if frame_idx % 60 == 0 or frame_idx == n_frames:
            print(f"  {frame_idx}/{n_frames}")

    cap.release()
    for wr in writers.values():
        wr.release()
    print(f"Re-rendered {frame_idx} frames -> {out_dir}")


if __name__ == "__main__":
    clip_dir_name = sys.argv[1]
    input_video = sys.argv[2]
    rerender(clip_dir_name, input_video)
