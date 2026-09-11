"""
QSAID face-detection model comparison test.

Runs two pairings over every frame of the input clip:
  Pairing A: SCRFD (primary face detector) + YOLOX-s person safety net
             -> any tracked person with no overlapping face gets their head
                region flagged and redacted too.
  Pairing B: SCRFD union YOLOv11n-face (two independent face detectors,
             boxes merged, agreement noted)

For each pairing this writes an ANNOTATED video (boxes + scores, source
colour-coded, nothing redacted -> for judging detection quality) and a
REDACTED video (heavy pixelation over the same boxes -> for judging what a
viewer would actually see), plus a per-frame CSV log and a summary.

This is a detection-quality test only. It intentionally has NO tracking,
NO temporal smoothing, and uses reversible pixelation rather than the
destructive fill the compliance research requires for a real release -
see the chat writeup for why blur/pixelation alone is not production-safe.
"""
import csv
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from models import (
    ScrfdDetector,
    YoloFaceDetector,
    YoloxPersonDetector,
    box_center_inside,
    dilate_box,
    head_region_from_person,
    iou,
)

HERE = os.path.dirname(__file__)

if len(sys.argv) > 1:
    INPUT_VIDEO = sys.argv[1]
    clip_name = os.path.splitext(os.path.basename(INPUT_VIDEO))[0]
    OUTPUT_DIR = os.path.join(HERE, "..", "output", clip_name)
else:
    INPUT_VIDEO = os.path.join(HERE, "..", "input", "test_clip.mp4")
    OUTPUT_DIR = os.path.join(HERE, "..", "output", "test_clip")

FACE_CONF_THRESHOLD = 0.4      # applied on top of each model's own default
PERSON_CONF_THRESHOLD = 0.3
DILATE_FRAC = 0.20             # box padding before redaction
FACE_MATCH_IOU = 0.3           # when do two face boxes count as "the same face"
MAX_PIXELATION_BLOCKS = 6      # face region always collapses to at most this many
                               # blocks per axis, REGARDLESS of resolution - a close
                               # 4K face and a distant one both get squashed to the
                               # same coarse grid. This is what actually destroys
                               # identity; dividing box size by a constant (the old
                               # approach) does the opposite - bigger/closer faces
                               # got *more*, finer blocks and stayed recognisable.

COLOR_SCRFD = (60, 200, 60)          # green  (BGR)
COLOR_YOLOFACE = (200, 130, 40)      # blue
COLOR_BOTH_AGREE = (200, 60, 200)    # magenta
COLOR_ORPHAN_SAFETYNET = (30, 90, 235)  # orange-red


def pixelate_region(frame, box):
    x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    # Cap block count, don't derive it from box size - a face that spans 400px
    # at 4K must be squashed just as hard as one that spans 60px at 480p.
    small_w = max(1, min(MAX_PIXELATION_BLOCKS, w))
    small_h = max(1, min(MAX_PIXELATION_BLOCKS, h))
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    frame[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def draw_box(frame, box, color, label):
    x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text_y = max(12, y1 - 6)
    cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def build_pairing_a(scrfd_boxes, person_boxes, frame_w, frame_h):
    """SCRFD faces + YOLOX-s orphan-person safety net."""
    tagged = [("scrfd", b) for b in scrfd_boxes]
    orphan_count = 0
    for p in person_boxes:
        covered = any(box_center_inside(f, p) or iou(f, p) > 0.02 for f in scrfd_boxes)
        if not covered:
            head = head_region_from_person(p)
            tagged.append(("orphan_safetynet", (*head, p[4])))
            orphan_count += 1
    return tagged, orphan_count


def build_pairing_b(scrfd_boxes, yoloface_boxes):
    """Union of two independent face detectors, deduped by IoU, agreement tagged."""
    used_yf = set()
    tagged = []
    for s in scrfd_boxes:
        best_j, best_iou = -1, 0.0
        for j, y in enumerate(yoloface_boxes):
            if j in used_yf:
                continue
            i = iou(s, y)
            if i > best_iou:
                best_iou, best_j = i, j
        if best_iou >= FACE_MATCH_IOU:
            used_yf.add(best_j)
            tagged.append(("both_agree", s))
        else:
            tagged.append(("scrfd_only", s))
    for j, y in enumerate(yoloface_boxes):
        if j not in used_yf:
            tagged.append(("yoloface_only", y))
    return tagged


def color_for_tag(tag):
    return {
        "scrfd": COLOR_SCRFD,
        "scrfd_only": COLOR_SCRFD,
        "yoloface_only": COLOR_YOLOFACE,
        "both_agree": COLOR_BOTH_AGREE,
        "orphan_safetynet": COLOR_ORPHAN_SAFETYNET,
    }[tag]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading detectors...")
    t0 = time.time()
    scrfd = ScrfdDetector()
    yoloface = YoloFaceDetector(conf_threshold=FACE_CONF_THRESHOLD)
    yolox = YoloxPersonDetector(conf_threshold=PERSON_CONF_THRESHOLD)
    print(f"  loaded in {time.time() - t0:.1f}s")

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {INPUT_VIDEO}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Input: {w}x{h} @ {fps:.1f}fps, {n_frames} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = {
        name: cv2.VideoWriter(os.path.join(OUTPUT_DIR, f"{name}.mp4"), fourcc, fps, (w, h))
        for name in [
            "pairing_a_scrfd_yolox_annotated",
            "pairing_a_scrfd_yolox_redacted",
            "pairing_b_scrfd_yoloface_annotated",
            "pairing_b_scrfd_yoloface_redacted",
        ]
    }

    csv_path = os.path.join(OUTPUT_DIR, "detections_log.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "t_sec", "pairing", "source", "x1", "y1", "x2", "y2", "score"])

    stats = {
        "frames": 0,
        "scrfd_detections": 0,
        "yoloface_detections": 0,
        "person_detections": 0,
        "orphan_safetynet_triggers": 0,
        "both_agree_count": 0,
        "scrfd_only_count": 0,
        "yoloface_only_count": 0,
        "frames_with_zero_faces_a": 0,
        "frames_with_zero_faces_b": 0,
        "per_model_time_s": {"scrfd": 0.0, "yoloface": 0.0, "yolox": 0.0},
    }

    frame_idx = 0
    t_start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t = time.time()
        scrfd_boxes = scrfd.detect(frame)
        stats["per_model_time_s"]["scrfd"] += time.time() - t

        t = time.time()
        yoloface_boxes = yoloface.detect(frame)
        stats["per_model_time_s"]["yoloface"] += time.time() - t

        t = time.time()
        person_boxes = yolox.detect(frame)
        stats["per_model_time_s"]["yolox"] += time.time() - t

        stats["scrfd_detections"] += len(scrfd_boxes)
        stats["yoloface_detections"] += len(yoloface_boxes)
        stats["person_detections"] += len(person_boxes)

        # ---- Pairing A ----
        tagged_a, orphan_count = build_pairing_a(scrfd_boxes, person_boxes, w, h)
        stats["orphan_safetynet_triggers"] += orphan_count
        if not tagged_a:
            stats["frames_with_zero_faces_a"] += 1

        ann_a = frame.copy()
        red_a = frame.copy()
        for source, box in tagged_a:
            dbox = dilate_box(box, DILATE_FRAC, w, h)
            color = color_for_tag(source)
            draw_box(ann_a, dbox, color, f"{source}:{box[4]:.2f}")
            pixelate_region(red_a, dbox)
            csv_writer.writerow([frame_idx, frame_idx / fps, "A", source,
                                  *[round(v, 1) for v in dbox[:4]], round(box[4], 3)])
        writers["pairing_a_scrfd_yolox_annotated"].write(ann_a)
        writers["pairing_a_scrfd_yolox_redacted"].write(red_a)

        # ---- Pairing B ----
        tagged_b = build_pairing_b(scrfd_boxes, yoloface_boxes)
        if not tagged_b:
            stats["frames_with_zero_faces_b"] += 1

        ann_b = frame.copy()
        red_b = frame.copy()
        for source, box in tagged_b:
            if source == "both_agree":
                stats["both_agree_count"] += 1
            elif source == "scrfd_only":
                stats["scrfd_only_count"] += 1
            elif source == "yoloface_only":
                stats["yoloface_only_count"] += 1
            dbox = dilate_box(box, DILATE_FRAC, w, h)
            color = color_for_tag(source)
            draw_box(ann_b, dbox, color, f"{source}:{box[4]:.2f}")
            pixelate_region(red_b, dbox)
            csv_writer.writerow([frame_idx, frame_idx / fps, "B", source,
                                  *[round(v, 1) for v in dbox[:4]], round(box[4], 3)])
        writers["pairing_b_scrfd_yoloface_annotated"].write(ann_b)
        writers["pairing_b_scrfd_yoloface_redacted"].write(red_b)

        stats["frames"] += 1
        frame_idx += 1
        if frame_idx % 30 == 0 or frame_idx == n_frames:
            elapsed = time.time() - t_start
            print(f"  frame {frame_idx}/{n_frames}  ({elapsed:.1f}s elapsed, "
                  f"{elapsed / frame_idx * 1000:.0f} ms/frame avg)")

    cap.release()
    for wr in writers.values():
        wr.release()
    csv_file.close()

    total_time = time.time() - t_start
    stats["total_wallclock_s"] = round(total_time, 1)
    stats["ms_per_frame_avg"] = round(total_time / max(1, stats["frames"]) * 1000, 1)
    for k in stats["per_model_time_s"]:
        stats["per_model_time_s"][k] = round(stats["per_model_time_s"][k], 2)

    summary_path = os.path.join(OUTPUT_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(stats, f, indent=2)

    print()
    print("=" * 70)
    print(f"Done. {stats['frames']} frames in {total_time:.1f}s "
          f"({stats['ms_per_frame_avg']:.0f} ms/frame, CPU, no GPU)")
    print(f"  SCRFD total detections:      {stats['scrfd_detections']}")
    print(f"  YOLOv11n-face detections:    {stats['yoloface_detections']}")
    print(f"  YOLOX-s person detections:   {stats['person_detections']}")
    print(f"  Pairing A orphan-safetynet triggers (person, no matching face): "
          f"{stats['orphan_safetynet_triggers']}")
    print(f"  Pairing B agreement: both={stats['both_agree_count']} "
          f"scrfd_only={stats['scrfd_only_count']} yoloface_only={stats['yoloface_only_count']}")
    print(f"  Frames with zero flagged regions - pairing A: {stats['frames_with_zero_faces_a']}, "
          f"pairing B: {stats['frames_with_zero_faces_b']}")
    print(f"  Per-model time: {stats['per_model_time_s']}")
    print()
    print(f"Outputs written to: {OUTPUT_DIR}")
    print(f"  - detections_log.csv, summary.json")
    for name in writers:
        print(f"  - {name}.mp4")
    print("=" * 70)


if __name__ == "__main__":
    main()
