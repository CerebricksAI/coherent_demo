import cv2
import os
import queue
from typing import Callable


def extract_frames(
    video_path,
    output_folder,
    interval_seconds=1.0,
    *,
    frame_queue: queue.Queue | None = None,
    on_frame_saved: Callable[[str], None] | None = None,
):
    """Extract frames at interval_seconds. Optionally push each filename to frame_queue."""
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    sample_interval = max(1, int(fps * interval_seconds))

    frame_count = 0
    saved_count = 0

    print("Extracting frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % sample_interval == 0:
            filename = f"frame_{saved_count:04d}.jpg"
            filepath = os.path.join(output_folder, filename)
            cv2.imwrite(filepath, frame)
            saved_count += 1
            if on_frame_saved is not None:
                on_frame_saved(filename)
            if frame_queue is not None:
                frame_queue.put(filename)

        frame_count += 1

    cap.release()
    if frame_queue is not None:
        frame_queue.put(None)

    print(f"Extraction complete. Saved {saved_count} frames to '{output_folder}'.")
    return saved_count


def get_video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return frame_count / fps if fps else 0.0