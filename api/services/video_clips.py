import os
import subprocess

import imageio_ffmpeg


def cut_video_clip(
    source_path: str,
    output_path: str,
    start_sec: float,
    end_sec: float,
) -> bool:
    """Cut a segment from source video using ffmpeg. Returns True on success."""
    if end_sec <= start_sec:
        return False

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    duration = end_sec - start_sec

    # Re-encode for reliable clip boundaries (stream copy can fail at non-keyframes)
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-ss",
            str(max(0.0, start_sec)),
            "-i",
            source_path,
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            output_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        # Fallback without audio
        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-ss",
                str(max(0.0, start_sec)),
                "-i",
                source_path,
                "-t",
                str(duration),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "23",
                "-an",
                "-movflags",
                "+faststart",
                output_path,
            ],
            capture_output=True,
            text=True,
        )
    return result.returncode == 0 and os.path.isfile(output_path) and os.path.getsize(output_path) > 0
