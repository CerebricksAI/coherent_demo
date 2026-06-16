import json
import os
import subprocess

import imageio_ffmpeg

from azure_client import get_whisper_client, get_whisper_endpoint, get_whisper_model


def extract_audio(video_path: str, audio_path: str) -> bool:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            audio_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr or ""
        if "does not contain any stream" in stderr or "Output file is empty" in stderr:
            return False
        raise RuntimeError(f"ffmpeg failed to extract audio:\n{stderr}")
    return os.path.getsize(audio_path) > 0


def transcribe_with_whisper(audio_path: str) -> dict:
    """Transcribe audio with Azure OpenAI Whisper."""
    client = get_whisper_client()
    model = get_whisper_model()
    endpoint = get_whisper_endpoint()
    print(f"Using Whisper '{model}' @ {endpoint}")

    with open(audio_path, "rb") as audio_file:
        try:
            response = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        except TypeError:
            audio_file.seek(0)
            response = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format="verbose_json",
            )
        except Exception:
            audio_file.seek(0)
            text = client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format="text",
            )
            return {"text": text.strip(), "segments": []}

    if isinstance(response, str):
        return {"text": response.strip(), "segments": []}

    segments = []
    for segment in getattr(response, "segments", None) or []:
        if isinstance(segment, dict):
            segments.append(
                {
                    "start": segment.get("start"),
                    "end": segment.get("end"),
                    "text": segment.get("text", "").strip(),
                }
            )
        else:
            segments.append(
                {
                    "start": getattr(segment, "start", None),
                    "end": getattr(segment, "end", None),
                    "text": getattr(segment, "text", "").strip(),
                }
            )

    text = getattr(response, "text", None) or ""
    if not text and segments:
        text = " ".join(s["text"] for s in segments if s["text"])

    return {"text": text.strip(), "segments": segments}


def transcribe_video(video_path: str, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    audio_path = os.path.join(output_dir, "audio.wav")
    result = {"text": "", "segments": []}

    print("Extracting audio from video...")
    has_audio = extract_audio(video_path, audio_path)
    if not has_audio:
        print("No audio track found in video. Continuing with visual analysis only.")
    else:
        print("Transcribing audio with Whisper...")
        try:
            result = transcribe_with_whisper(audio_path)
        except Exception as exc:
            print(f"WARNING: Whisper transcription failed ({exc}). Continuing without transcript.")

    transcript_path = os.path.join(output_dir, "transcript.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(result["text"])

    segments_path = os.path.join(output_dir, "transcript_segments.json")
    with open(segments_path, "w", encoding="utf-8") as f:
        json.dump(result["segments"], f, indent=2)

    print(f"Transcript saved to '{transcript_path}'")
    return result
