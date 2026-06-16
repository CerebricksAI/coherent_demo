import json
import os
import re
from typing import Callable

from audio_transcription import transcribe_video
from azure_client import get_client, get_writer_model, print_model_config, validate_azure_config
from key_frame_generator import extract_frames, get_video_duration
from report_builder import build_report
from sop_stateless import analyze_frames, consolidate_sequence
from time_utils import seconds_to_timestamp


def _frame_index(filename: str) -> int:
    match = re.search(r"frame_(\d+)", filename)
    return int(match.group(1)) if match else 0


def _format_visual_steps(sequence: list, interval_seconds: float) -> str:
    lines = []
    for step in sequence:
        start_idx = _frame_index(step["start"])
        end_idx = _frame_index(step["end"])
        start_time = start_idx * interval_seconds
        end_time = (end_idx + 1) * interval_seconds
        lines.append(
            f"- [{seconds_to_timestamp(start_time)} - {seconds_to_timestamp(end_time)}] "
            f"{step['action']} (frames {step['start']} to {step['end']})"
        )
    return "\n".join(lines)


def _format_transcript_segments(segments: list) -> str:
    if not segments:
        return ""
    lines = []
    for segment in segments:
        start = segment.get("start")
        end = segment.get("end")
        text = segment.get("text", "")
        if start is not None and end is not None:
            lines.append(f"- [{start:.1f}s - {end:.1f}s] {text}")
        else:
            lines.append(f"- {text}")
    return "\n".join(lines)


def generate_work_instructions(
    visual_sequence: list,
    transcript: dict,
    interval_seconds: float,
    video_duration_sec: float,
) -> str:
    """Combine visual steps and spoken transcript into formal work instructions."""
    client = get_client()
    model = get_writer_model()

    visual_steps = _format_visual_steps(visual_sequence, interval_seconds)
    spoken_steps = transcript.get("text", "").strip()
    timed_transcript = _format_transcript_segments(transcript.get("segments", []))

    context_parts = []
    if visual_steps:
        context_parts.append("## Observed actions from video frames\n" + visual_steps)
    if spoken_steps:
        context_parts.append("## Spoken narration from audio transcript\n" + spoken_steps)
    if timed_transcript:
        context_parts.append("## Timestamped transcript segments\n" + timed_transcript)

    if not context_parts:
        return "No visual or audio content was available to generate work instructions."

    duration_label = seconds_to_timestamp(video_duration_sec)
    prompt = (
        f"Create professional work instructions for an industrial/factory task using the "
        f"observations below. The source video is {video_duration_sec:.0f} seconds long "
        f"(ends at {duration_label}). Merge visual actions and spoken narration into one "
        f"coherent procedure covering the FULL video. Resolve duplicates; keep chronological order.\n\n"
        "TIMESTAMP RULES (critical):\n"
        f"- Use M:SS.ss format only, e.g. [0:12.00 - 0:18.00] or [2:45.00 - 3:04.00]\n"
        f"- Never use timestamps beyond {duration_label}\n"
        "- Do NOT write [43:00 - 50:00] to mean 43–50 seconds; write [0:43.00 - 0:50.00] instead\n"
        "- Prefix each numbered action step with its timestamp range: **[0:12.00 - 0:18.00]** description\n"
        "- Steps MUST be sequential and NON-OVERLAPPING (end time of step N must be <= start time of step N+1)\n"
        "- Keep each step between 8 and 35 seconds when possible\n"
        "- Write 2-3 explanatory sentences per step (what to do and why), not bullet-only lists\n\n"
        "INSTRUMENTS (critical):\n"
        "- In Required tools/materials/PPE, list instruments, tools, and materials observed "
        "in frame analysis or mentioned in the audio transcript\n"
        "- For each numbered step, when tools/instruments are used in that time range, add a "
        "final line: Instruments used: flashlight, screwdriver, tie-down strap\n"
        "- Only list instruments actually seen in video analysis or heard in audio for that step\n\n"
        "Include:\n"
        "1. Title and brief purpose\n"
        "2. Required tools, materials, and PPE (highlight important instruments from video/audio)\n"
        "3. Numbered step-by-step instructions (clear and actionable)\n"
        "4. Safety warnings and quality checks\n"
        "5. Post-task cleanup or verification steps\n\n"
        + "\n\n".join(context_parts)
    )

    print(f"Generating work instructions with '{model}'...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional industrial technical writer. "
                    "Produce clear, operator-ready work instructions."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def run_pipeline(
    video_path: str,
    output_dir: str,
    frame_interval: float = 1.0,
    transcript_chunk_seconds: float = 10.0,
    visual_chunk_seconds: float = 20.0,
    vision_workers: int = 10,
    analyze_every: int = 1,
    on_status: Callable[[str, int], None] | None = None,
    api_mode: bool = False,
) -> dict:
    """Run the full video analysis pipeline. Called by the API with a temp output_dir."""
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    print_model_config()
    validate_azure_config(check_whisper=True)

    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "extracted_frames")
    os.makedirs(frames_dir, exist_ok=True)

    if on_status:
        on_status("extracting_frames", 10)

    print(f"\n=== Step 1/5: Extracting frames (every {frame_interval}s) ===")
    frame_count = extract_frames(video_path, frames_dir, interval_seconds=frame_interval)
    video_duration = get_video_duration(video_path)
    print(f"Video duration: {video_duration:.1f}s ({video_duration / 60:.1f} min)")

    if on_status:
        on_status("transcribing", 30)

    print("\n=== Step 2/5: Transcribing audio ===")
    transcript = transcribe_video(video_path, output_dir)

    if on_status:
        on_status("analyzing_frames", 55)

    print("\n=== Step 3/5: Analyzing frames ===")
    cache_path = os.path.join(output_dir, "frame_analysis_cache.json")
    raw_results = analyze_frames(
        frames_dir,
        workers=vision_workers,
        analyze_every=analyze_every,
        cache_path=cache_path,
    )
    if not raw_results:
        print(
            "\nERROR: No frames were analyzed successfully.\n"
            "  Deploy gpt-4o on your Azure OpenAI resource\n"
            "  and ensure AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o in .env."
        )
    visual_sequence = consolidate_sequence(raw_results)

    sequence_path = os.path.join(output_dir, "visual_sequence.json")
    with open(sequence_path, "w", encoding="utf-8") as f:
        json.dump(visual_sequence, f, indent=2)

    if on_status and not api_mode:
        on_status("generating_wi", 75)

    work_instructions = ""
    instructions_path = os.path.join(output_dir, "work_instructions.txt")
    report_artifacts: dict = {}

    if api_mode:
        print("\n=== Step 4–5: Skipped for API (deferred to background after streaming) ===")
    else:
        print("\n=== Step 4/5: Generating work instructions ===")
        work_instructions = generate_work_instructions(
            visual_sequence,
            transcript,
            frame_interval,
            video_duration,
        )

        with open(instructions_path, "w", encoding="utf-8") as f:
            f.write(work_instructions)

        if on_status:
            on_status("building_report", 85)

        print("\n=== Step 5/5: Building interactive report ===")
        report_artifacts = build_report(
            video_path=video_path,
            output_dir=output_dir,
            transcript=transcript,
            visual_sequence=visual_sequence,
            work_instructions=work_instructions,
            frame_interval=frame_interval,
            video_duration_sec=video_duration,
            chunk_seconds=transcript_chunk_seconds,
            visual_chunk_seconds=visual_chunk_seconds,
            summarize_visual=True,
        )

        if on_status:
            on_status("building_report", 90)

    return {
        "frames_dir": frames_dir,
        "frame_count": frame_count,
        "transcript": transcript,
        "visual_sequence": visual_sequence,
        "work_instructions": work_instructions,
        "work_instructions_path": instructions_path,
        "report_path": report_artifacts.get("report_path", ""),
        "video_duration": video_duration,
    }
