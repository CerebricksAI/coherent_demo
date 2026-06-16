import asyncio
import os
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Awaitable, Callable

from api.services import blob_storage
from api.services.job_manager import job_manager
from api.services.step_enricher import build_step_event
from api.services.video_clips import cut_video_clip
from api.services.wi_steps import (
    prepare_segment_payload,
    stream_aligned_work_steps,
    wi_text_similar,
)
from key_frame_generator import get_video_duration
from pipeline import generate_work_instructions, run_pipeline
from report_builder import build_report

StatusCallback = Callable[[str, int], None]

STAGE_PROGRESS = {
    "extracting_frames": 10,
    "transcribing": 30,
    "analyzing_frames": 55,
    "generating_wi": 75,
    "building_report": 85,
    "processing_steps": 88,
    "uploading_artifacts": 95,
    "completed": 100,
}

CLIP_WORKERS = 4
WI_WORKERS = 5


class OrderedStepEmitter:
    """
    Buffer parallel WI results and emit WebSocket steps in strict order 1, 2, 3…
    Skips duplicate WI text so the client never sees repeated instructions.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        emit_fn: Callable[[int, int, dict], Awaitable[None]],
    ):
        self._loop = loop
        self._emit_fn = emit_fn
        self._buffer: dict[int, dict] = {}
        self._next_segment = 1
        self._stream_index = 0
        self._last_wi = ""
        self._lock: asyncio.Lock | None = None
        self.emitted_count = 0

    def _ensure_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def offer(self, segment_index: int, step: dict) -> None:
        asyncio.run_coroutine_threadsafe(
            self._offer_async(segment_index, step), self._loop
        )

    async def _offer_async(self, segment_index: int, step: dict) -> None:
        async with self._ensure_lock():
            self._buffer[segment_index] = step
            await self._drain()

    async def _drain(self) -> None:
        while self._next_segment in self._buffer:
            step = self._buffer.pop(self._next_segment)
            wi = step.get("text", "")
            if self._stream_index > 0 and wi_text_similar(wi, self._last_wi):
                print(
                    f"  Skipping duplicate WI for segment {self._next_segment} "
                    f"(similar to previous step)."
                )
                self._next_segment += 1
                continue

            self._stream_index += 1
            await self._emit_fn(self._stream_index, self._next_segment, step)
            self._last_wi = wi
            self.emitted_count += 1
            self._next_segment += 1


async def _emit_status(job_id: str, status: str, progress: int | None = None) -> None:
    if progress is None:
        progress = STAGE_PROGRESS.get(status, 0)
    await job_manager.update_metadata(job_id, status=status, progress=progress)
    await job_manager.publish(
        job_id,
        {"type": "status", "job_id": job_id, "status": status, "progress": progress},
    )


def _make_status_callback(job_id: str, loop: asyncio.AbstractEventLoop) -> StatusCallback:
    def on_status(stage: str, progress: int) -> None:
        asyncio.run_coroutine_threadsafe(_emit_status(job_id, stage, progress), loop)

    return on_status


async def _clip_blob_sas_url(job_id: str, local_path: str, step_index: int) -> str | None:
    """Upload clip to Azure Blob and return a read SAS URL."""
    if not os.path.isfile(local_path):
        return None
    blob_path = blob_storage.job_blob_path(job_id, "clips", f"step_{step_index:03d}.mp4")
    await asyncio.to_thread(blob_storage.upload_file, blob_path, local_path)
    return await asyncio.to_thread(blob_storage.generate_sas_url, blob_path)


async def _upload_artifacts(job_id: str, work_dir: str, video_path: str) -> dict[str, str]:
    paths: dict[str, str] = {}
    artifact_map = {
        "work_instructions.txt": "artifacts/work_instructions.txt",
        "visual_sequence.json": "artifacts/visual_sequence.json",
        "transcript_segments.json": "artifacts/transcript_segments.json",
        "transcript.txt": "artifacts/transcript.txt",
        "timeline.json": "artifacts/timeline.json",
        "report.html": "artifacts/report.html",
        "transcript.vtt": "artifacts/transcript.vtt",
    }
    for local_name, blob_suffix in artifact_map.items():
        local_path = os.path.join(work_dir, local_name)
        if os.path.isfile(local_path):
            blob_path = blob_storage.job_blob_path(job_id, blob_suffix)
            await asyncio.to_thread(blob_storage.upload_file, blob_path, local_path)
            paths[blob_suffix] = blob_path
    return paths


async def _finalize_job_background(
    job_id: str,
    work_dir: str,
    video_path: str,
    result: dict,
    frame_interval: float,
    step_count: int,
) -> None:
    """Generate report, upload artifacts and clips to blob after streaming completes."""
    try:
        await _emit_status(job_id, "uploading_artifacts", 95)
        video_duration = result.get("video_duration") or get_video_duration(video_path)

        wi_path = os.path.join(work_dir, "work_instructions.txt")
        if not os.path.isfile(wi_path):
            work_instructions = await asyncio.to_thread(
                generate_work_instructions,
                result["visual_sequence"],
                result["transcript"],
                frame_interval,
                video_duration,
            )
            with open(wi_path, "w", encoding="utf-8") as f:
                f.write(work_instructions)
            result["work_instructions"] = work_instructions

        if not os.path.isfile(os.path.join(work_dir, "report.html")):
            await asyncio.to_thread(
                build_report,
                video_path=video_path,
                output_dir=work_dir,
                transcript=result["transcript"],
                visual_sequence=result["visual_sequence"],
                work_instructions=result.get("work_instructions", ""),
                frame_interval=frame_interval,
                video_duration_sec=video_duration,
                chunk_seconds=10.0,
                visual_chunk_seconds=20.0,
                summarize_visual=True,
            )

        clips_dir = os.path.join(work_dir, "clips")
        if os.path.isdir(clips_dir):
            for name in sorted(os.listdir(clips_dir)):
                if not name.endswith(".mp4"):
                    continue
                step_index = int(name.replace("step_", "").replace(".mp4", ""))
                blob_path = blob_storage.job_blob_path(
                    job_id, "clips", f"step_{step_index:03d}.mp4"
                )
                if not blob_storage.blob_exists(blob_path):
                    local_path = os.path.join(clips_dir, name)
                    await asyncio.to_thread(blob_storage.upload_file, blob_path, local_path)

        artifact_paths = await _upload_artifacts(job_id, work_dir, video_path)
        metadata_url = await asyncio.to_thread(
            blob_storage.generate_sas_url, job_manager.metadata_path(job_id)
        )
        wi_blob = artifact_paths.get("artifacts/work_instructions.txt")
        report_blob = artifact_paths.get("artifacts/report.html")

        await job_manager.update_metadata(
            job_id,
            status="completed",
            progress=100,
            step_count=step_count,
            artifact_paths=artifact_paths,
        )

        await job_manager.publish(
            job_id,
            {
                "type": "complete",
                "job_id": job_id,
                "metadata_url": metadata_url,
                "work_instructions_url": (
                    await asyncio.to_thread(blob_storage.generate_sas_url, wi_blob)
                    if wi_blob
                    else None
                ),
                "report_url": (
                    await asyncio.to_thread(blob_storage.generate_sas_url, report_blob)
                    if report_blob
                    else None
                ),
                "step_count": step_count,
            },
        )
    except Exception as exc:
        await job_manager.update_metadata(
            job_id, status="failed", progress=0, error=str(exc)
        )
        await job_manager.publish(
            job_id,
            {"type": "error", "job_id": job_id, "message": str(exc)},
        )
    finally:
        cleared = job_manager.clear_work_dir(job_id)
        if cleared and os.path.isdir(cleared):
            shutil.rmtree(cleared, ignore_errors=True)


def _cut_clips_parallel(
    video_path: str,
    clips_dir: str,
    segment_payload: list[dict],
) -> dict[int, str]:
    """Cut all clips in parallel; returns step_index -> local path."""
    os.makedirs(clips_dir, exist_ok=True)
    paths: dict[int, str] = {}
    workers = max(1, min(CLIP_WORKERS, len(segment_payload)))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for seg in segment_payload:
            idx = seg["step_index"]
            out = os.path.join(clips_dir, f"step_{idx:03d}.mp4")
            futures[
                executor.submit(
                    cut_video_clip,
                    video_path,
                    out,
                    seg["start_sec"],
                    seg["end_sec"],
                )
            ] = (idx, out)

        for future in as_completed(futures):
            idx, out = futures[future]
            try:
                if future.result():
                    paths[idx] = out
            except Exception as exc:
                print(f"WARNING: clip cut failed for step {idx}: {exc}")

    return paths


async def run_job_pipeline(
    job_id: str,
    video_path: str,
    *,
    frame_interval: float = 1.0,
    vision_workers: int = 10,
    analyze_every: int = 1,
) -> None:
    work_dir = tempfile.mkdtemp(prefix=f"wi_job_{job_id}_")
    job_manager.set_work_dir(job_id, work_dir)
    loop = asyncio.get_running_loop()

    try:
        await _emit_status(job_id, "extracting_frames")
        on_status = _make_status_callback(job_id, loop)

        result = await loop.run_in_executor(
            None,
            lambda: run_pipeline(
                video_path=video_path,
                output_dir=work_dir,
                frame_interval=frame_interval,
                vision_workers=vision_workers,
                analyze_every=analyze_every,
                on_status=on_status,
                api_mode=True,
            ),
        )

        video_duration = result.get("video_duration") or get_video_duration(video_path)
        await _emit_status(job_id, "processing_steps", 88)

        segment_payload = prepare_segment_payload(
            result["visual_sequence"],
            result["transcript"],
            frame_interval,
            video_duration,
        )
        if not segment_payload:
            raise RuntimeError("No visual segments to generate work instructions.")

        clips_dir = os.path.join(work_dir, "clips")
        print("Cutting video clips in parallel...")
        clip_paths = await loop.run_in_executor(
            None,
            _cut_clips_parallel,
            video_path,
            clips_dir,
            segment_payload,
        )

        async def _emit_step(stream_index: int, segment_index: int, step: dict) -> None:
            clip_url = None
            local_clip = clip_paths.get(segment_index)
            if local_clip:
                try:
                    clip_url = await _clip_blob_sas_url(job_id, local_clip, stream_index)
                except Exception as exc:
                    print(
                        f"WARNING: clip upload failed for stream step {stream_index} "
                        f"(segment {segment_index}): {exc}"
                    )
            event = build_step_event(job_id, stream_index, step, clip_url)
            await job_manager.publish(job_id, event, persist_step=True)

        ordered_emitter = OrderedStepEmitter(loop, _emit_step)

        emit_futures: list = []
        emit_futures_lock = threading.Lock()

        def on_step_ready(step_index: int, step: dict) -> None:
            fut = asyncio.run_coroutine_threadsafe(
                ordered_emitter._offer_async(step_index, step), loop
            )
            with emit_futures_lock:
                emit_futures.append(fut)

        steps = await loop.run_in_executor(
            None,
            lambda: stream_aligned_work_steps(
                segment_payload,
                video_duration,
                on_step_ready,
                max_workers=WI_WORKERS,
            ),
        )

        def _wait_emits() -> None:
            with emit_futures_lock:
                pending = list(emit_futures)
            for fut in pending:
                fut.result(timeout=600)

        await loop.run_in_executor(None, _wait_emits)
        streamed_count = ordered_emitter.emitted_count

        # Finalization must complete before the request temp video is cleaned up
        # by the caller in jobs.py finally block.
        await _finalize_job_background(
            job_id,
            work_dir,
            video_path,
            result,
            frame_interval,
            streamed_count,
        )

    except Exception as exc:
        await job_manager.update_metadata(
            job_id, status="failed", progress=0, error=str(exc)
        )
        await job_manager.publish(
            job_id,
            {"type": "error", "job_id": job_id, "message": str(exc)},
        )
        cleared = job_manager.clear_work_dir(job_id)
        if cleared and os.path.isdir(cleared):
            shutil.rmtree(cleared, ignore_errors=True)
        raise
