import asyncio
import os
import tempfile

from fastapi import APIRouter, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from api.models import JobCreateRequest, JobCreateResponse, JobStatusResponse, JobUploadResponse
from api.services import blob_storage
from api.services.job_manager import job_manager
from api.services.pipeline_runner import run_job_pipeline

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
UPLOAD_CHUNK_BYTES = 1024 * 1024

DEFAULT_FRAME_INTERVAL = max(1.0, float(os.getenv("FRAME_INTERVAL_SECONDS", "1.0")))
DEFAULT_VISION_WORKERS = int(os.getenv("DEFAULT_VISION_WORKERS", "8"))
DEFAULT_ANALYZE_EVERY = max(1, int(os.getenv("DEFAULT_ANALYZE_EVERY", "1")))

UPLOAD_PROGRESS_MAX = 5

# Keep strong refs so pipeline tasks are not garbage-collected mid-run.
_running_tasks: set[asyncio.Task] = set()


def _validate_filename_value(filename: str) -> str:
    if not filename or not filename.strip():
        raise ValueError("filename is required")
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported video format. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )
    return ext


def _validate_filename(filename: str) -> str:
    try:
        return _validate_filename_value(filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _pipeline_from_request(body: JobCreateRequest) -> dict:
    return {
        "frame_interval": max(1.0, body.frame_interval),
        "vision_workers": body.vision_workers,
        "analyze_every": body.analyze_every,
    }


def _upload_progress(received: int, total: int | None) -> int:
    if total and total > 0:
        return min(UPLOAD_PROGRESS_MAX, max(1, int((received / total) * UPLOAD_PROGRESS_MAX)))
    return 1


def _spawn_task(coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    _running_tasks.add(task)
    task.add_done_callback(_running_tasks.discard)
    return task


async def _publish_status(
    job_id: str,
    status: str,
    progress: int,
    *,
    message: str | None = None,
) -> None:
    fields: dict = {"status": status, "progress": progress}
    if message is not None:
        fields["error"] = message
    elif status != "failed":
        fields["error"] = None
    # Frequency of uploading ticks must not block on Azure Blob writes.
    persist = status not in ("uploading",)
    await job_manager.update_metadata(job_id, persist=persist, **fields)
    event: dict = {
        "type": "status",
        "job_id": job_id,
        "status": status,
        "progress": progress,
    }
    if message:
        event["message"] = message
    await job_manager.publish(job_id, event)


async def _publish_job_error(job_id: str, message: str) -> None:
    text = message or "Job failed"
    await job_manager.update_metadata(job_id, status="failed", error=text, progress=0)
    await job_manager.publish(
        job_id,
        {"type": "error", "job_id": job_id, "message": text},
    )


async def _save_upload_to_temp(
    job_id: str,
    video: UploadFile,
    suffix: str,
    content_length: int | None = None,
) -> str:
    """Stream UploadFile to disk; publish progress at most every ~5% / 20 MB."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        local_path = tmp.name
        total = 0
        last_progress = 0
        last_emit_bytes = 0
        emit_every_bytes = 20 * 1024 * 1024  # avoid hammering blob metadata every MB
        while True:
            chunk = await video.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            tmp.write(chunk)
            total += len(chunk)
            progress = _upload_progress(total, content_length)
            should_emit = (
                progress > last_progress
                or (total - last_emit_bytes) >= emit_every_bytes
            )
            if should_emit:
                await _publish_status(job_id, "uploading", progress)
                last_progress = progress
                last_emit_bytes = total

    if total == 0:
        os.unlink(local_path)
        raise ValueError("Empty video file")
    print(f"INFO: job {job_id} saved upload ({total / (1024 * 1024):.1f} MB)")
    return local_path


async def _run_job_from_local(
    job_id: str,
    local_path: str,
    _source_blob: str,
    frame_interval: float,
    vision_workers: int,
    analyze_every: int,
) -> None:
    try:
        await _publish_status(job_id, "processing", 6)
        await run_job_pipeline(
            job_id,
            local_path,
            frame_interval=frame_interval,
            vision_workers=vision_workers,
            analyze_every=analyze_every,
        )
    except Exception as exc:
        print(f"ERROR: job {job_id} failed: {exc!r}")
        await _publish_job_error(job_id, str(exc) or repr(exc))
    finally:
        if os.path.isfile(local_path):
            os.unlink(local_path)


@router.post("", response_model=JobCreateResponse)
async def create_job(body: JobCreateRequest):
    """
    Create a job and return job_id immediately (no file).
    Open WS /api/v1/jobs/{job_id}/stream, then POST /api/v1/jobs/{job_id}/video.
    """
    ext = _validate_filename(body.filename)
    pipeline = _pipeline_from_request(body)

    job_id = job_manager.new_job_id()
    source_blob = blob_storage.job_blob_path(job_id, "source", f"video{ext}")

    try:
        await job_manager.create_job(
            job_id,
            source_blob=source_blob,
            original_filename=body.filename,
            pipeline=pipeline,
            status="awaiting_upload",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create job: {exc}") from exc

    return JobCreateResponse(job_id=job_id, status="awaiting_upload")


@router.post("/{job_id}/video", response_model=JobUploadResponse)
async def upload_job_video(
    job_id: str,
    video: UploadFile = File(...),
):
    """
    Upload video for an existing job.

    Reads the file during this request (WebSocket receives `uploading` progress).
    Returns when the file is saved (`uploaded`); then the pipeline runs in the background
    and continues streaming on the same WebSocket (`processing` → `step` → `complete`).
    """
    metadata = await job_manager.try_start_upload(job_id)
    if metadata is None:
        existing = await job_manager.get_metadata(job_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Job not found")
        raise HTTPException(
            status_code=409,
            detail=f"Job cannot accept upload in status '{existing.get('status', 'unknown')}'",
        )

    if not video.filename:
        await _publish_status(
            job_id, "awaiting_upload", 0, message="No video file provided"
        )
        raise HTTPException(status_code=400, detail="No video file provided")

    try:
        ext = _validate_filename_value(video.filename)
    except ValueError as exc:
        await _publish_status(job_id, "awaiting_upload", 0, message=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    expected_ext = os.path.splitext(metadata.get("original_filename", ""))[1].lower()
    if expected_ext and ext != expected_ext:
        await _publish_status(
            job_id,
            "awaiting_upload",
            0,
            message=f"Expected {expected_ext} file, got {ext}",
        )
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected_ext} file, got {ext}",
        )

    pipeline = metadata.get("pipeline") or {}
    frame_interval = max(1.0, float(pipeline.get("frame_interval", DEFAULT_FRAME_INTERVAL)))
    vision_workers = int(pipeline.get("vision_workers", DEFAULT_VISION_WORKERS))
    analyze_every = int(pipeline.get("analyze_every", DEFAULT_ANALYZE_EVERY))
    source_blob = metadata.get("source_blob") or blob_storage.job_blob_path(
        job_id, "source", f"video{ext}"
    )

    content_length = None
    size_header = getattr(video, "size", None)
    if isinstance(size_header, int) and size_header > 0:
        content_length = size_header

    await _publish_status(job_id, "uploading", 1)

    local_path: str | None = None
    try:
        local_path = await _save_upload_to_temp(
            job_id, video, ext, content_length=content_length
        )
    except ValueError as exc:
        await _publish_status(job_id, "awaiting_upload", 0, message=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # ClientDisconnect often has empty str() — keep WS open with a clear status.
        msg = str(exc) or repr(exc) or "Upload interrupted"
        print(f"ERROR: job {job_id} upload failed: {exc!r}")
        await _publish_status(job_id, "awaiting_upload", 0, message=msg)
        if local_path and os.path.isfile(local_path):
            os.unlink(local_path)
        raise HTTPException(status_code=400, detail=msg) from exc
    finally:
        await video.close()

    await _publish_status(job_id, "uploaded", UPLOAD_PROGRESS_MAX)

    _spawn_task(
        _run_job_from_local(
            job_id,
            local_path,
            source_blob,
            frame_interval,
            vision_workers,
            analyze_every,
        )
    )

    return JobUploadResponse(job_id=job_id, status="uploaded")


@router.post("/upload", response_model=JobCreateResponse)
async def create_job_with_video(
    video: UploadFile = File(...),
    frame_interval: float = DEFAULT_FRAME_INTERVAL,
    vision_workers: int = DEFAULT_VISION_WORKERS,
    analyze_every: int = DEFAULT_ANALYZE_EVERY,
):
    """
    Legacy single-request upload. Prefer POST /jobs then POST /jobs/{id}/video.
    """
    if not video.filename:
        raise HTTPException(status_code=400, detail="No video file provided")

    frame_interval = max(1.0, frame_interval)
    ext = _validate_filename(video.filename)

    job_id = job_manager.new_job_id()
    source_blob = blob_storage.job_blob_path(job_id, "source", f"video{ext}")
    pipeline = {
        "frame_interval": frame_interval,
        "vision_workers": vision_workers,
        "analyze_every": analyze_every,
    }

    try:
        await job_manager.create_job(
            job_id,
            source_blob=source_blob,
            original_filename=video.filename,
            pipeline=pipeline,
            status="uploading",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create job: {exc}") from exc

    await _publish_status(job_id, "uploading", 1)

    local_path: str | None = None
    try:
        local_path = await _save_upload_to_temp(job_id, video, ext)
    except Exception as exc:
        msg = str(exc) or repr(exc)
        await _publish_job_error(job_id, msg)
        raise HTTPException(status_code=400, detail=msg) from exc
    finally:
        await video.close()

    await _publish_status(job_id, "uploaded", UPLOAD_PROGRESS_MAX)

    _spawn_task(
        _run_job_from_local(
            job_id,
            local_path,
            source_blob,
            frame_interval,
            vision_workers,
            analyze_every,
        )
    )

    return JobCreateResponse(job_id=job_id, status="uploaded")


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job(job_id: str):
    metadata = await job_manager.get_metadata(job_id)
    if metadata is None:
        raise HTTPException(status_code=404, detail="Job not found")

    steps = await job_manager.list_saved_steps(job_id)
    return JobStatusResponse(
        job_id=metadata["job_id"],
        status=metadata.get("status", "unknown"),
        progress=metadata.get("progress", 0),
        step_count=metadata.get("step_count", len(steps)),
        original_filename=metadata.get("original_filename"),
        created_at=metadata.get("created_at"),
        updated_at=metadata.get("updated_at"),
        error=metadata.get("error"),
        artifact_paths=metadata.get("artifact_paths"),
        steps=steps,
    )


@router.websocket("/{job_id}/stream")
async def stream_job(websocket: WebSocket, job_id: str):
    metadata = await job_manager.get_metadata(job_id)
    if metadata is None:
        await websocket.close(code=4004, reason="Job not found")
        return

    await websocket.accept()

    status = metadata.get("status", "queued")
    if status == "completed":
        steps = await job_manager.list_saved_steps(job_id)
        for step in steps:
            await websocket.send_json(step)
        artifact_paths = metadata.get("artifact_paths") or {}
        wi_blob = artifact_paths.get("artifacts/work_instructions.txt")
        report_blob = artifact_paths.get("artifacts/report.html")
        await websocket.send_json(
            {
                "type": "complete",
                "job_id": job_id,
                "step_count": len(steps),
                "metadata_url": blob_storage.generate_sas_url(
                    job_manager.metadata_path(job_id)
                ),
                "work_instructions_url": (
                    blob_storage.generate_sas_url(wi_blob) if wi_blob else None
                ),
                "report_url": (
                    blob_storage.generate_sas_url(report_blob) if report_blob else None
                ),
            }
        )
        await websocket.close()
        return

    if status == "failed":
        await websocket.send_json(
            {
                "type": "error",
                "job_id": job_id,
                "message": metadata.get("error") or "Job failed",
            }
        )
        await websocket.close()
        return

    saved_steps = await job_manager.list_saved_steps(job_id)
    for step in saved_steps:
        await websocket.send_json(step)

    queue = await job_manager.subscribe(job_id)
    seen_step_indices = {s.get("step_index") for s in saved_steps}

    try:
        await websocket.send_json(
            {
                "type": "status",
                "job_id": job_id,
                "status": status,
                "progress": metadata.get("progress", 0),
            }
        )

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping", "job_id": job_id})
                continue

            if event.get("type") == "step":
                idx = event.get("step_index")
                if idx in seen_step_indices:
                    continue
                seen_step_indices.add(idx)

            await websocket.send_json(event)

            if event.get("type") in ("complete", "error"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        await job_manager.unsubscribe(job_id, queue)
