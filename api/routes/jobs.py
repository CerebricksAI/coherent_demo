import asyncio
import os
import tempfile

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, WebSocket, WebSocketDisconnect

from api.models import JobCreateResponse, JobStatusResponse
from api.services import blob_storage
from api.services.job_manager import job_manager
from api.services.pipeline_runner import run_job_pipeline

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


@router.post("", response_model=JobCreateResponse)
async def create_job(
    background_tasks: BackgroundTasks,
    video: UploadFile,
    frame_interval: float = 1.0,
    vision_workers: int = 10,
    analyze_every: int = 1,
):
    if not video.filename:
        raise HTTPException(status_code=400, detail="No video file provided")

    ext = os.path.splitext(video.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    content = await video.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty video file")

    job_id = job_manager.new_job_id()
    source_blob = blob_storage.job_blob_path(job_id, "source", f"video{ext}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(content)
        local_path = tmp.name

    await job_manager.create_job(
        job_id,
        source_blob=source_blob,
        original_filename=video.filename,
    )

    background_tasks.add_task(
        _run_job_from_local,
        job_id,
        local_path,
        source_blob,
        frame_interval,
        vision_workers,
        analyze_every,
    )

    return JobCreateResponse(job_id=job_id)


async def _run_job_from_local(
    job_id: str,
    local_path: str,
    source_blob: str,
    frame_interval: float,
    vision_workers: int,
    analyze_every: int,
) -> None:
    try:
        await asyncio.to_thread(blob_storage.upload_file, source_blob, local_path)
        await job_manager.update_metadata(job_id, status="processing", progress=5)

        await run_job_pipeline(
            job_id,
            local_path,
            frame_interval=frame_interval,
            vision_workers=vision_workers,
            analyze_every=analyze_every,
        )
    except Exception:
        pass
    finally:
        if os.path.isfile(local_path):
            os.unlink(local_path)


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
                "message": metadata.get("error", "Job failed"),
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
