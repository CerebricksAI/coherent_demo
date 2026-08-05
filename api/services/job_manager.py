import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from api.services import blob_storage


class JobManager:
    """Job lifecycle with blob persistence and in-memory WebSocket fan-out."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()
        self._work_dirs: dict[str, str] = {}
        self._local_metadata: dict[str, dict] = {}

    def new_job_id(self) -> str:
        return str(uuid4())

    def metadata_path(self, job_id: str) -> str:
        return blob_storage.job_blob_path(job_id, "metadata.json")

    def step_path(self, job_id: str, step_index: int) -> str:
        return blob_storage.job_blob_path(job_id, "steps", f"step_{step_index:03d}.json")

    async def create_job(
        self,
        job_id: str,
        *,
        source_blob: str,
        original_filename: str,
        pipeline: dict | None = None,
        status: str = "queued",
    ) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        metadata = {
            "job_id": job_id,
            "status": status,
            "progress": 0,
            "original_filename": original_filename,
            "source_blob": source_blob,
            "created_at": now,
            "updated_at": now,
            "step_count": 0,
            "error": None,
            "pipeline": pipeline or {},
        }
        self._local_metadata[job_id] = metadata
        asyncio.create_task(self._persist_metadata(job_id, metadata))
        return metadata

    async def _persist_metadata(self, job_id: str, metadata: dict) -> None:
        try:
            await asyncio.to_thread(
                blob_storage.upload_json, self.metadata_path(job_id), metadata
            )
        except Exception as exc:
            print(f"WARNING: metadata blob upload failed for {job_id}: {exc}")

    async def update_metadata(
        self,
        job_id: str,
        *,
        persist: bool = True,
        **fields: Any,
    ) -> dict:
        metadata = await self.get_metadata(job_id)
        if metadata is None:
            raise KeyError(f"Job not found: {job_id}")
        metadata.update(fields)
        metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._local_metadata[job_id] = metadata
        if persist:
            await self._persist_metadata(job_id, metadata)
        else:
            # Fire-and-forget so frequent progress ticks (e.g. uploading) stay fast.
            asyncio.create_task(self._persist_metadata(job_id, dict(metadata)))
        return metadata

    async def get_metadata(self, job_id: str) -> dict | None:
        local = self._local_metadata.get(job_id)
        if local is not None:
            return dict(local)
        data = await asyncio.to_thread(
            blob_storage.download_json, self.metadata_path(job_id)
        )
        if data is not None:
            self._local_metadata[job_id] = data
        return data

    async def try_start_upload(self, job_id: str) -> dict | None:
        """
        Atomically move a job from awaiting_upload → uploading.
        Returns updated metadata, or None if the job is missing or not uploadable.
        """
        async with self._lock:
            metadata = self._local_metadata.get(job_id)
            if metadata is None:
                data = await asyncio.to_thread(
                    blob_storage.download_json, self.metadata_path(job_id)
                )
                if data is None:
                    return None
                metadata = data
                self._local_metadata[job_id] = metadata

            status = metadata.get("status", "")
            if status not in ("awaiting_upload", "queued"):
                return None

            now = datetime.now(timezone.utc).isoformat()
            metadata["status"] = "uploading"
            metadata["progress"] = 1
            metadata["updated_at"] = now
            self._local_metadata[job_id] = dict(metadata)
            snapshot = dict(metadata)

        asyncio.create_task(self._persist_metadata(job_id, snapshot))
        return snapshot

    async def subscribe(self, job_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            self._subscribers.setdefault(job_id, []).append(queue)
        return queue

    async def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        async with self._lock:
            subs = self._subscribers.get(job_id, [])
            if queue in subs:
                subs.remove(queue)
            if not subs:
                self._subscribers.pop(job_id, None)

    async def publish(self, job_id: str, event: dict, *, persist_step: bool = False) -> None:
        if persist_step and event.get("type") == "step":
            step_index = event.get("step_index", 0)
            path = self.step_path(job_id, step_index)
            await asyncio.to_thread(blob_storage.upload_json, path, event)

        async with self._lock:
            queues = list(self._subscribers.get(job_id, []))

        for queue in queues:
            await queue.put(event)

    async def list_saved_steps(self, job_id: str) -> list[dict]:
        prefix = blob_storage.job_blob_path(job_id, "steps") + "/"
        names = await asyncio.to_thread(blob_storage.list_blobs, prefix)
        steps = []
        for name in sorted(names):
            if not name.endswith(".json"):
                continue
            data = await asyncio.to_thread(blob_storage.download_json, name)
            if data:
                steps.append(data)
        return steps

    def set_work_dir(self, job_id: str, work_dir: str) -> None:
        self._work_dirs[job_id] = work_dir

    def get_work_dir(self, job_id: str) -> str | None:
        return self._work_dirs.get(job_id)

    def clear_work_dir(self, job_id: str) -> str | None:
        return self._work_dirs.pop(job_id, None)

    def clip_local_path(self, job_id: str, clip_name: str) -> str | None:
        work_dir = self.get_work_dir(job_id)
        if not work_dir:
            return None
        path = os.path.join(work_dir, "clips", clip_name)
        return path if os.path.isfile(path) else None


job_manager = JobManager()
