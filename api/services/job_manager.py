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
    ) -> dict:
        now = datetime.now(timezone.utc).isoformat()
        metadata = {
            "job_id": job_id,
            "status": "queued",
            "progress": 0,
            "original_filename": original_filename,
            "source_blob": source_blob,
            "created_at": now,
            "updated_at": now,
            "step_count": 0,
            "error": None,
        }
        await asyncio.to_thread(blob_storage.upload_json, self.metadata_path(job_id), metadata)
        return metadata

    async def update_metadata(self, job_id: str, **fields: Any) -> dict:
        metadata = await self.get_metadata(job_id)
        if metadata is None:
            raise KeyError(f"Job not found: {job_id}")
        metadata.update(fields)
        metadata["updated_at"] = datetime.now(timezone.utc).isoformat()
        await asyncio.to_thread(
            blob_storage.upload_json, self.metadata_path(job_id), metadata
        )
        return metadata

    async def get_metadata(self, job_id: str) -> dict | None:
        return await asyncio.to_thread(
            blob_storage.download_json, self.metadata_path(job_id)
        )

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
