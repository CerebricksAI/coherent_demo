import json
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from azure.storage.blob import BlobSasPermissions, BlobServiceClient, ContentSettings, generate_blob_sas

from dotenv import load_dotenv

load_dotenv()


def _account_name() -> str:
    name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "").strip()
    if not name:
        raise RuntimeError("AZURE_STORAGE_ACCOUNT_NAME is not set in .env")
    return name


def _account_key() -> str:
    key = os.getenv("AZURE_STORAGE_ACCOUNT_KEY", "").strip()
    if not key:
        raise RuntimeError("AZURE_STORAGE_ACCOUNT_KEY is not set in .env")
    return key


def container_name() -> str:
    name = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "").strip()
    if not name:
        raise RuntimeError("AZURE_STORAGE_CONTAINER_NAME is not set in .env")
    return name


def job_prefix() -> str:
    return os.getenv("AZURE_STORAGE_JOB_PREFIX", "jobs").strip().rstrip("/")


def sas_expiry_hours() -> int:
    return int(os.getenv("AZURE_BLOB_SAS_EXPIRY_HOURS", "24"))


def _service_client() -> BlobServiceClient:
    account = _account_name()
    return BlobServiceClient(
        account_url=f"https://{account}.blob.core.windows.net",
        credential=_account_key(),
    )


def job_blob_path(job_id: str, *parts: str) -> str:
    base = f"{job_prefix()}/{job_id}"
    if parts:
        return "/".join([base, *parts])
    return base


def ensure_container() -> None:
    client = _service_client()
    container = client.get_container_client(container_name())
    if not container.exists():
        container.create_container()


def upload_bytes(blob_path: str, data: bytes, content_type: str | None = None) -> str:
    ensure_container()
    client = _service_client()
    blob = client.get_blob_client(container=container_name(), blob=blob_path)
    content_settings = ContentSettings(content_type=content_type) if content_type else None
    blob.upload_blob(data, overwrite=True, content_settings=content_settings)
    return blob_path


def upload_file(blob_path: str, local_path: str, content_type: str | None = None) -> str:
    with open(local_path, "rb") as f:
        data = f.read()
    if not content_type:
        ext = os.path.splitext(local_path)[1].lower()
        content_type = {
            ".mp4": "video/mp4",
            ".json": "application/json",
            ".txt": "text/plain",
            ".html": "text/html",
            ".vtt": "text/vtt",
            ".jpg": "image/jpeg",
            ".wav": "audio/wav",
        }.get(ext, "application/octet-stream")
    return upload_bytes(blob_path, data, content_type=content_type)


def upload_json(blob_path: str, payload: Any) -> str:
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    return upload_bytes(blob_path, data, content_type="application/json")


def download_json(blob_path: str) -> Any | None:
    client = _service_client()
    blob = client.get_blob_client(container=container_name(), blob=blob_path)
    if not blob.exists():
        return None
    raw = blob.download_blob().readall()
    return json.loads(raw.decode("utf-8"))


def blob_exists(blob_path: str) -> bool:
    client = _service_client()
    blob = client.get_blob_client(container=container_name(), blob=blob_path)
    return blob.exists()


def list_blobs(prefix: str) -> list[str]:
    client = _service_client()
    container = client.get_container_client(container_name())
    return [b.name for b in container.list_blobs(name_starts_with=prefix)]


def download_file(blob_path: str, local_path: str) -> None:
    client = _service_client()
    blob = client.get_blob_client(container=container_name(), blob=blob_path)
    if not blob.exists():
        raise FileNotFoundError(f"Blob not found: {blob_path}")
    with open(local_path, "wb") as f:
        f.write(blob.download_blob().readall())


def generate_sas_url(blob_path: str) -> str:
    account = _account_name()
    key = _account_key()
    expiry = datetime.now(timezone.utc) + timedelta(hours=sas_expiry_hours())
    token = generate_blob_sas(
        account_name=account,
        container_name=container_name(),
        blob_name=blob_path,
        account_key=key,
        permission=BlobSasPermissions(read=True),
        expiry=expiry,
    )
    return f"https://{account}.blob.core.windows.net/{container_name()}/{blob_path}?{token}"
