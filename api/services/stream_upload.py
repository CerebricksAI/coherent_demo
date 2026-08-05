"""Stream HTTP request bodies to disk without blocking the route response."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Awaitable, Callable

from fastapi import Request
from multipart import FormParser
from multipart.multipart import parse_options_header

UPLOAD_CHUNK_BYTES = 1024 * 1024
VIDEO_FIELD_NAME = "video"

ProgressCallback = Callable[[int, int | None], Awaitable[None]]


async def _stream_chunks(
    request: Request,
    write_fn: Callable[[bytes], None],
    *,
    on_progress: ProgressCallback | None = None,
) -> int:
    total = 0
    content_length = request.headers.get("content-length")
    expected = int(content_length) if content_length and content_length.isdigit() else None
    async for chunk in request.stream():
        if not chunk:
            continue
        write_fn(chunk)
        total += len(chunk)
        if on_progress and (total % (UPLOAD_CHUNK_BYTES * 2) < len(chunk) or expected is None):
            await on_progress(total, expected)
    if on_progress:
        await on_progress(total, expected)
    return total


async def stream_raw_body_to_file(
    request: Request,
    suffix: str,
    *,
    on_progress: ProgressCallback | None = None,
) -> tuple[str, int]:
    """Save a raw (non-multipart) request body to a temp file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        path = tmp.name

        def write_fn(chunk: bytes) -> None:
            tmp.write(chunk)

        total = await _stream_chunks(request, write_fn, on_progress=on_progress)

    if total == 0:
        os.unlink(path)
        raise ValueError("Empty video file")
    return path, total


def _copy_file_object_to_path(file_obj, dest_path: str) -> int:
    total = 0
    with open(dest_path, "wb") as out:
        while True:
            chunk = file_obj.read(UPLOAD_CHUNK_BYTES)
            if not chunk:
                break
            out.write(chunk)
            total += len(chunk)
    return total


async def stream_multipart_video_to_file(
    request: Request,
    *,
    field_name: str = VIDEO_FIELD_NAME,
    on_progress: ProgressCallback | None = None,
) -> tuple[str, str, int]:
    """
    Incrementally parse multipart/form-data and write the video field to disk.
    Returns (local_path, original_filename, bytes_written).
    """
    raw_content_type = request.headers.get("content-type", "")
    media_type, options = parse_options_header(raw_content_type)
    media_type_str = media_type.decode("latin-1") if isinstance(media_type, bytes) else str(media_type)
    if media_type_str.lower() != "multipart/form-data":
        raise ValueError("Expected multipart/form-data")

    boundary = options.get(b"boundary")
    if not boundary:
        raise ValueError("Missing multipart boundary in Content-Type")

    content_length = request.headers.get("content-length")
    expected = int(content_length) if content_length and content_length.isdigit() else None

    local_path: str | None = None
    original_filename = ""
    total = 0

    def on_file(file) -> None:
        nonlocal local_path, original_filename, total
        name = (getattr(file, "field_name", None) or b"").decode("utf-8", errors="replace")
        if name != field_name:
            return
        original_filename = (getattr(file, "file_name", None) or b"").decode("utf-8", errors="replace")
        if not original_filename:
            raise ValueError(f"Multipart field '{field_name}' has no filename")

        suffix = os.path.splitext(original_filename)[1].lower() or ".bin"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            local_path = tmp.name
            file.file_object.seek(0)
            total = _copy_file_object_to_path(file.file_object, local_path)

    # FormParser wants the media type alone + boundary separately (not the full header).
    parser = FormParser(
        "multipart/form-data",
        on_field=None,
        on_file=on_file,
        boundary=boundary,
        config={"MAX_MEMORY_FILE_SIZE": 0},
    )

    async for chunk in request.stream():
        if not chunk:
            continue
        parser.write(chunk)
        if on_progress and parser.bytes_received % (UPLOAD_CHUNK_BYTES * 2) < len(chunk):
            await on_progress(parser.bytes_received, expected)

    parser.finalize()

    if not local_path or total == 0:
        if local_path and os.path.isfile(local_path):
            os.unlink(local_path)
        raise ValueError(f"No '{field_name}' file in multipart upload")

    if on_progress:
        await on_progress(total, expected)

    return local_path, original_filename, total


def is_multipart(request: Request) -> bool:
    content_type = request.headers.get("content-type", "")
    return "multipart/form-data" in content_type.lower()


def expected_ext_from_metadata(metadata: dict) -> str:
    return os.path.splitext(metadata.get("original_filename", ""))[1].lower()
