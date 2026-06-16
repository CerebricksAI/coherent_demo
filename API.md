# Work Instructions API

Video analysis runs **only through the API**. Upload a video via POST, then connect to the WebSocket for streamed results. All artifacts are stored in Azure Blob Storage.

## Setup

1. Copy `.env.example` to `.env` and fill in Azure OpenAI + Blob Storage values.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the server:

```bash
python -m api
```

Or:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Endpoints

### `POST /api/v1/jobs`

Upload a video file (`multipart/form-data` field: `video`).

Optional query params: `frame_interval`, `vision_workers`, `analyze_every`.

**Response:**

```json
{
  "job_id": "uuid"
}
```

Processing runs in the background. Connect to `ws://localhost:8000/api/v1/jobs/{job_id}/stream` for streamed results.

### `GET /api/v1/jobs/{job_id}`

Poll job status and retrieve saved step payloads.

### `WS /api/v1/jobs/{job_id}/stream`

Connect after POST to receive streamed events:

**Status:**
```json
{ "type": "status", "job_id": "...", "status": "transcribing", "progress": 30 }
```

**Step:**
```json
{
  "type": "step",
  "job_id": "...",
  "step_index": 1,
  "WI": "Begin by inspecting all optics...",
  "timestamp_start": "0:00.00",
  "timestamp_end": "0:08.20",
  "video_clip_url": "https://.../clips/step_001.mp4?...",
  "instruments": ["flashlight"]
}
```

**Complete:**
```json
{
  "type": "complete",
  "job_id": "...",
  "metadata_url": "...",
  "work_instructions_url": "...",
  "report_url": "...",
  "step_count": 12
}
```

## Blob layout

```
{container}/jobs/{job_id}/
  source/video.mp4
  clips/step_001.mp4
  artifacts/work_instructions.txt
  artifacts/report.html
  steps/step_001.json
  metadata.json
```

## Frontend flow

1. `POST /api/v1/jobs` with video → receive `job_id`
2. Open WebSocket to `ws_url`
3. Render each `step` event as it arrives
4. Optionally poll `GET /api/v1/jobs/{job_id}` if WebSocket disconnects
