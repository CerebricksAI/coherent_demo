# CoherentDemoWI

API-first video-to-work-instructions pipeline for factory/industrial workflows.

Upload a video, process it in the background, and stream ordered step-by-step work instructions over WebSocket with per-step clip URLs from Azure Blob Storage.

## What This Repo Does

- Extracts frames from uploaded videos.
- Transcribes audio using Azure Whisper.
- Analyzes visual frames using Azure OpenAI (GPT-4o vision).
- Builds timeline-aligned work instruction steps.
- Cuts per-step clips and uploads artifacts to Azure Blob Storage.
- Streams status and step events over WebSocket.

## Tech Stack

- Python + FastAPI
- Azure OpenAI (vision + writer + Whisper transcription)
- Azure Blob Storage (source video, clips, metadata, artifacts)
- Uvicorn

## Project Structure

- `api/main.py` - FastAPI app and CORS setup
- `api/routes/jobs.py` - REST + WebSocket endpoints
- `api/services/pipeline_runner.py` - background orchestration, ordered streaming, clip upload
- `api/services/wi_steps.py` - segment prep, filtering, deduplication, step generation
- `api/services/blob_storage.py` - blob upload/download/SAS
- `pipeline.py` - core analysis pipeline
- `API.md` - endpoint-focused API documentation

## Prerequisites

- Python 3.10+ (3.11 recommended)
- FFmpeg available on PATH (for clip cutting)
- Azure OpenAI resources configured
- Azure Blob Storage account/container configured

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy environment file:

```bash
cp .env.example .env
```

4. Fill `.env` with your Azure values.

## Environment Variables

See `.env.example` for the full list. Main variables:

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_KEY`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_CHAT_DEPLOYMENT`
- `AZURE_OPENAI_WHISPER_ENDPOINT`
- `AZURE_OPENAI_WHISPER_KEY`
- `AZURE_OPENAI_WHISPER_DEPLOYMENT`
- `AZURE_STORAGE_ACCOUNT_NAME`
- `AZURE_STORAGE_ACCOUNT_KEY`
- `AZURE_STORAGE_CONTAINER_NAME`
- `AZURE_STORAGE_JOB_PREFIX`
- `AZURE_BLOB_SAS_EXPIRY_HOURS`
- `API_HOST`
- `API_PORT`
- `CORS_ORIGINS`

## Run the API

```bash
python -m api
```

Or:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```bash
GET http://localhost:8000/health
```

## API Flow

1. `POST /api/v1/jobs` with multipart video upload.
2. Receive `job_id`.
3. Connect `WS /api/v1/jobs/{job_id}/stream`.
4. Consume `status` and `step` events.
5. Receive `complete` with artifact URLs.

## Endpoints

### `POST /api/v1/jobs`

Creates a processing job from an uploaded video.

Query params:

- `frame_interval` (default: `1.0`)
- `vision_workers` (default: `10`)
- `analyze_every` (default: `1`)

Response:

```json
{
  "job_id": "uuid"
}
```

### `GET /api/v1/jobs/{job_id}`

Returns job metadata, progress, and saved streamed steps.

### `WS /api/v1/jobs/{job_id}/stream`

Streams events such as:

- `status` (`extracting_frames`, `transcribing`, `analyzing_frames`, etc.)
- `step` (ordered sequentially: `1,2,3,...`)
- `complete`
- `error`

Step event shape:

```json
{
  "type": "step",
  "job_id": "uuid",
  "step_index": 1,
  "WI": "Operator instruction text",
  "timestamp_start": "0:00.00",
  "timestamp_end": "0:10.00",
  "video_clip_url": "https://...sas...",
  "instruments": ["flashlight"]
}
```

## Output and Storage

Artifacts are stored in blob layout:

```text
{container}/jobs/{job_id}/
  source/video.ext
  clips/step_001.mp4
  artifacts/work_instructions.txt
  artifacts/report.html
  steps/step_001.json
  metadata.json
```

## Current Processing Behavior

- WI generation runs in parallel for speed.
- Streaming is ordered sequentially to clients.
- Non-important windows are filtered where possible.
- Similar adjacent segments are merged to reduce duplicates.
- Duplicate WI text is skipped during emission.

## Troubleshooting

- If jobs fail at startup, verify Azure credentials and deployment names in `.env`.
- If clip URLs are missing, check FFmpeg availability and blob permissions.
- If processing is slow, this is usually frame analysis latency (`analyzing_frames` stage).

## Notes

- Do not commit real secrets in `.env` or docs.
- For endpoint-heavy details and examples, see `API.md`.
