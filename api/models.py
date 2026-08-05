from pydantic import BaseModel, Field


class JobCreateRequest(BaseModel):
    filename: str
    frame_interval: float = 1.0
    vision_workers: int = 8
    analyze_every: int = 1


class JobCreateResponse(BaseModel):
    job_id: str
    status: str = "awaiting_upload"


class JobUploadResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int = 0
    step_count: int = 0
    original_filename: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    error: str | None = None
    artifact_paths: dict[str, str] | None = None
    steps: list[dict] = Field(default_factory=list)
