from pydantic import BaseModel, Field


class JobCreateResponse(BaseModel):
    job_id: str


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
