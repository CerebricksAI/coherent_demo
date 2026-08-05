export const PIPELINE_STAGES = [
  { key: "awaiting_upload", label: "Ready for upload", progress: 0 },
  { key: "uploading", label: "Uploading video", progress: 1 },
  { key: "uploaded", label: "Video uploaded", progress: 5 },
  { key: "processing", label: "Starting pipeline", progress: 6 },
  { key: "extracting_frames", label: "Extracting frames", progress: 10 },
  { key: "transcribing", label: "Transcribing audio", progress: 30 },
  { key: "analyzing_frames", label: "Analyzing video", progress: 55 },
  { key: "generating_wi", label: "Generating work instructions", progress: 75 },
  { key: "processing_steps", label: "Building steps & clips", progress: 88 },
  { key: "completed", label: "Complete", progress: 100 },
];

export function stageIndex(status) {
  if (!status || status === "idle") return -1;
  const idx = PIPELINE_STAGES.findIndex((s) => s.key === status);
  if (idx >= 0) return idx;
  if (status === "queued") return PIPELINE_STAGES.findIndex((s) => s.key === "processing");
  return PIPELINE_STAGES.findIndex((s) => s.key === "processing");
}

export function formatTimestamp(label) {
  if (!label) return "—";
  return label;
}

export function formatDuration(start, end) {
  if (!start || !end) return "";
  return `${start} → ${end}`;
}

export function saveJobToHistory(entry) {
  try {
    const raw = localStorage.getItem("wi_job_history");
    const list = raw ? JSON.parse(raw) : [];
    const filtered = list.filter((j) => j.jobId !== entry.jobId);
    filtered.unshift(entry);
    localStorage.setItem("wi_job_history", JSON.stringify(filtered.slice(0, 20)));
  } catch {
    /* ignore */
  }
}

export function loadJobHistory() {
  try {
    const raw = localStorage.getItem("wi_job_history");
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}
