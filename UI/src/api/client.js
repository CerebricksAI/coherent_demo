const DEFAULT_API_BASE = import.meta.env.VITE_API_BASE_URL || "";

export function getApiBase() {
  return DEFAULT_API_BASE.replace(/\/$/, "");
}

export function getWsBase() {
  const base = getApiBase();
  if (base.startsWith("http")) {
    return base.replace(/^http/, "ws");
  }
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}`;
}

export async function checkHealth() {
  const res = await fetch(`${getApiBase()}/health`);
  if (!res.ok) throw new Error("API health check failed");
  const data = await res.json();
  if (data.service !== "qsaid-wi" || data.status !== "ok") {
    throw new Error(
      "Wrong service on this port (expected QSAID WI API). " +
        "Stop other apps and run: python -m api"
    );
  }
  return data;
}

export async function createJobRecord(videoFile, options = {}) {
  let createRes;
  try {
    createRes = await fetch(`${getApiBase()}/api/v1/jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: videoFile.name,
        frame_interval: options.frameInterval ?? 1.0,
        vision_workers: options.visionWorkers ?? 8,
        analyze_every: options.analyzeEvery ?? 1,
      }),
    });
  } catch (err) {
    throw new Error(
      "Could not reach the API. Ensure python -m api is running on port 8001 " +
        "and UI/.env sets VITE_API_BASE_URL=http://localhost:8001"
    );
  }
  if (!createRes.ok) {
    const err = await createRes.json().catch(() => ({}));
    throw new Error(_formatApiError(err, createRes.status));
  }
  const data = await createRes.json();
  return {
    job_id: data.job_id,
    status: data.status ?? "awaiting_upload",
  };
}

export async function uploadJobVideo(jobId, videoFile) {
  const form = new FormData();
  form.append("video", videoFile);
  let uploadRes;
  try {
    uploadRes = await fetch(`${getApiBase()}/api/v1/jobs/${jobId}/video`, {
      method: "POST",
      body: form,
    });
  } catch (err) {
    throw new Error("Could not upload video to the API.");
  }
  if (!uploadRes.ok) {
    const err = await uploadRes.json().catch(() => ({}));
    throw new Error(_formatApiError(err, uploadRes.status));
  }
  return uploadRes.json();
}

export async function createJob(videoFile, options = {}) {
  const { job_id: jobId } = await createJobRecord(videoFile, options);
  await uploadJobVideo(jobId, videoFile);
  return { job_id: jobId };
}

function _formatApiError(err, status) {
  const detail = err.detail;
  if (status === 404) {
    return (
      "Work Instructions API not found on this port — another app may be using it. " +
      "Run python -m api (port 8001) and set VITE_API_BASE_URL=http://localhost:8001"
    );
  }
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
  }
  return `Upload failed (${status})`;
}

export async function getJob(jobId) {
  const res = await fetch(`${getApiBase()}/api/v1/jobs/${jobId}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Failed to load job (${res.status})`);
  }
  return res.json();
}

export function openJobStream(jobId, handlers) {
  const wsUrl = `${getWsBase()}/api/v1/jobs/${jobId}/stream`;
  const ws = new WebSocket(wsUrl);

  ws.onopen = () => handlers.onOpen?.();
  ws.onerror = () => handlers.onError?.("WebSocket connection error");
  ws.onclose = (ev) => {
    if (!handlers._completed && ev.code !== 1000) {
      handlers.onError?.("Connection closed unexpectedly");
    }
    handlers.onClose?.();
  };
  ws.onmessage = (event) => {
    let data;
    try {
      data = JSON.parse(event.data);
    } catch {
      return;
    }

    switch (data.type) {
      case "status":
        handlers.onStatus?.(data);
        break;
      case "step":
        handlers.onStep?.(data);
        break;
      case "complete":
        handlers._completed = true;
        handlers.onComplete?.(data);
        break;
      case "error":
        handlers._completed = true;
        handlers.onError?.(data.message || "Job failed");
        break;
      case "ping":
        break;
      default:
        handlers.onMessage?.(data);
    }
  };

  return () => {
    handlers._completed = true;
    if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
      ws.close();
    }
  };
}
