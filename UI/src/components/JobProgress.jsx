import { PIPELINE_STAGES, stageIndex } from "../utils/helpers";

export default function JobProgress({ status, progress, stepCount, active = true }) {
  const isIdle = status === "idle";
  const currentIdx = isIdle ? -1 : stageIndex(status);
  const pct = Math.min(100, Math.max(0, progress ?? 0));
  const label = isIdle
    ? "Upload a video from New Job to begin"
    : status === "uploading"
      ? "Uploading video"
      : status === "uploaded"
        ? "Video uploaded"
        : PIPELINE_STAGES[currentIdx]?.label || status || "Starting…";

  return (
    <div className={`progress-panel ${active && !isIdle ? "progress-panel-active" : ""}`}>
      <div className="progress-header">
        <div>
          <h3>Pipeline progress</h3>
          <p>{label}</p>
        </div>
        <div className="progress-stats">
          <span className="progress-pct">{pct}%</span>
          {stepCount > 0 && (
            <span className="step-count-badge">{stepCount} steps</span>
          )}
        </div>
      </div>

      <div className="progress-bar-track">
        <div
          className="progress-bar-fill"
          style={{ width: `${pct}%` }}
        />
      </div>

      <ol className="stage-list">
        {PIPELINE_STAGES.map((stage, i) => {
          const done = !isIdle && (i < currentIdx || (status === "completed" && i <= currentIdx));
          const isActive = !isIdle && i === currentIdx && status !== "completed";
          return (
            <li
              key={stage.key}
              className={`stage-item ${done ? "done" : ""} ${isActive ? "active" : ""}`}
            >
              <span className="stage-dot" />
              <span className="stage-label">{stage.label}</span>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
