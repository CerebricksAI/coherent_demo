import { Download, ExternalLink, FileText, Play, Wrench } from "lucide-react";
import { formatDuration } from "../utils/helpers";
import { SkeletonLine } from "./Skeleton";

function VideoSkeleton() {
  return <div className="skeleton-video" aria-hidden="true" />;
}

function WiSkeleton() {
  return (
    <div className="skeleton-wi" aria-hidden="true">
      <SkeletonLine width="100%" />
      <SkeletonLine width="92%" />
      <SkeletonLine width="88%" />
      <SkeletonLine width="65%" />
    </div>
  );
}

export default function StepDetail({ step, loadingClip = false, loadingWi = false }) {  if (!step) {
    return (
      <div className="step-detail empty">
        <Play size={40} strokeWidth={1.25} />
        <h3>Select a step</h3>
        <p>Choose a work instruction step to view the clip and details.</p>
      </div>
    );
  }

  const instruments = step.instruments || [];

  return (
    <div className="step-detail">
      <div className="detail-top">
        <span className="detail-step-label">Step {step.step_index}</span>
        <h2>{formatDuration(step.timestamp_start, step.timestamp_end)}</h2>
      </div>

      <div className="detail-body">
        {loadingClip ? (
          <VideoSkeleton />
        ) : step.video_clip_url ? (
          <div className="video-wrap">
            <video
              key={step.video_clip_url}
              src={step.video_clip_url}
              controls
              playsInline
              className="step-video"
            />
          </div>
        ) : (
          <div className="video-placeholder">
            <Play size={28} strokeWidth={1.5} />
            <p>Clip unavailable for this step</p>
          </div>
        )}

        <div className="detail-section">
          <h4>
            <FileText size={14} />
            Work instructions
          </h4>
          {loadingWi || !step.WI ? (
            <WiSkeleton />
          ) : (
            <p className="wi-body">{step.WI}</p>
          )}
        </div>
        {instruments.length > 0 && (
          <div className="detail-section">
            <h4>
              <Wrench size={14} />
              Instruments &amp; tools
            </h4>
            <div className="instrument-tags large">
              {instruments.map((tool) => (
                <span key={tool} className="tag">
                  {tool}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export function CompletionPanel({ complete, filename, onNewJob }) {
  if (!complete) return null;

  return (
    <div className="completion-panel">
      <div className="completion-content">
        <h3>Job complete</h3>
        <p>
          {complete.step_count} work instruction steps generated for{" "}
          <strong>{filename}</strong>
        </p>
      </div>
      <div className="completion-actions">
        {complete.work_instructions_url && (
          <a
            href={complete.work_instructions_url}
            target="_blank"
            rel="noreferrer"
            className="btn btn-secondary btn-sm"
          >
            <Download size={14} />
            Download WI
          </a>
        )}
        {complete.report_url && (
          <a
            href={complete.report_url}
            target="_blank"
            rel="noreferrer"
            className="btn btn-secondary btn-sm"
          >
            <ExternalLink size={14} />
            Report
          </a>
        )}
        <button type="button" className="btn btn-primary btn-sm" onClick={onNewJob}>
          New job
        </button>
      </div>
    </div>
  );
}
