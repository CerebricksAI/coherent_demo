import { ChevronRight } from "lucide-react";

const PHASE_LABELS = {
  idle: "Idle",
  creating: "Creating job",
  uploading: "Uploading",
  processing: "Processing",
  complete: "Complete",
  error: "Error",
};

export default function Header({ title, subtitle, jobId, phase, breadcrumb }) {
  return (
    <header className="app-header">
      <div className="header-left">
        {breadcrumb && (
          <div className="header-breadcrumb">
            <span>{breadcrumb}</span>
            <ChevronRight size={12} />
            <span>{title}</span>
          </div>
        )}
        <h1>{title}</h1>
        {subtitle && <p className="header-subtitle">{subtitle}</p>}
      </div>
      <div className="header-actions">
        {jobId && phase && (
          <span className={`phase-badge phase-${phase}`}>
            {PHASE_LABELS[phase] || phase}
          </span>
        )}
        {jobId && (
          <code className="job-id-chip" title={jobId}>
            {jobId.slice(0, 8)}
          </code>
        )}
      </div>
    </header>
  );
}
