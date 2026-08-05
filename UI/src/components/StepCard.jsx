import { Clock, Wrench } from "lucide-react";
import { formatDuration } from "../utils/helpers";

export default function StepCard({ step, selected, onSelect }) {
  const idx = step.step_index;
  const instruments = step.instruments || [];

  return (
    <button
      type="button"
      className={`step-card ${selected ? "selected" : ""}`}
      onClick={() => onSelect(step)}
    >
      <div className="step-card-header">
        <span className="step-number">Step {idx}</span>
        <span className="step-time">
          <Clock size={12} />
          {formatDuration(step.timestamp_start, step.timestamp_end)}
        </span>
      </div>
      <p className="step-wi-preview">{step.WI}</p>
      {instruments.length > 0 && (
        <div className="instrument-tags">
          <Wrench size={12} />
          {instruments.map((tool) => (
            <span key={tool} className="tag">
              {tool}
            </span>
          ))}
        </div>
      )}
    </button>
  );
}
