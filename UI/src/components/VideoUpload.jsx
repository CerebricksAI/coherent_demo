import { useCallback, useRef, useState } from "react";
import { FileVideo, Upload, X } from "lucide-react";

const ACCEPT = ".mp4,.mov,.avi,.mkv,.webm,.m4v";

export default function VideoUpload({ onUpload, disabled, large = false }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);
  const [selected, setSelected] = useState(null);

  const pickFile = useCallback((file) => {
    if (!file) return;
    setSelected(file);
  }, []);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragOver(false);
      if (disabled) return;
      const file = e.dataTransfer.files?.[0];
      pickFile(file);
    },
    [disabled, pickFile]
  );

  const handleSubmit = () => {
    if (selected && !disabled) onUpload(selected);
  };

  const clear = () => setSelected(null);

  return (
    <div className={`upload-panel ${large ? "upload-panel--large" : ""}`}>
      <div
        className={`dropzone ${large ? "dropzone--large" : ""} ${dragOver ? "drag-over" : ""} ${disabled ? "disabled" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => !disabled && inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          hidden
          onChange={(e) => pickFile(e.target.files?.[0])}
        />
        <div className="dropzone-icon">
          <Upload size={large ? 48 : 32} />
        </div>
        <h3>{large ? "Drop your video here" : "Drop video file here"}</h3>
        <p>or click to browse · MP4, MOV, AVI, MKV, WebM</p>
      </div>

      {selected && (
        <div className="selected-file">
          <FileVideo size={20} />
          <div className="file-meta">
            <strong>{selected.name}</strong>
            <span>{(selected.size / (1024 * 1024)).toFixed(1)} MB</span>
          </div>
          <button type="button" className="icon-btn" onClick={clear} aria-label="Remove">
            <X size={16} />
          </button>
        </div>
      )}

      <button
        type="button"
        className={`btn btn-primary ${large ? "btn-lg upload-submit--large" : "btn-lg"}`}
        disabled={!selected || disabled}
        onClick={handleSubmit}
      >
        Start processing
      </button>
    </div>
  );
}
