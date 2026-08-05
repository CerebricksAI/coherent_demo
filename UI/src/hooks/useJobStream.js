import { useCallback, useEffect, useRef, useState } from "react";
import { createJobRecord, openJobStream, uploadJobVideo } from "../api/client";
import { saveJobToHistory } from "../utils/helpers";

const INITIAL_STATE = {
  phase: "idle",
  jobId: null,
  filename: null,
  status: null,
  progress: 0,
  steps: [],
  complete: null,
  error: null,
};

export function useJobStream() {
  const [state, setState] = useState(INITIAL_STATE);
  const closeRef = useRef(null);
  const stepsRef = useRef([]);

  const reset = useCallback(() => {
    closeRef.current?.();
    closeRef.current = null;
    stepsRef.current = [];
    setState(INITIAL_STATE);
  }, []);

  const startJob = useCallback(async (file, options = {}) => {
    reset();
    setState((s) => ({
      ...s,
      phase: "creating",
      filename: file.name,
      error: null,
      progress: 0,
      status: "creating",
    }));

    try {
      const { job_id: jobId } = await createJobRecord(file, options);
      stepsRef.current = [];

      setState((s) => ({
        ...s,
        phase: "processing",
        jobId,
        progress: 0,
        status: "awaiting_upload",
      }));

      closeRef.current = openJobStream(jobId, {
        onStatus: (data) => {
          setState((s) => ({
            ...s,
            status: data.status,
            progress: data.progress ?? s.progress,
          }));
        },
        onStep: (data) => {
          const idx = data.step_index;
          const existing = stepsRef.current.findIndex((st) => st.step_index === idx);
          const step = { ...data };
          if (existing >= 0) {
            stepsRef.current[existing] = step;
          } else {
            stepsRef.current = [...stepsRef.current, step].sort(
              (a, b) => a.step_index - b.step_index
            );
          }
          setState((s) => ({
            ...s,
            steps: [...stepsRef.current],
          }));
        },
        onComplete: (data) => {
          setState((s) => ({
            ...s,
            phase: "complete",
            status: "completed",
            progress: 100,
            complete: data,
          }));
          saveJobToHistory({
            jobId,
            filename: file.name,
            stepCount: data.step_count ?? stepsRef.current.length,
            completedAt: new Date().toISOString(),
          });
        },
        onError: (message) => {
          setState((s) => ({
            ...s,
            phase: "error",
            error: message,
          }));
        },
      });

      setState((s) => ({
        ...s,
        phase: "uploading",
        status: "uploading",
        progress: 1,
      }));

      uploadJobVideo(jobId, file)
        .then(() => {
          setState((s) => ({
            ...s,
            phase: "processing",
            status: s.status === "uploading" ? "uploaded" : s.status,
          }));
        })
        .catch((err) => {
          setState((s) => ({
            ...s,
            phase: "error",
            error: err.message || "Failed to upload video",
          }));
        });
    } catch (err) {
      setState((s) => ({
        ...s,
        phase: "error",
        error: err.message || "Failed to start job",
      }));
    }
  }, [reset]);

  useEffect(() => {
    return () => closeRef.current?.();
  }, []);

  return { ...state, startJob, reset };
}
