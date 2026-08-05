import { useCallback, useEffect, useState } from "react";
import { AlertCircle, Plus } from "lucide-react";
import { checkHealth } from "./api/client";
import Header from "./components/Header";
import JobHistory from "./components/JobHistory";
import JobProgress from "./components/JobProgress";
import SettingsPanel from "./components/SettingsPanel";
import Sidebar from "./components/Sidebar";
import StepCard from "./components/StepCard";
import StepDetail, { CompletionPanel } from "./components/StepDetail";
import { DashboardSkeleton } from "./components/Skeleton";
import VideoUpload from "./components/VideoUpload";
import { useJobStream } from "./hooks/useJobStream";

const VIEW_META = {
  dashboard: { title: "Dashboard", breadcrumb: "Workspace" },
  upload: { title: "New Job", breadcrumb: "Workspace" },
  history: { title: "Job History", breadcrumb: "Workspace" },
  settings: { title: "Settings", breadcrumb: "System" },
};

export default function App() {
  const [activeView, setActiveView] = useState("dashboard");
  const [apiOnline, setApiOnline] = useState(false);
  const [selectedStep, setSelectedStep] = useState(null);
  const job = useJobStream();

  const pollHealth = useCallback(async () => {
    try {
      await checkHealth();
      setApiOnline(true);
    } catch {
      setApiOnline(false);
    }
  }, []);

  useEffect(() => {
    pollHealth();
    const id = setInterval(pollHealth, 15000);
    return () => clearInterval(id);
  }, [pollHealth]);

  useEffect(() => {
    if (job.steps.length > 0 && !selectedStep) {
      setSelectedStep(job.steps[0]);
    }
    if (job.steps.length > 0 && selectedStep) {
      const updated = job.steps.find((s) => s.step_index === selectedStep.step_index);
      if (updated) setSelectedStep(updated);
    }
  }, [job.steps, selectedStep]);

  const handleUpload = (file) => {
    setActiveView("dashboard");
    setSelectedStep(null);
    job.startJob(file);
  };

  const handleNewJob = () => {
    job.reset();
    setSelectedStep(null);
    setActiveView("upload");
  };

  const isBusy =
    job.phase === "creating" ||
    job.phase === "uploading" ||
    job.phase === "processing";
  const isIdle = job.phase === "idle";
  const hasActiveJob = !isIdle;
  const showSkeletonSteps = job.steps.length === 0 && job.phase !== "error";

  const progressStatus =
    job.phase === "idle"
      ? "idle"
      : job.phase === "creating"
        ? "awaiting_upload"
        : job.status || job.phase || "processing";

  const progressValue = isIdle ? 0 : job.progress;
  const viewMeta = VIEW_META[activeView] || VIEW_META.dashboard;

  const headerSubtitle =
    job.filename && hasActiveJob && activeView === "dashboard"
      ? job.filename
      : activeView === "dashboard" && isIdle
        ? "Work instruction steps and video clips will appear here"
        : null;

  const toolbarTitle = isIdle
    ? "No active job"
    : job.filename || "Processing job";

  return (
    <div className="app-shell">
      <Sidebar activeView={activeView} onNavigate={setActiveView} apiOnline={apiOnline} />

      <div className="app-main">
        <Header
          title={viewMeta.title}
          breadcrumb={viewMeta.breadcrumb}
          subtitle={headerSubtitle}
          jobId={hasActiveJob && activeView === "dashboard" ? job.jobId : null}
          phase={job.phase}
        />

        <main className="app-content">
          {activeView === "settings" && <SettingsPanel />}

          {activeView === "history" && (
            <div className="page-container page-container--wide">
              <JobHistory />
            </div>
          )}

          {activeView === "upload" && (
            <div className="upload-page-center">
              <div className="upload-page upload-page--centered">
                <div className="upload-hero upload-hero--centered">
                  <span className="upload-hero-eyebrow">New job</span>
                  <h2>Generate work instructions from video</h2>
                  <p>
                    Upload a factory walkthrough. The pipeline extracts frames, transcribes
                    audio, analyzes visuals, and streams structured steps in real time.
                  </p>
                </div>
                <VideoUpload onUpload={handleUpload} disabled={isBusy} large />
              </div>
            </div>
          )}

          {activeView === "dashboard" && (
            <div className="page-container page-container--wide">
              <div className="workspace">
                {job.error && (
                  <div className="alert alert-error">
                    <AlertCircle size={16} />
                    <span style={{ flex: 1 }}>{job.error}</span>
                    <button type="button" className="btn btn-secondary btn-sm" onClick={handleNewJob}>
                      Try again
                    </button>
                  </div>
                )}

                <div className="workspace-toolbar">
                  <div className="workspace-meta">
                    <h2>{toolbarTitle}</h2>
                    {job.steps.length > 0 && (
                      <>
                        <span className="meta-divider" />
                        <span>{job.steps.length} steps</span>
                      </>
                    )}
                  </div>
                  <button type="button" className="btn btn-secondary btn-sm" onClick={handleNewJob}>
                    <Plus size={14} />
                    New job
                  </button>
                </div>

                <JobProgress
                  status={progressStatus}
                  progress={progressValue}
                  stepCount={job.steps.length}
                  active={!isIdle && job.phase !== "complete"}
                />

                {job.phase === "complete" && (
                  <CompletionPanel
                    complete={job.complete}
                    filename={job.filename}
                    onNewJob={handleNewJob}
                  />
                )}

                {showSkeletonSteps && (
                  <div className="steps-workspace">
                    <DashboardSkeleton />
                  </div>
                )}

                {job.steps.length > 0 && (
                  <div className="steps-workspace">
                    <div className="steps-list-panel">
                      <div className="steps-list-header">
                        <h3>Steps</h3>
                        <span className="steps-count">{job.steps.length}</span>
                      </div>
                      <div className="steps-list">
                        {job.steps.map((step) => (
                          <StepCard
                            key={step.step_index}
                            step={step}
                            selected={selectedStep?.step_index === step.step_index}
                            onSelect={setSelectedStep}
                          />
                        ))}
                      </div>
                    </div>
                    <div className="steps-detail-panel">
                      <StepDetail
                        step={selectedStep}
                        loadingClip={
                          job.phase === "processing" &&
                          selectedStep &&
                          !selectedStep.video_clip_url
                        }
                        loadingWi={
                          job.phase === "processing" &&
                          selectedStep &&
                          !selectedStep.WI
                        }
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
