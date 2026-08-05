export function SkeletonBlock({ className = "", style }) {
  return <div className={`skeleton-block ${className}`.trim()} style={style} aria-hidden="true" />;
}

export function SkeletonLine({ width = "100%", className = "" }) {
  return (
    <SkeletonBlock
      className={`skeleton-line ${className}`.trim()}
      style={{ width }}
    />
  );
}

export function StepCardSkeleton() {
  return (
    <div className="step-card-skeleton">
      <div className="step-card-skeleton-header">
        <SkeletonLine width="4rem" />
        <SkeletonLine width="5.5rem" />
      </div>
      <SkeletonLine width="95%" />
      <SkeletonLine width="78%" />
    </div>
  );
}

export function DashboardSkeleton() {
  return (
    <>
      <div className="steps-list-panel">
        <div className="steps-list-header">
          <h3>Steps</h3>
          <SkeletonBlock className="skeleton-pill" />
        </div>
        <div className="steps-list">
          <StepCardSkeleton />
          <StepCardSkeleton />
          <StepCardSkeleton />
        </div>
      </div>
      <div className="steps-detail-panel">
        <div className="step-detail step-detail-skeleton">
          <div className="detail-top">
            <SkeletonLine width="3.5rem" />
            <SkeletonLine width="8rem" className="skeleton-line-sm" />
          </div>
          <div className="detail-body">
            <div className="skeleton-video" />
            <div className="detail-section">
              <SkeletonLine width="7rem" className="skeleton-line-sm" />
              <div className="skeleton-wi">
                <SkeletonLine width="100%" />
                <SkeletonLine width="92%" />
                <SkeletonLine width="88%" />
                <SkeletonLine width="65%" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
