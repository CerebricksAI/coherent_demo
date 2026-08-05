import {
  ClipboardList,
  LayoutDashboard,
  Settings,
  Upload,
} from "lucide-react";

const NAV_MAIN = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "upload", label: "New Job", icon: Upload },
  { id: "history", label: "Job History", icon: ClipboardList },
];

const NAV_SYSTEM = [
  { id: "settings", label: "Settings", icon: Settings },
];

export default function Sidebar({ activeView, onNavigate, apiOnline }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="brand-icon">
          <img src="/logo.png" alt="QSAID logo" className="brand-logo" />
        </div>
        <div>
          <div className="brand-title">QSAID</div>
          <div className="brand-sub">Work Instructions</div>
        </div>
      </div>

      <nav className="sidebar-nav">
        <p className="nav-section-label">Workspace</p>
        {NAV_MAIN.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            type="button"
            className={`nav-item ${activeView === id ? "active" : ""}`}
            onClick={() => onNavigate(id)}
          >
            <Icon size={17} strokeWidth={activeView === id ? 2.25 : 2} />
            <span>{label}</span>
          </button>
        ))}

        <p className="nav-section-label">System</p>
        {NAV_SYSTEM.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            type="button"
            className={`nav-item ${activeView === id ? "active" : ""}`}
            onClick={() => onNavigate(id)}
          >
            <Icon size={17} strokeWidth={activeView === id ? 2.25 : 2} />
            <span>{label}</span>
          </button>
        ))}
      </nav>

      <div className="sidebar-footer">
        <div className={`api-status ${apiOnline ? "online" : "offline"}`}>
          <span className="api-status-dot" />
          <span>{apiOnline ? "API connected" : "API offline"}</span>
        </div>
        <p className="sidebar-version">v1.0</p>
      </div>
    </aside>
  );
}
