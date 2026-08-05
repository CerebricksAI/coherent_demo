import { getApiBase } from "../api/client";

export default function SettingsPanel() {
  const apiBase = getApiBase() || "(proxy → localhost:8001)";

  return (
    <div className="settings-panel page-container">
      <h3>Settings</h3>
      <p>API and environment configuration for this deployment.</p>
      <div className="settings-grid">
        <label>
          <span>API base URL</span>
          <input type="text" value={apiBase} readOnly />
          <small>Set VITE_API_BASE_URL in .env for production builds</small>
        </label>
        <label>
          <span>WebSocket endpoint</span>
          <input
            type="text"
            value={`${apiBase || window.location.origin}/api/v1/jobs/{id}/stream`}
            readOnly
          />
        </label>
      </div>
      <div className="settings-note">
        <p>
          <strong>Development:</strong> Run <code>python -m api</code> on port{" "}
          <strong>8001</strong>, then <code>npm run dev</code> in the UI folder.
        </p>
      </div>
    </div>
  );
}
