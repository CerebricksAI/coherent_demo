import { useEffect, useState } from "react";
import { Clock, FileVideo } from "lucide-react";
import { loadJobHistory } from "../utils/helpers";

export default function JobHistory({ onOpenJob }) {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    setHistory(loadJobHistory());
  }, []);

  if (history.length === 0) {
    return (
      <div className="empty-state">
        <FileVideo size={48} strokeWidth={1} />
        <h3>No job history yet</h3>
        <p>Completed jobs will appear here for quick reference.</p>
      </div>
    );
  }

  return (
    <div className="history-table-wrap">
      <table className="history-table">
        <thead>
          <tr>
            <th>Video</th>
            <th>Job ID</th>
            <th>Steps</th>
            <th>Completed</th>
          </tr>
        </thead>
        <tbody>
          {history.map((job) => (
            <tr key={job.jobId}>
              <td>{job.filename}</td>
              <td className="history-job-id-cell">
                <button
                  type="button"
                  className="link-btn history-job-id"
                  onClick={() => onOpenJob?.(job.jobId)}
                  title={job.jobId}
                >
                  <code>{job.jobId}</code>
                </button>
              </td>
              <td>{job.stepCount}</td>
              <td>
                <Clock size={12} style={{ marginRight: 4, verticalAlign: -2 }} />
                {new Date(job.completedAt).toLocaleString()}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
