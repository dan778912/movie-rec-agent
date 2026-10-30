import { useEffect, useState } from "react";
import { api, ApiError, type HistorySessionOut } from "../api";

interface Props {
  userId: number;
  refreshKey: number;
}

export function HistoryPanel({ userId, refreshKey }: Props) {
  const [sessions, setSessions] = useState<HistorySessionOut[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .getHistory(userId)
      .then((result) => !cancelled && setSessions(result))
      .catch((err: unknown) => !cancelled && setError(err instanceof ApiError ? err.message : "Failed to load history."));
    return () => {
      cancelled = true;
    };
  }, [userId, refreshKey]);

  if (error) return <div className="bubble error">{error}</div>;
  if (sessions === null) return <p>Loading history...</p>;
  if (sessions.length === 0) return <p>No past sessions yet.</p>;

  return (
    <div className="history">
      <h2>History</h2>
      {sessions.map((session) => (
        <div className="history-session" key={session.session_id}>
          <div className="history-session-header">
            <span>Session #{session.session_id}</span>
            <span className={`status status-${session.status}`}>{session.status}</span>
            <span className="history-session-time">{new Date(session.created_at).toLocaleString()}</span>
          </div>
          {session.latest_recommendation ? (
            <ul className="history-items">
              {session.latest_recommendation.map((item) => (
                <li key={item.movie_id}>
                  #{item.rank} {item.title ?? `Movie ${item.movie_id}`} ({item.score.toFixed(2)})
                </li>
              ))}
            </ul>
          ) : (
            <p className="hint">No recommendations yet for this session.</p>
          )}
        </div>
      ))}
    </div>
  );
}
