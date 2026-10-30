import { useEffect, useState } from "react";
import { api, ApiError, type HistorySessionOut } from "../api";

interface Props {
  userId: number;
  refreshKey: number;
}

export function HistoryPanel({ userId, refreshKey }: Props) {
  const [sessions, setSessions] = useState<HistorySessionOut[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<number | null>(null);

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
      <div className="notes-list">
        {sessions.map((session) => {
          const expanded = expandedId === session.session_id;
          const topTitle = session.latest_recommendation?.[0];
          const headline = topTitle
            ? (topTitle.title ?? `Movie ${topTitle.movie_id}`)
            : `Session #${session.session_id}`;
          return (
            <div className="notes-entry" key={session.session_id}>
              <button
                type="button"
                className="notes-row"
                onClick={() => setExpandedId(expanded ? null : session.session_id)}
                aria-expanded={expanded}
              >
                <span className="notes-row-meta">{new Date(session.created_at).toLocaleDateString()}</span>
                <span className="notes-row-title">{headline}</span>
                <span className={`notes-tag status-${session.status}`}>{session.status}</span>
              </button>
              {expanded && (
                <div className="notes-detail">
                  {session.latest_recommendation ? (
                    <ul className="plain-list">
                      {session.latest_recommendation.map((item) => (
                        <li key={item.movie_id}>
                          #{item.rank} {item.title ?? `Movie ${item.movie_id}`} ({item.score.toFixed(2)})
                        </li>
                      ))}
                    </ul>
                  ) : (
                    "No recommendations yet for this session."
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
