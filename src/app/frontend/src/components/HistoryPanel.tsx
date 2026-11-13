import { useEffect, useState } from 'react';
import { api, ApiError, type HistorySessionOut } from '../api';
import { RecommendationList } from './RecommendationList';

interface Props { userId: number; refreshKey: number; onStart: () => void }
const statusLabels: Record<string, string> = { completed: 'Ready', awaiting_clarification: 'Awaiting your answer', active: 'In progress' };

export function HistoryPanel({ userId, refreshKey, onStart }: Props) {
  const [state, setState] = useState<{ sessions: HistorySessionOut[] | null; error: string | null }>({ sessions: null, error: null });
  const [attempt, setAttempt] = useState(0);
  const [expandedId, setExpandedId] = useState<number | null>(null);
  useEffect(() => {
    let cancelled = false;
    api.getHistory(userId).then(sessions => { if (!cancelled) setState({ sessions, error: null }); })
      .catch(err => { if (!cancelled) setState({ sessions: null, error: err instanceof ApiError ? err.message : 'Could not load your history.' }); });
    return () => { cancelled = true; };
  }, [userId, refreshKey, attempt]);
  return <section className="history">
    <div className="page-intro compact"><p className="eyebrow">Your collection</p><h1>Past recommendations.</h1></div>
    {state.error ? <div className="error-message" role="alert"><p>{state.error}</p><button className="link-button" onClick={() => { setState({ sessions: null, error: null }); setAttempt(n => n + 1); }}>Retry history</button></div>
      : state.sessions === null ? <p role="status" className="loading-state">Loading history…</p>
      : state.sessions.length === 0 ? <div className="empty-state"><p>Your recommendations will appear here.</p><button className="primary" onClick={onStart}>Rate a movie<span aria-hidden="true">↗</span></button></div>
      : <div className="history-list">{state.sessions.map(session => {
        const expanded = expandedId === session.session_id;
        const first = session.latest_recommendation?.[0];
        return <article className="history-entry" key={session.session_id}>
          <button className="history-row" aria-expanded={expanded} aria-controls={`history-${session.session_id}`} onClick={() => setExpandedId(expanded ? null : session.session_id)}>
            <span className="history-date">{new Date(session.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}</span>
            <span className="history-title">{first?.title ?? 'Recommendation in progress'}</span>
            <span className="history-status">{statusLabels[session.status] ?? 'In progress'}</span><span aria-hidden="true">{expanded ? '−' : '+'}</span>
          </button>
          {expanded && <div id={`history-${session.session_id}`} className="history-detail">{session.latest_recommendation?.length ? <RecommendationList items={session.latest_recommendation} /> : <p className="muted">No recommendations saved yet. Return to your current recommendations to continue.</p>}</div>}
        </article>;
      })}</div>}
  </section>;
}
