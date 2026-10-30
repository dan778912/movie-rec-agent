import { useState } from "react";
import "./App.css";
import { api, ApiError, type RatingInput, type SessionOut } from "./api";
import { Conversation } from "./components/Conversation";
import { HistoryPanel } from "./components/HistoryPanel";
import { RatingForm } from "./components/RatingForm";

type View = "rate" | "chat" | "history";

function App() {
  const [view, setView] = useState<View>("rate");
  const [session, setSession] = useState<SessionOut | null>(null);
  const [userId, setUserId] = useState<number | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [historyRefreshKey, setHistoryRefreshKey] = useState(0);

  const startSession = async (ratings: RatingInput[]) => {
    setSubmitting(true);
    setError(null);
    try {
      const created = await api.createSession(ratings, userId ?? undefined);
      setSession(created);
      setUserId(created.user_id);
      setView("chat");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not start a session.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="app">
      <header>
        <h1>Movie Recommendation Agent</h1>
        <nav>
          <button className={view === "rate" ? "active" : ""} onClick={() => setView("rate")}>
            Rate movies
          </button>
          <button className={view === "chat" ? "active" : ""} onClick={() => setView("chat")} disabled={!session}>
            Chat
          </button>
          <button className={view === "history" ? "active" : ""} onClick={() => setView("history")} disabled={!userId}>
            History
          </button>
        </nav>
      </header>
      <main>
        {error && <div className="bubble error">{error}</div>}
        {view === "rate" && <RatingForm onSubmit={startSession} submitting={submitting} />}
        {view === "chat" && session && (
          <Conversation session={session} onCompleted={() => setHistoryRefreshKey((key) => key + 1)} />
        )}
        {view === "history" && userId && <HistoryPanel userId={userId} refreshKey={historyRefreshKey} />}
      </main>
    </div>
  );
}

export default App;
