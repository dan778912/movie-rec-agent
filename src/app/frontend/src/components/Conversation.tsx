import { useEffect, useState } from "react";
import { api, ApiError, type RecommendationItemOut, type SessionOut } from "../api";

interface Props {
  session: SessionOut;
  onCompleted: () => void;
}

type Message =
  | { kind: "question"; text: string }
  | { kind: "answer"; text: string }
  | { kind: "recommendations"; items: RecommendationItemOut[] }
  | { kind: "error"; text: string };

export function Conversation({ session, onCompleted }: Props) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [awaitingAnswer, setAwaitingAnswer] = useState(false);
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    api
      .advanceTurn(session.id)
      .then((result) => {
        if (cancelled) return;
        if (result.action === "ask") {
          setMessages([{ kind: "question", text: result.question }]);
          setAwaitingAnswer(true);
        } else {
          setMessages([{ kind: "recommendations", items: result.items }]);
          onCompleted();
        }
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setMessages([{ kind: "error", text: error instanceof ApiError ? error.message : "Something went wrong." }]);
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
    // Only re-run when the session identity changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session.id]);

  const submitAnswer = async (event: React.FormEvent) => {
    event.preventDefault();
    const text = answer.trim();
    if (!text) return;
    setMessages((current) => [...current, { kind: "answer", text }]);
    setAnswer("");
    setAwaitingAnswer(false);
    setLoading(true);
    try {
      const result = await api.answerClarification(session.id, text);
      if (result.action === "ask") {
        setMessages((current) => [...current, { kind: "question", text: result.question }]);
        setAwaitingAnswer(true);
      } else {
        setMessages((current) => [...current, { kind: "recommendations", items: result.items }]);
        onCompleted();
      }
    } catch (error) {
      setMessages((current) => [
        ...current,
        { kind: "error", text: error instanceof ApiError ? error.message : "Something went wrong." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="conversation">
      <h2>Session #{session.id}</h2>
      <div className="transcript">
        {messages.map((message, index) => (
          <MessageBubble key={index} message={message} />
        ))}
        {loading && <div className="bubble system">Thinking...</div>}
      </div>
      {awaitingAnswer && !loading && (
        <form className="answer-form" onSubmit={submitAnswer}>
          <input
            type="text"
            placeholder="Your answer..."
            value={answer}
            onChange={(e) => setAnswer(e.target.value)}
            autoFocus
          />
          <button type="submit" disabled={!answer.trim()}>
            Send
          </button>
        </form>
      )}
    </div>
  );
}

function MessageBubble({ message }: { message: Message }) {
  if (message.kind === "question") {
    return <div className="bubble system">{message.text}</div>;
  }
  if (message.kind === "answer") {
    return <div className="bubble user">{message.text}</div>;
  }
  if (message.kind === "error") {
    return <div className="bubble error">{message.text}</div>;
  }
  return (
    <div className="recommendations">
      {message.items.map((item) => (
        <div className="recommendation-card" key={item.movie_id}>
          <div className="recommendation-rank">#{item.rank}</div>
          <div className="recommendation-body">
            <div className="recommendation-title">{item.title ?? `Movie ${item.movie_id}`}</div>
            <div className="recommendation-score">score {item.score.toFixed(2)}</div>
            {item.explanation && <p className="recommendation-explanation">{item.explanation}</p>}
          </div>
        </div>
      ))}
    </div>
  );
}
