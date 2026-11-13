import type { Message } from '../useRecommendation';
import { RecommendationList } from './RecommendationList';

interface Props {
  messages: Message[]; answer: string; onAnswerChange: (value: string) => void;
  awaitingAnswer: boolean; loading: boolean; error: string | null;
  onSend: () => void; onRetry: () => void; onEdit: () => void;
}

export function Conversation({ messages, answer, onAnswerChange, awaitingAnswer, loading, error, onSend, onRetry, onEdit }: Props) {
  const complete = messages.some(message => message.kind === 'recommendations');
  return <section className="conversation" aria-busy={loading}>
    <div className="page-intro compact"><p className="eyebrow">Your next watch</p><h1>{complete ? 'Something to watch.' : awaitingAnswer ? 'One more thing.' : 'Finding your next watch.'}</h1>
      {complete && <p className="intro-copy">Open a title to see why it was suggested.</p>}
    </div>
    <div className="transcript">{messages.map((message, index) => message.kind === 'recommendations'
      ? <RecommendationList key={index} items={message.items} />
      : <div key={index} className={`message ${message.kind}`}><span className="message-label">{message.kind === 'question' ? 'Question' : 'You'}</span><p>{message.text}</p></div>)}
    </div>
    {loading && <p className="loading-state" role="status"><span className="loading-dot" />{awaitingAnswer ? 'Updating your recommendations…' : 'Preparing recommendations…'}</p>}
    {error && <div className="error-message" role="alert"><p>{error}</p>{!awaitingAnswer && <button type="button" className="link-button" disabled={loading} onClick={onRetry}>Try again</button>}</div>}
    {awaitingAnswer && <form className="answer-form" onSubmit={event => { event.preventDefault(); if (!loading && answer.trim()) onSend(); }}>
      <label className="sr-only" htmlFor="clarification-answer">Your answer</label>
      <input id="clarification-answer" type="text" placeholder="Your answer…" value={answer} onChange={event => onAnswerChange(event.target.value)} disabled={loading} autoFocus />
      <button className="primary" type="submit" disabled={loading || !answer.trim()}>Send<span aria-hidden="true">↑</span></button>
    </form>}
    {complete && <div className="result-actions"><button className="primary" onClick={onEdit}>Edit ratings<span aria-hidden="true">↗</span></button></div>}
  </section>;
}
