import { useRef, useState } from 'react';
import './App.css';
import { Conversation } from './components/Conversation';
import { HistoryPanel } from './components/HistoryPanel';
import { RatingForm } from './components/RatingForm';
import { useRecommendation } from './useRecommendation';

type View = 'rate' | 'chat' | 'history';

function App() {
  const [view, setView] = useState<View>('rate');
  const rec = useRecommendation();
  const mainRef = useRef<HTMLElement>(null);
  const demo = import.meta.env.VITE_SERVICE_B_MODE !== 'http';
  function navigate(next: View) {
    mainRef.current?.focus();
    setView(next);
  }

  return (
    <div className="app">
      <a className="skip-link" href="#main">Skip to content</a>
      <header className="site-header">
        <button className="wordmark" onClick={() => navigate('rate')} aria-label="Movie Recommendation Agent home">
          <span className="brand-mark" aria-hidden="true">m.</span><span>Movie Recommendation Agent</span>
        </button>
        <nav aria-label="Main navigation">
          {rec.session && <button aria-current={view === 'rate' ? 'page' : undefined} onClick={() => navigate('rate')}>Edit ratings</button>}
          {rec.session && <button aria-current={view === 'chat' ? 'page' : undefined} onClick={() => navigate('chat')}>Current recommendations</button>}
          {rec.session && <button className="history-link" aria-current={view === 'history' ? 'page' : undefined} onClick={() => navigate('history')}>History</button>}
        </nav>
      </header>
      {demo && <div className="demo-notice"><span className="status-dot" aria-hidden="true" />Demo mode · Recommendations are sample output, not personalized model results.</div>}
      <main id="main" ref={mainRef} tabIndex={-1}>
        <section hidden={view !== 'rate'} aria-label="Rate movies">
          <RatingForm onSubmit={ratings => void rec.start(ratings, () => navigate('chat'))} submitting={rec.busy} />
          {view === 'rate' && rec.creationError && <div className="error-message" role="alert">{rec.creationError}</div>}
        </section>
        {view === 'chat' && rec.session && <Conversation
          messages={rec.messages} answer={rec.answer} onAnswerChange={rec.setAnswer}
          awaitingAnswer={rec.awaitingAnswer} loading={rec.busy} error={rec.error}
          onSend={rec.send} onRetry={rec.retry} onEdit={() => navigate('rate')}
        />}
        {view === 'history' && rec.session && <HistoryPanel userId={rec.session.user_id} refreshKey={rec.historyVersion} onStart={() => navigate('rate')} />}
      </main>
      <footer className="site-footer"><span>MovieLens 25M catalog</span></footer>
    </div>
  );
}
export default App;
