import { useRef, useState } from 'react';
import { api, ApiError, type RatingInput, type SessionOut, type TurnResponse, type RecommendationItemOut } from './api';

export type Message =
  | { kind: 'question'; text: string }
  | { kind: 'answer'; text: string }
  | { kind: 'recommendations'; items: RecommendationItemOut[] };

// Mutating requests start only in event handlers, never in mount effects.
// State lives above navigation so a view change cannot repeat a model turn.
export function useRecommendation() {
  const [session, setSession] = useState<SessionOut | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [answer, setAnswer] = useState('');
  const [awaitingAnswer, setAwaitingAnswer] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [creationError, setCreationError] = useState<string | null>(null);
  const [historyVersion, setHistoryVersion] = useState(0);
  const inFlight = useRef(false);
  const retryAction = useRef<'turn' | 'answer'>('turn');
  const lastSubmittedAnswer = useRef<string | undefined>(undefined);
  const safeAnswerRetry = useRef(false);

  function accept(result: TurnResponse, submittedAnswer?: string) {
    setError(null);
    setMessages(current => [
      ...current,
      ...(submittedAnswer ? [{ kind: 'answer' as const, text: submittedAnswer }] : []),
      result.action === 'ask'
        ? { kind: 'question' as const, text: result.question }
        : { kind: 'recommendations' as const, items: result.items },
    ]);
    setAwaitingAnswer(result.action === 'ask');
    setAnswer('');
    setHistoryVersion(v => v + 1);
  }

  function fail(err: unknown) {
    safeAnswerRetry.current = err instanceof ApiError && err.status === 422;
    setError(err instanceof ApiError ? err.message : 'The request could not finish. Your input is still here.');
  }

  async function start(ratings: RatingInput[], onCreated: () => void) {
    if (inFlight.current) return;
    inFlight.current = true;
    setBusy(true);
    setCreationError(null);
    // Creation failures must not change the current conversation or its recovery state.
    try {
      let created: SessionOut;
      try {
        created = await api.createSession(ratings, session?.user_id);
      } catch (err) {
        setCreationError(err instanceof ApiError ? err.message : 'Could not create a recommendation. Your ratings are still here.');
        return;
      }
      setError(null);
      setSession(created);
      setMessages([]);
      setAnswer('');
      setAwaitingAnswer(false);
      retryAction.current = 'turn';
      lastSubmittedAnswer.current = undefined;
      safeAnswerRetry.current = false;
      onCreated();
      accept(await api.advanceTurn(created.id));
    } catch (err) {
      fail(err);
    } finally {
      inFlight.current = false;
      setBusy(false);
    }
  }

  async function advance(submittedAnswer?: string, recover = false) {
    if (!session || inFlight.current) return;
    inFlight.current = true;
    setBusy(true);
    setError(null);
    retryAction.current = submittedAnswer ? 'answer' : 'turn';
    if (!recover) lastSubmittedAnswer.current = submittedAnswer;
    try {
      // A failed response may follow a successful server write. Read status
      // before retrying so a committed answer is not posted a second time.
      const current = recover ? await api.getSession(session.id) : null;
      if (current?.status === 'completed') {
        accept(await api.getRecommendations(session.id), lastSubmittedAnswer.current);
      } else if (current?.status === 'active') {
        accept(await api.advanceTurn(session.id), lastSubmittedAnswer.current);
      } else if (recover && current?.status === 'awaiting_clarification' && (!submittedAnswer || !safeAnswerRetry.current)) {
        setError('We could not confirm which question is pending. Your answer is kept here. Edit your ratings to start a new recommendation.');
      } else {
        lastSubmittedAnswer.current = submittedAnswer;
        const result = submittedAnswer
          ? await api.answerClarification(session.id, submittedAnswer)
          : await api.advanceTurn(session.id);
        accept(result, submittedAnswer);
      }
    } catch (err) {
      fail(err);
    } finally {
      inFlight.current = false;
      setBusy(false);
    }
  }

  return { session, messages, answer, setAnswer, awaitingAnswer, busy, error, creationError, historyVersion, start,
    send: () => { if (answer.trim()) void advance(answer.trim(), Boolean(error)); },
    retry: () => void advance(retryAction.current === 'answer' ? answer.trim() : undefined, true),
  };
}
