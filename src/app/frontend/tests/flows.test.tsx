import { StrictMode } from 'react';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from '../src/App';
import { MoviePicker } from '../src/components/MoviePicker';
import { api, ApiError } from '../src/api';

const movie = { movie_id: 1, title: 'Arrival (2016)', genres: 'Drama|Sci-Fi', rating_avg: 4.2, rating_count: 1200 };
const session = { id: 10, user_id: 5, status: 'active' as const, created_at: '2026-11-12T12:00:00Z' };
const question = { action: 'ask' as const, session_id: 10, clarification_id: 7, question: 'What mood are you in?' };
const recommendation = { action: 'recommend' as const, session_id: 10, recommendation_id: 3, items: [
  { movie_id: 2, title: 'Contact (1997)', genres: 'Drama|Sci-Fi', rating_avg: 4.1, rating_count: 700, rank: 1, score: 0.9, explanation: 'A thoughtful science fiction story.' },
] };

beforeEach(() => {
  vi.spyOn(api, 'searchMovies').mockResolvedValue([movie]);
  vi.spyOn(api, 'createSession').mockResolvedValue(session);
  vi.spyOn(api, 'advanceTurn').mockResolvedValue(question);
  vi.spyOn(api, 'getHistory').mockResolvedValue([]);
  vi.spyOn(api, 'getSession').mockResolvedValue({ ...session, status: 'awaiting_clarification' });
  vi.spyOn(api, 'answerClarification').mockResolvedValue(recommendation);
});
afterEach(() => { cleanup(); vi.restoreAllMocks(); });

async function selectMovie() {
  const input = screen.getByPlaceholderText(/search.*title/i);
  await userEvent.type(input, 'Arrival');
  await userEvent.click(await screen.findByRole('button', { name: /Arrival/ }));
}
async function start() {
  await selectMovie();
  await userEvent.click(screen.getByRole('button', { name: /^(Start|Get recommendations)$/i }));
  await screen.findByText(question.question);
}

test('a session advances once under StrictMode and survives history navigation', async () => {
  render(<StrictMode><App /></StrictMode>);
  await start();
  await userEvent.click(screen.getByRole('button', { name: /^History$/i }));
  await userEvent.click(screen.getByRole('button', { name: /^(Chat|Current recommendations)$/i }));
  expect(screen.getByText(question.question)).toBeTruthy();
  expect(api.advanceTurn).toHaveBeenCalledTimes(1);
});

test('rated movies and rating values survive a trip to the conversation', async () => {
  render(<App />);
  await start();
  await userEvent.click(screen.getByRole('button', { name: /^(Rate movies|Edit ratings)$/i }));
  expect(screen.getByText(movie.title)).toBeTruthy();
  expect((screen.getByRole('combobox', { name: /rating/i }) as HTMLSelectElement).value).toBe('5');
});

test('failed clarification retains the answer and can recover without duplicate transcript entries', async () => {
  vi.mocked(api.answerClarification).mockRejectedValueOnce(new ApiError(422, 'Please revise your answer'));
  render(<App />);
  await start();
  const input = screen.getByPlaceholderText(/your answer/i);
  await userEvent.type(input, 'Something thoughtful');
  await userEvent.click(screen.getByRole('button', { name: /^Send$/i }));
  await waitFor(() => expect(screen.getByRole('alert')).toBeTruthy());
  expect((screen.getByPlaceholderText(/your answer/i) as HTMLInputElement).value).toBe('Something thoughtful');
  await userEvent.click(screen.getByRole('button', { name: /^Send$/i }));
  await screen.findByRole('button', { name: /Contact/ });
  expect(screen.getAllByText('Something thoughtful')).toHaveLength(1);
  expect(screen.queryByText(/match 0.90/)).toBeNull();
});

test('search failure is visible and retry finds a movie', async () => {
  vi.mocked(api.searchMovies).mockRejectedValueOnce(new Error('offline'));
  render(<MoviePicker selected={null} onSelect={() => {}} />);
  await userEvent.type(screen.getByPlaceholderText(/search.*title/i), 'Arrival');
  await screen.findByRole('alert');
  await userEvent.click(screen.getByRole('button', { name: /retry/i }));
  expect(await screen.findByRole('button', { name: /Arrival/ })).toBeTruthy();
});

test('late search responses cannot replace results for a newer query', async () => {
  let resolveOld!: (value: typeof movie[]) => void;
  vi.mocked(api.searchMovies).mockImplementation((q) => q === 'Old'
    ? new Promise(resolve => { resolveOld = resolve; })
    : Promise.resolve([{ ...movie, movie_id: 3, title: 'New Movie (2000)' }]));
  render(<MoviePicker selected={null} onSelect={() => {}} />);
  const input = screen.getByPlaceholderText(/search.*title/i);
  fireEvent.change(input, { target: { value: 'Old' } });
  await waitFor(() => expect(resolveOld).toBeTypeOf('function'));
  fireEvent.change(input, { target: { value: 'New' } });
  await screen.findByRole('button', { name: /New Movie/ });
  await act(async () => resolveOld([{ ...movie, title: 'Old Movie (1990)' }]));
  expect(screen.queryByRole('button', { name: /Old Movie/ })).toBeNull();
  expect(screen.getByRole('button', { name: /New Movie/ })).toBeTruthy();
});

test('duplicate movie selections cannot be submitted', async () => {
  render(<App />);
  await selectMovie();
  await userEvent.click(screen.getByRole('button', { name: /add another/i }));
  await selectMovie();
  expect((screen.getByRole('button', { name: /^(Start|Get recommendations)$/i }) as HTMLButtonElement).disabled).toBe(true);
  expect(api.createSession).not.toHaveBeenCalled();
});

test('keyboard users can move into movie search results and select a title', async () => {
  const onSelect = vi.fn();
  render(<MoviePicker selected={null} onSelect={onSelect} />);
  const input = screen.getByPlaceholderText(/search.*title/i);
  await userEvent.type(input, 'Arrival');
  await screen.findByRole('button', { name: /Arrival/ });
  await userEvent.keyboard('{ArrowDown}{Enter}');
  expect(onSelect).toHaveBeenCalledWith(movie);
});


test('an ambiguous lost answer response cannot post the old answer to a new pending question', async () => {
  vi.mocked(api.answerClarification).mockRejectedValueOnce(new TypeError('Failed to fetch'));
  render(<App />);
  await start();
  await userEvent.type(screen.getByPlaceholderText(/your answer/i), 'Thoughtful');
  await userEvent.click(screen.getByRole('button', { name: /^Send$/i }));
  await screen.findByRole('alert');
  await userEvent.click(screen.getByRole('button', { name: /^Send$/i }));
  await waitFor(() => expect(api.getSession).toHaveBeenCalled());
  expect(api.answerClarification).toHaveBeenCalledTimes(1);
  expect(screen.getByRole('alert').textContent).toMatch(/new recommendation/i);
});

test('a lost completed response recovers saved recommendations without reposting the answer', async () => {
  vi.mocked(api.answerClarification).mockRejectedValueOnce(new TypeError('Failed to fetch'));
  vi.mocked(api.getSession).mockResolvedValue({ ...session, status: 'completed' });
  vi.spyOn(api, 'getRecommendations').mockResolvedValue(recommendation);
  render(<App />);
  await start();
  await userEvent.type(screen.getByPlaceholderText(/your answer/i), 'Original answer');
  await userEvent.click(screen.getByRole('button', { name: /^Send$/i }));
  await screen.findByRole('alert');
  await userEvent.clear(screen.getByPlaceholderText(/your answer/i));
  await userEvent.type(screen.getByPlaceholderText(/your answer/i), 'Edited after failure');
  await userEvent.click(screen.getByRole('button', { name: /^Send$/i }));
  await screen.findByRole('button', { name: /Contact/ });
  expect(api.answerClarification).toHaveBeenCalledTimes(1);
  expect(screen.getByText('Original answer')).toBeTruthy();
  expect(screen.queryByText('Edited after failure')).toBeNull();
});

test('double submit creates only one session while the request is pending', async () => {
  let resolveSession!: (value: typeof session) => void;
  vi.mocked(api.createSession).mockImplementation(() => new Promise(resolve => { resolveSession = resolve; }));
  render(<App />);
  await selectMovie();
  const button = screen.getByRole('button', { name: /^Get recommendations$/i });
  await userEvent.dblClick(button);
  expect(api.createSession).toHaveBeenCalledTimes(1);
  await act(async () => resolveSession(session));
  await screen.findByText(question.question);
});

for (const previousFailure of [false, true]) {
  test(`failed session creation preserves the old conversation${previousFailure ? ' and its validation retry' : ''}`, async () => {
    render(<App />);
    await start();
    await userEvent.type(screen.getByPlaceholderText(/your answer/i), 'Something thoughtful');
    if (previousFailure) {
      vi.mocked(api.answerClarification).mockRejectedValueOnce(new ApiError(422, 'Please revise your answer'));
      await userEvent.click(screen.getByRole('button', { name: /^Send$/i }));
      await screen.findByRole('alert');
    }
    await userEvent.click(screen.getByRole('button', { name: /^Edit ratings$/i }));
    vi.mocked(api.createSession).mockRejectedValueOnce(new ApiError(503, 'Creation unavailable'));
    await userEvent.click(screen.getByRole('button', { name: /^Get recommendations$/i }));
    expect((await screen.findByRole('alert')).textContent).toContain('Creation unavailable');
    await userEvent.click(screen.getByRole('button', { name: /^Current recommendations$/i }));
    if (previousFailure) expect(screen.getByRole('alert').textContent).toContain('Please revise your answer');
    await userEvent.click(screen.getByRole('button', { name: /^Send$/i }));
    await screen.findByRole('button', { name: /Contact/ });
    expect(screen.getAllByText('Something thoughtful')).toHaveLength(1);
    expect(api.answerClarification).toHaveBeenLastCalledWith(10, 'Something thoughtful');
    expect(api.answerClarification).toHaveBeenCalledTimes(previousFailure ? 2 : 1);
    if (!previousFailure) expect(api.getSession).not.toHaveBeenCalled();
  });
}
