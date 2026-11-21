import { afterEach, expect, test, vi } from 'vitest';
import { api } from '../src/api';

afterEach(() => { vi.unstubAllGlobals(); vi.useRealTimers(); });

test('structured validation errors become readable messages instead of crashing React', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: [{ msg: 'Rating must be at most 5' }] }), { status: 422 })));
  await expect(api.createSession([{ movie_id: 1, title: 'Example', rating: 6 }])).rejects.toThrow('Rating must be at most 5');
});

test('a stalled request times out and releases the caller', async () => {
  vi.useFakeTimers();
  vi.stubGlobal('fetch', vi.fn((_url, init: RequestInit) => new Promise((_resolve, reject) => {
    init.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
  })));
  const pending = api.searchMovies('Arrival');
  const assertion = expect(pending).rejects.toThrow(/timed out/i);
  await vi.advanceTimersByTimeAsync(45000);
  await assertion;
});
