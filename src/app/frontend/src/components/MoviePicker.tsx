import { useEffect, useId, useRef, useState, type KeyboardEvent } from 'react';
import { api, type MovieSearchResult } from '../api';
import { formatRating, primaryGenres, splitYear } from '../movieTitle';

interface Props { selected: MovieSearchResult | null; onSelect: (movie: MovieSearchResult | null) => void; disabled?: boolean }

export function MoviePicker({ selected, onSelect, disabled = false }: Props) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<MovieSearchResult[]>([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);
  const [attempt, setAttempt] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const dismissed = useRef(false);
  const id = useId();

  useEffect(() => {
    if (selected || query.trim().length < 2 || disabled) return;
    let cancelled = false;
    const timer = setTimeout(() => {
      api.searchMovies(query.trim()).then(found => {
        if (!cancelled) { setResults(found); setError(false); }
      }).catch(() => {
        if (!cancelled) { setResults([]); setError(true); }
      }).finally(() => {
        if (!cancelled) { setLoading(false); if (!dismissed.current) setOpen(true); }
      });
    }, 250);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [query, selected, attempt, disabled]);

  function editQuery(value: string) {
    setQuery(value); setResults([]); setError(false);
    setLoading(value.trim().length >= 2); setOpen(value.trim().length >= 2);
    dismissed.current = false;
  }
  function close() { dismissed.current = true; setOpen(false); }
  function onKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === 'Escape') { event.preventDefault(); close(); inputRef.current?.focus(); return; }
    if (!open) return;
    if (event.key === 'Enter' && event.target === inputRef.current) {
      // Selection must be explicit; Enter in the search field never submits the ratings.
      event.preventDefault(); return;
    }
    if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return;
    const buttons = Array.from(containerRef.current?.querySelectorAll<HTMLButtonElement>('button[data-result]') ?? []);
    if (!buttons.length) return;
    event.preventDefault();
    const current = buttons.indexOf(document.activeElement as HTMLButtonElement);
    const next = event.key === 'ArrowDown' ? Math.min(current + 1, buttons.length - 1) : current - 1;
    if (next < 0) inputRef.current?.focus(); else buttons[next]?.focus();
  }

  if (selected) return <div className="movie-picker-selected"><span className="selected-title">{selected.title}</span>
    <button disabled={disabled} type="button" className="link-button" onClick={() => { onSelect(null); editQuery(''); requestAnimationFrame(() => inputRef.current?.focus()); }}>Change</button></div>;

  return <div className="movie-picker" ref={containerRef} onKeyDown={onKeyDown} onBlur={event => {
    if (!event.currentTarget.contains(event.relatedTarget as Node | null)) close();
  }}>
    <div className="search-bar"><svg aria-hidden="true" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="10.5" cy="10.5" r="6.5" /><path d="m16 16 5 5" /></svg>
      <label className="sr-only" htmlFor={id}>Search a movie title</label>
      <input id={id} ref={inputRef} type="search" autoComplete="off" placeholder="Search a movie title…" value={query} disabled={disabled}
        aria-controls={open ? `${id}-results` : undefined} aria-describedby={`${id}-status`}
        onChange={event => editQuery(event.target.value)} onFocus={() => { dismissed.current = false; if (query.trim().length >= 2) setOpen(true); }} />
    </div>
    <span className="sr-only" id={`${id}-status`} role="status">{loading ? 'Searching movies' : error ? 'Movie search failed' : `${results.length} results`}</span>
    {open && <div className="search-results" id={`${id}-results`}>
      {loading ? <p className="search-message">Searching…</p> : error ? <div className="search-message"><p role="alert">Could not search movies. Please try again.</p><button type="button" className="link-button" onClick={() => { setError(false); setLoading(true); setAttempt(n => n + 1); }}>Retry search</button></div>
        : results.length === 0 ? <p className="search-message">No movies found. Try a shorter title or different spelling.</p>
        : <ul>{results.map(movie => { const { name, year } = splitYear(movie.title); return <li key={movie.movie_id}>
          <button type="button" data-result className="search-result" onClick={() => { onSelect(movie); editQuery(''); close(); }}>
            <span className="search-result-title">{name}<small>{primaryGenres(movie.genres)}</small></span>
            <span className="search-result-meta">{year}<small>{formatRating(movie.rating_avg, movie.rating_count)}</small></span>
          </button></li>; })}</ul>}
    </div>}
  </div>;
}
