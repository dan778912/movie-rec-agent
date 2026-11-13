import { useRef, useState } from 'react';
import type { MovieSearchResult, RatingInput } from '../api';
import { MoviePicker } from './MoviePicker';

interface Props { onSubmit: (ratings: RatingInput[]) => void; submitting: boolean }
interface Row { id: number; movie: MovieSearchResult | null; rating: string }
const STAR_VALUES = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5];

export function RatingForm({ onSubmit, submitting }: Props) {
  const nextId = useRef(1);
  const [rows, setRows] = useState<Row[]>([{ id: 0, movie: null, rating: '5' }]);
  const updateRow = (id: number, patch: Partial<Row>) => setRows(current => current.map(row => row.id === id ? { ...row, ...patch } : row));
  const selected = rows.filter(row => row.movie);
  const duplicates = new Set(selected.map(row => row.movie!.movie_id)).size !== selected.length;
  const canSubmit = selected.length > 0 && !duplicates && !submitting;

  return (
    <form className="rating-form" onSubmit={event => {
      event.preventDefault();
      if (canSubmit) onSubmit(selected.map(row => ({ movie_id: row.movie!.movie_id, title: row.movie!.title, rating: Number(row.rating) })));
    }}>
      <div className="page-intro"><p className="eyebrow">Your next watch</p><h1>Start with what<br />you’ve watched.</h1><p className="intro-copy">Rate a movie to find something new.</p></div>
      <fieldset disabled={submitting} className="rating-fields">
        <legend className="sr-only">Movies you have watched</legend>
        <div className="list-heading" aria-hidden="true"><span>Movie</span><span>Your rating</span></div>
        {rows.map((row, index) => <div className="rating-row" key={row.id}>
          <span className="row-number" aria-hidden="true">{String(index + 1).padStart(2, '0')}</span>
          <MoviePicker selected={row.movie} onSelect={movie => updateRow(row.id, { movie })} disabled={submitting} />
          <select aria-label={`Your rating for ${row.movie?.title ?? `movie ${index + 1}`}`} value={row.rating} onChange={e => updateRow(row.id, { rating: e.target.value })}>
            {STAR_VALUES.map(value => <option key={value} value={value}>{value} ★</option>)}
          </select>
          <button type="button" className="icon-button" disabled={rows.length === 1} aria-label={`Remove ${row.movie?.title ?? `movie ${index + 1}`}`} onClick={() => setRows(current => current.filter(r => r.id !== row.id))}>×</button>
        </div>)}
        <button type="button" className="add-movie" onClick={() => { const id = nextId.current++; setRows(current => [...current, { id, movie: null, rating: '5' }]); }}>+ Add another movie</button>
      </fieldset>
      {duplicates && <p className="error-message" role="alert">This movie is already on your list. Keep one rating per movie.</p>}
      <div className="form-bottom"><span className="muted">{selected.length ? `${selected.length} ${selected.length === 1 ? 'movie' : 'movies'} selected` : 'Choose a movie to begin'}</span>
        <button className="primary" disabled={!canSubmit} type="submit">{submitting ? 'Starting…' : 'Get recommendations'}<span aria-hidden="true">↗</span></button>
      </div>
    </form>
  );
}
