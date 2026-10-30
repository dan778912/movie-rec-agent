import { useState } from "react";
import type { MovieSearchResult, RatingInput } from "../api";
import { MoviePicker } from "./MoviePicker";

interface Props {
  onSubmit: (ratings: RatingInput[]) => void;
  submitting: boolean;
}

interface Row {
  id: number;
  movie: MovieSearchResult | null;
  rating: string;
}

let nextRowId = 0;
const emptyRow = (): Row => ({ id: nextRowId++, movie: null, rating: "5" });

const STAR_VALUES = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5];

export function RatingForm({ onSubmit, submitting }: Props) {
  const [rows, setRows] = useState<Row[]>([emptyRow()]);

  const updateRow = (id: number, patch: Partial<Row>) => {
    setRows((current) => current.map((row) => (row.id === id ? { ...row, ...patch } : row)));
  };

  const addRow = () => setRows((current) => [...current, emptyRow()]);
  const removeRow = (id: number) => setRows((current) => current.filter((row) => row.id !== id));

  const validRows = rows.filter((row) => row.movie && row.rating.trim());
  const canSubmit = validRows.length > 0 && !submitting;

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    if (!canSubmit) return;
    onSubmit(
      validRows.map((row) => ({
        movie_id: row.movie!.movie_id,
        title: row.movie!.title,
        rating: Number(row.rating),
      })),
    );
  };

  return (
    <form className="rating-form" onSubmit={handleSubmit}>
      <h2>What have you watched?</h2>
      <p className="hint">Search a title and pick it from the list — rate as many as you like.</p>
      {rows.map((row) => (
        <div className="rating-row" key={row.id}>
          <MoviePicker selected={row.movie} onSelect={(movie) => updateRow(row.id, { movie })} />
          <select value={row.rating} onChange={(e) => updateRow(row.id, { rating: e.target.value })}>
            {STAR_VALUES.map((value) => (
              <option key={value} value={value}>
                {value} ★
              </option>
            ))}
          </select>
          <button
            type="button"
            className="icon-button"
            onClick={() => removeRow(row.id)}
            disabled={rows.length === 1}
            aria-label="Remove row"
          >
            &times;
          </button>
        </div>
      ))}
      <div className="rating-form-actions">
        <button type="button" className="secondary" onClick={addRow}>
          + Add another
        </button>
        <button type="submit" className="primary" disabled={!canSubmit}>
          {submitting ? "Starting..." : "Start"}
        </button>
      </div>
    </form>
  );
}
