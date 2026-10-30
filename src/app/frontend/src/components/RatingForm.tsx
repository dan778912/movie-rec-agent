import { useState } from "react";
import type { RatingInput } from "../api";

interface Props {
  onSubmit: (ratings: RatingInput[]) => void;
  submitting: boolean;
}

interface Row {
  movieId: string;
  title: string;
  rating: string;
}

const emptyRow = (): Row => ({ movieId: "", title: "", rating: "5" });

export function RatingForm({ onSubmit, submitting }: Props) {
  const [rows, setRows] = useState<Row[]>([emptyRow()]);

  const updateRow = (index: number, patch: Partial<Row>) => {
    setRows((current) => current.map((row, i) => (i === index ? { ...row, ...patch } : row)));
  };

  const addRow = () => setRows((current) => [...current, emptyRow()]);
  const removeRow = (index: number) => setRows((current) => current.filter((_, i) => i !== index));

  const validRows = rows.filter((row) => row.movieId.trim() && row.title.trim() && row.rating.trim());
  const canSubmit = validRows.length > 0 && !submitting;

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    if (!canSubmit) return;
    onSubmit(
      validRows.map((row) => ({
        movie_id: Number(row.movieId),
        title: row.title.trim(),
        rating: Number(row.rating),
      })),
    );
  };

  return (
    <form className="rating-form" onSubmit={handleSubmit}>
      <h2>Rate a few movies to get started</h2>
      <p className="hint">Movie ID is the MovieLens catalog ID. Rating is 0.5-5 stars.</p>
      {rows.map((row, index) => (
        <div className="rating-row" key={index}>
          <input
            type="number"
            placeholder="Movie ID"
            value={row.movieId}
            onChange={(e) => updateRow(index, { movieId: e.target.value })}
          />
          <input
            type="text"
            placeholder="Title"
            value={row.title}
            onChange={(e) => updateRow(index, { title: e.target.value })}
          />
          <input
            type="number"
            step="0.5"
            min="0.5"
            max="5"
            placeholder="Rating"
            value={row.rating}
            onChange={(e) => updateRow(index, { rating: e.target.value })}
          />
          <button type="button" onClick={() => removeRow(index)} disabled={rows.length === 1} aria-label="Remove row">
            &times;
          </button>
        </div>
      ))}
      <div className="rating-form-actions">
        <button type="button" onClick={addRow}>
          + Add movie
        </button>
        <button type="submit" disabled={!canSubmit}>
          {submitting ? "Starting..." : "Start"}
        </button>
      </div>
    </form>
  );
}
