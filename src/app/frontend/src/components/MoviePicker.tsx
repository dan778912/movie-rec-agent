import { useEffect, useRef, useState } from "react";
import { api, type MovieSearchResult } from "../api";

interface Props {
  selected: MovieSearchResult | null;
  onSelect: (movie: MovieSearchResult | null) => void;
}

// Search-as-you-type title picker so nobody has to know a MovieLens catalog
// ID. Matching is plain substring search (server-side, see
// src/app/db/repository.py search_movies) — it tolerates partial titles and
// wrong word order to some extent, but not real misspellings. True
// typo-tolerant search is a documented gap, not solved here.
export function MoviePicker({ selected, onSelect }: Props) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<MovieSearchResult[]>([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (query.trim().length < 2) {
      setResults([]);
      setOpen(false);
      return;
    }
    setLoading(true);
    const handle = setTimeout(() => {
      api
        .searchMovies(query)
        .then((found) => {
          setResults(found);
          setOpen(true);
        })
        .finally(() => setLoading(false));
    }, 250);
    return () => clearTimeout(handle);
  }, [query]);

  useEffect(() => {
    const onClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onClickOutside);
    return () => document.removeEventListener("mousedown", onClickOutside);
  }, []);

  if (selected) {
    return (
      <div className="movie-picker movie-picker-selected">
        <span className="movie-picker-title">{selected.title}</span>
        <button type="button" className="link-button" onClick={() => onSelect(null)}>
          change
        </button>
      </div>
    );
  }

  return (
    <div className="movie-picker" ref={containerRef}>
      <div className="search-bar">
        <span className="search-icon" aria-hidden="true">
          &#8981;
        </span>
        <input
          type="text"
          placeholder="search a title..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => results.length > 0 && setOpen(true)}
        />
      </div>
      {open && (
        <div className="notes-list-panel">
          {loading && <div className="movie-picker-hint">searching...</div>}
          {!loading && results.length === 0 && (
            <div className="movie-picker-hint">no matches — try fewer words or a different spelling.</div>
          )}
          {!loading && results.length > 0 && (
            <div className="notes-list">
              {results.map((movie) => (
                <div className="notes-entry" key={movie.movie_id}>
                  <button
                    type="button"
                    className="notes-row"
                    onClick={() => {
                      onSelect(movie);
                      setQuery("");
                      setOpen(false);
                    }}
                  >
                    <span className="notes-row-title">{movie.title}</span>
                    <span className="notes-tag">{movie.genres.split("|")[0] || "—"}</span>
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
