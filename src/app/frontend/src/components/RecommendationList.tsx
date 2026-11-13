import { useId, useState } from 'react';
import type { RecommendationItemOut } from '../api';
import { formatRating, primaryGenres, splitYear } from '../movieTitle';

export function RecommendationList({ items }: { items: RecommendationItemOut[] }) {
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const id = useId();
  if (!items.length) return <p className="empty-state">No recommendations were returned. Try a different set of ratings.</p>;
  return <ol className="recommendation-list">{items.map(item => {
    const { name, year } = splitYear(item.title ?? `Movie ${item.movie_id}`);
    const expanded = expandedId === item.movie_id;
    return <li className="recommendation" key={item.movie_id}>
      <button type="button" className="recommendation-row" onClick={() => setExpandedId(expanded ? null : item.movie_id)} aria-expanded={expanded} aria-controls={`${id}-${item.movie_id}`}>
        <span className="rank">{String(item.rank).padStart(2, '0')}</span>
        <span className="film-title">{name}<span className="film-meta">{year && <span>{year}</span>}<span>{primaryGenres(item.genres)}</span></span></span>
        <span className="catalog-rating" title="MovieLens average from training ratings">{formatRating(item.rating_avg, item.rating_count)}</span>
        <span className="expand-icon" aria-hidden="true">{expanded ? '−' : '+'}</span>
      </button>
      <div id={`${id}-${item.movie_id}`} hidden={!expanded} className="recommendation-detail"><p>{item.explanation ?? 'No explanation is available for this movie.'}</p></div>
    </li>;
  })}</ol>;
}
