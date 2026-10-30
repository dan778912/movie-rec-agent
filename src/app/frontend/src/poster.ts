// No poster images exist anywhere in this project — MovieLens doesn't ship
// them, and nothing here calls an external poster API (TMDB etc., by
// deliberate choice: no extra API key / external dependency / per-search
// network call). Every "poster" in the UI is a deterministic color block
// generated from the title, so the same movie always gets the same color
// and every card looks intentional even with zero real artwork.
const PALETTE = ["#a3321f", "#2f4858", "#6b7a3f", "#8a5a2b", "#4a3b5c", "#2e6b5e", "#7a2e3f", "#5c5230"];

export function posterColor(title: string): string {
  let hash = 0;
  for (let i = 0; i < title.length; i += 1) {
    hash = (hash * 31 + title.charCodeAt(i)) >>> 0;
  }
  return PALETTE[hash % PALETTE.length];
}
