// MovieLens titles are formatted "Name (YYYY)" (a handful of entries have no
// year). Splitting it out gives a dedicated year column, A24's /television
// table style, without storing year separately anywhere in the backend.
const YEAR_PATTERN = /\s*\((\d{4})\)\s*$/;

export function splitYear(title: string): { name: string; year: string | null } {
  const match = title.match(YEAR_PATTERN);
  if (!match) return { name: title, year: null };
  return { name: title.slice(0, match.index).trim(), year: match[1] };
}

export function primaryGenres(genres: string | null, max = 2): string {
  if (!genres) return "—";
  return genres.split("|").slice(0, max).join(", ");
}
