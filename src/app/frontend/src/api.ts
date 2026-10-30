// Typed client for Service A's REST API. See src/app/service_a/schemas.py
// for the authoritative shapes this mirrors.

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export interface RatingInput {
  movie_id: number;
  title: string;
  rating: number;
}

export interface MovieSearchResult {
  movie_id: number;
  title: string;
  genres: string;
}

export interface SessionOut {
  id: number;
  user_id: number;
  status: "active" | "awaiting_clarification" | "completed";
  created_at: string;
}

export interface RecommendationItemOut {
  movie_id: number;
  title: string | null;
  genres: string | null;
  rank: number;
  score: number;
  explanation: string | null;
}

export interface AskTurnResponse {
  action: "ask";
  session_id: number;
  clarification_id: number;
  question: string;
}

export interface RecommendTurnResponse {
  action: "recommend";
  session_id: number;
  recommendation_id: number;
  items: RecommendationItemOut[];
}

export type TurnResponse = AskTurnResponse | RecommendTurnResponse;

export interface HistorySessionOut {
  session_id: number;
  status: string;
  created_at: string;
  latest_recommendation: RecommendationItemOut[] | null;
}

class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new ApiError(response.status, body.detail ?? `Request to ${path} failed with ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  searchMovies(query: string): Promise<MovieSearchResult[]> {
    if (!query.trim()) return Promise.resolve([]);
    return request<MovieSearchResult[]>(`/movies/search?q=${encodeURIComponent(query)}`);
  },
  createSession(ratings: RatingInput[], userId?: number): Promise<SessionOut> {
    return request<SessionOut>("/sessions", {
      method: "POST",
      body: JSON.stringify({ ratings, user_id: userId ?? null }),
    });
  },
  advanceTurn(sessionId: number): Promise<TurnResponse> {
    return request<TurnResponse>(`/sessions/${sessionId}/turn`, { method: "POST" });
  },
  answerClarification(sessionId: number, answer: string): Promise<TurnResponse> {
    return request<TurnResponse>(`/sessions/${sessionId}/clarifications`, {
      method: "POST",
      body: JSON.stringify({ answer }),
    });
  },
  getRecommendations(sessionId: number): Promise<RecommendTurnResponse> {
    return request<RecommendTurnResponse>(`/sessions/${sessionId}/recommendations`);
  },
  getHistory(userId: number): Promise<HistorySessionOut[]> {
    return request<HistorySessionOut[]>(`/users/${userId}/history`);
  },
};

export { ApiError };
