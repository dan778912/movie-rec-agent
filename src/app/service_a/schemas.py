"""Pydantic request/response models for Service A's REST API."""

from __future__ import annotations

import datetime

from pydantic import BaseModel, Field


class MovieSearchResult(BaseModel):
    movie_id: int
    title: str
    genres: str
    rating_avg: float | None
    rating_count: int | None


class WatchedItemIn(BaseModel):
    movie_id: int
    title: str
    rating: float = Field(ge=0.5, le=5.0)
    rated_at: datetime.datetime | None = None


class CreateSessionRequest(BaseModel):
    user_id: int | None = None
    ratings: list[WatchedItemIn] = Field(default_factory=list)


class SessionOut(BaseModel):
    id: int
    user_id: int
    status: str
    created_at: datetime.datetime


class ClarificationAnswerRequest(BaseModel):
    answer: str = Field(min_length=1)


class RecommendationItemOut(BaseModel):
    movie_id: int
    title: str | None
    genres: str | None
    rating_avg: float | None
    rating_count: int | None
    rank: int
    score: float
    explanation: str | None


class AskTurnResponse(BaseModel):
    action: str = "ask"
    session_id: int
    clarification_id: int
    question: str


class RecommendTurnResponse(BaseModel):
    action: str = "recommend"
    session_id: int
    recommendation_id: int
    items: list[RecommendationItemOut]


class HistorySessionOut(BaseModel):
    session_id: int
    status: str
    created_at: datetime.datetime
    latest_recommendation: list[RecommendationItemOut] | None
