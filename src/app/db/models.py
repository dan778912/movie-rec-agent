"""SQLAlchemy models for Service A's PostgreSQL store.

Schema is provisional: `AgentActionTrace.action` / `.payload` follow no team-agreed
action contract yet (Member 4 owns that). Kept loose (plain string + JSON blob) so
it can absorb the real contract later without a migration that drops data.
"""

from __future__ import annotations

import datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.app.db.database import Base


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


class Movie(Base):
    """Read-only reference row seeded from Member 1's frozen catalog."""

    __tablename__ = "movies"

    movie_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    genres: Mapped[str] = mapped_column(String, nullable=False, default="")
    text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # Kept so live profile text can reuse src/profiles/text.py's user_profile()
    # formatter exactly as Member 1's offline pipeline does (src/data/build.py).
    genome_tags: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    user_tags: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    # Real aggregates from Member 1's frozen train_ratings.csv (training split
    # only, not validation/test — see seed_catalog.py). Null until seeded.
    rating_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    rating_count: Mapped[int | None] = mapped_column(Integer, nullable=True)


class AppUser(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    sessions: Mapped[list["Session"]] = relationship(back_populates="user")


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    # One of: "active" (waiting on the next agent turn), "awaiting_clarification"
    # (a question was asked, waiting on the user's answer), "completed" (final
    # recommendations delivered). Provisional until Member 4's action contract lands.
    status: Mapped[str] = mapped_column(String, nullable=False, default="active")
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    user: Mapped[AppUser | None] = relationship(back_populates="sessions")
    watched_items: Mapped[list["WatchedItem"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="WatchedItem.id"
    )
    clarifications: Mapped[list["ClarificationExchange"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="ClarificationExchange.turn_index"
    )
    recommendations: Mapped[list["Recommendation"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="Recommendation.turn_index"
    )
    action_traces: Mapped[list["AgentActionTrace"]] = relationship(
        back_populates="session", cascade="all, delete-orphan", order_by="AgentActionTrace.turn_index"
    )


class WatchedItem(Base):
    """A rated/imported movie for a session. `movie_id` is not FK-constrained so
    imports can reference movies outside the current catalog snapshot; `title` is
    stored redundantly so the row stays self-contained either way."""

    __tablename__ = "watched_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), nullable=False)
    movie_id: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    rating: Mapped[float] = mapped_column(Float, nullable=False)
    rated_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    session: Mapped[Session] = relationship(back_populates="watched_items")


class ClarificationExchange(Base):
    __tablename__ = "clarification_exchanges"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), nullable=False)
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    asked_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    answered_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    session: Mapped[Session] = relationship(back_populates="clarifications")


class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), nullable=False)
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    session: Mapped[Session] = relationship(back_populates="recommendations")
    items: Mapped[list["RecommendationItem"]] = relationship(
        back_populates="recommendation", cascade="all, delete-orphan", order_by="RecommendationItem.rank"
    )


class RecommendationItem(Base):
    """Mirrors the roadmap's standard prediction record (movie_id, score, rank)
    plus the optional `explanation` field it allows for LLM methods."""

    __tablename__ = "recommendation_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    recommendation_id: Mapped[int] = mapped_column(ForeignKey("recommendations.id"), nullable=False)
    movie_id: Mapped[int] = mapped_column(Integer, nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)

    recommendation: Mapped[Recommendation] = relationship(back_populates="items")


class AgentActionTrace(Base):
    """Append-only log of what the agent decided each turn. `action` / `payload`
    are provisional until Member 4's action JSON contract is agreed."""

    __tablename__ = "agent_action_traces"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("sessions.id"), nullable=False)
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False)
    action: Mapped[str] = mapped_column(String, nullable=False)
    payload: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    session: Mapped[Session] = relationship(back_populates="action_traces")
