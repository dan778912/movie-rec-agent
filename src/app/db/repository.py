"""Data-access layer for Service A. Routes call only these functions — never
raw SQL or the ORM session directly — so the persistence shape can change
without touching request handlers."""

from __future__ import annotations

import datetime
from collections.abc import Iterable, Mapping

from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import selectinload

from src.app.db.models import (
    AgentActionTrace,
    AppUser,
    ClarificationExchange,
    Movie,
    Recommendation,
    RecommendationItem,
    Session,
    WatchedItem,
)


def create_user(db: DbSession) -> AppUser:
    user = AppUser()
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def create_session(db: DbSession, user_id: int | None = None) -> Session:
    session = Session(user_id=user_id, status="active")
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_session(db: DbSession, session_id: int) -> Session | None:
    return db.get(Session, session_id)


def set_session_status(db: DbSession, session_id: int, status: str) -> Session:
    session = db.get(Session, session_id)
    if session is None:
        raise ValueError(f"no session {session_id}")
    session.status = status
    db.commit()
    db.refresh(session)
    return session


def record_watched_items(
    db: DbSession, session_id: int, items: Iterable[Mapping]
) -> list[WatchedItem]:
    rows = [
        WatchedItem(
            session_id=session_id,
            movie_id=item["movie_id"],
            title=item["title"],
            rating=item["rating"],
            rated_at=item.get("rated_at"),
        )
        for item in items
    ]
    db.add_all(rows)
    db.commit()
    for row in rows:
        db.refresh(row)
    return rows


def get_watched_items(db: DbSession, session_id: int) -> list[WatchedItem]:
    stmt = select(WatchedItem).where(WatchedItem.session_id == session_id).order_by(WatchedItem.id)
    return list(db.scalars(stmt))


def append_clarification(db: DbSession, session_id: int, turn_index: int, question: str) -> ClarificationExchange:
    exchange = ClarificationExchange(session_id=session_id, turn_index=turn_index, question=question)
    db.add(exchange)
    db.commit()
    db.refresh(exchange)
    return exchange


def answer_clarification(db: DbSession, exchange_id: int, answer: str) -> ClarificationExchange:
    exchange = db.get(ClarificationExchange, exchange_id)
    if exchange is None:
        raise ValueError(f"no clarification exchange {exchange_id}")
    exchange.answer = answer
    exchange.answered_at = datetime.datetime.now(datetime.timezone.utc)
    db.commit()
    db.refresh(exchange)
    return exchange


def get_open_clarification(db: DbSession, session_id: int) -> ClarificationExchange | None:
    stmt = (
        select(ClarificationExchange)
        .where(ClarificationExchange.session_id == session_id, ClarificationExchange.answer.is_(None))
        .order_by(ClarificationExchange.turn_index.desc())
        .limit(1)
    )
    return db.scalars(stmt).first()


def get_clarifications(db: DbSession, session_id: int) -> list[ClarificationExchange]:
    stmt = (
        select(ClarificationExchange)
        .where(ClarificationExchange.session_id == session_id)
        .order_by(ClarificationExchange.turn_index)
    )
    return list(db.scalars(stmt))


def save_recommendations(
    db: DbSession, session_id: int, turn_index: int, items: Iterable[Mapping]
) -> Recommendation:
    recommendation = Recommendation(session_id=session_id, turn_index=turn_index)
    recommendation.items = [
        RecommendationItem(
            movie_id=item["movie_id"],
            rank=item["rank"],
            score=item["score"],
            explanation=item.get("explanation"),
        )
        for item in items
    ]
    db.add(recommendation)
    db.commit()
    db.refresh(recommendation)
    return recommendation


def get_latest_recommendation(db: DbSession, session_id: int) -> Recommendation | None:
    stmt = (
        select(Recommendation)
        .where(Recommendation.session_id == session_id)
        .options(selectinload(Recommendation.items))
        .order_by(Recommendation.turn_index.desc())
        .limit(1)
    )
    return db.scalars(stmt).first()


def append_action_trace(
    db: DbSession, session_id: int, turn_index: int, action: str, payload: dict | None = None
) -> AgentActionTrace:
    trace = AgentActionTrace(session_id=session_id, turn_index=turn_index, action=action, payload=payload or {})
    db.add(trace)
    db.commit()
    db.refresh(trace)
    return trace


def get_action_traces(db: DbSession, session_id: int) -> list[AgentActionTrace]:
    stmt = (
        select(AgentActionTrace)
        .where(AgentActionTrace.session_id == session_id)
        .order_by(AgentActionTrace.turn_index)
    )
    return list(db.scalars(stmt))


def get_user_history(db: DbSession, user_id: int) -> list[Session]:
    stmt = (
        select(Session)
        .where(Session.user_id == user_id)
        .options(selectinload(Session.recommendations).selectinload(Recommendation.items))
        .order_by(Session.created_at.desc())
    )
    return list(db.scalars(stmt))


def get_movies_by_ids(db: DbSession, movie_ids: Iterable[int]) -> dict[int, Movie]:
    ids = list(set(movie_ids))
    if not ids:
        return {}
    stmt = select(Movie).where(Movie.movie_id.in_(ids))
    return {movie.movie_id: movie for movie in db.scalars(stmt)}


def list_candidate_movies(db: DbSession, exclude_ids: Iterable[int], limit: int) -> list[Movie]:
    """Placeholder candidate lookup for `MockServiceBClient` only — NOT the
    real retrieval stage. Member 3's online retrieval module replaces this
    once Service B exists; plain catalog order has no relevance ranking."""

    exclude = set(exclude_ids)
    stmt = select(Movie).order_by(Movie.movie_id).limit(limit + len(exclude))
    candidates = [movie for movie in db.scalars(stmt) if movie.movie_id not in exclude]
    return candidates[:limit]
