"""Service A: the FastAPI business service. No ML, no GPU — stateful via
PostgreSQL. Talks to Service B (real or mock, see service_b_client.py) once
per turn and persists whatever it decides.

Run: uvicorn src.app.service_a.main:app --reload --port 8000
"""

from __future__ import annotations

import os
from collections.abc import Iterable

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session as DbSession

from src.app.db import repository as repo
from src.app.db.database import get_db
from src.app.db.models import Session as SessionModel
from src.app.service_a.profile import build_profile_text
from src.app.service_a.schemas import (
    AskTurnResponse,
    ClarificationAnswerRequest,
    CreateSessionRequest,
    HistorySessionOut,
    RecommendationItemOut,
    RecommendTurnResponse,
    SessionOut,
)
from src.app.service_a.service_b_client import HttpServiceBClient, MockServiceBClient, ServiceBClient

app = FastAPI(title="Movie Recommendation Agent - Service A")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173")],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_service_b_client(db: DbSession = Depends(get_db)) -> ServiceBClient:
    if os.environ.get("SERVICE_B_MODE", "mock") == "http":
        return HttpServiceBClient()

    def candidate_source(exclude_ids: Iterable[int]):
        movies = repo.list_candidate_movies(db, exclude_ids, limit=10)
        return [{"movie_id": movie.movie_id, "title": movie.title} for movie in movies]

    return MockServiceBClient(candidate_source=candidate_source)


def _recommendation_items_out(db: DbSession, items) -> list[RecommendationItemOut]:
    movies = repo.get_movies_by_ids(db, (item.movie_id for item in items))
    return [
        RecommendationItemOut(
            movie_id=item.movie_id,
            title=movies[item.movie_id].title if item.movie_id in movies else None,
            rank=item.rank,
            score=item.score,
            explanation=item.explanation,
        )
        for item in items
    ]


def _advance_turn(db: DbSession, session: SessionModel, client: ServiceBClient):
    watched = repo.get_watched_items(db, session.id)
    history_movie_ids = [item.movie_id for item in watched]
    clarifications = [
        {"question": exchange.question, "answer": exchange.answer}
        for exchange in repo.get_clarifications(db, session.id)
        if exchange.answer is not None
    ]
    profile_text = build_profile_text(db, session.id)
    turn_index = len(repo.get_action_traces(db, session.id)) + 1

    decision = client.decide(
        profile_text=profile_text, history_movie_ids=history_movie_ids, clarifications=clarifications
    )
    action = decision.get("action")

    if action == "ask":
        question = decision["question"]
        exchange = repo.append_clarification(db, session.id, turn_index, question)
        repo.set_session_status(db, session.id, "awaiting_clarification")
        repo.append_action_trace(db, session.id, turn_index, "ask", {"question": question})
        return AskTurnResponse(session_id=session.id, clarification_id=exchange.id, question=question)

    if action == "recommend":
        recommendation = repo.save_recommendations(db, session.id, turn_index, decision["items"])
        repo.set_session_status(db, session.id, "completed")
        repo.append_action_trace(db, session.id, turn_index, "recommend", {"item_count": len(decision["items"])})
        return RecommendTurnResponse(
            session_id=session.id,
            recommendation_id=recommendation.id,
            items=_recommendation_items_out(db, recommendation.items),
        )

    raise HTTPException(status_code=502, detail=f"Service B returned an unknown action: {action!r}")


@app.post("/sessions", response_model=SessionOut, status_code=201)
def create_session(payload: CreateSessionRequest, db: DbSession = Depends(get_db)) -> SessionOut:
    user_id = payload.user_id
    if user_id is None:
        user_id = repo.create_user(db).id
    session = repo.create_session(db, user_id=user_id)
    if payload.ratings:
        repo.record_watched_items(db, session.id, [item.model_dump() for item in payload.ratings])
    return SessionOut(id=session.id, user_id=user_id, status=session.status, created_at=session.created_at)


@app.get("/sessions/{session_id}", response_model=SessionOut)
def read_session(session_id: int, db: DbSession = Depends(get_db)) -> SessionOut:
    session = repo.get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return SessionOut(id=session.id, user_id=session.user_id, status=session.status, created_at=session.created_at)


@app.post("/sessions/{session_id}/turn", response_model=AskTurnResponse | RecommendTurnResponse)
def advance_turn(
    session_id: int, db: DbSession = Depends(get_db), client: ServiceBClient = Depends(get_service_b_client)
):
    session = repo.get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    if session.status == "completed":
        raise HTTPException(status_code=409, detail="session already completed")
    if session.status == "awaiting_clarification":
        raise HTTPException(status_code=409, detail="answer the open clarification before requesting another turn")
    return _advance_turn(db, session, client)


@app.post("/sessions/{session_id}/clarifications", response_model=AskTurnResponse | RecommendTurnResponse)
def answer_clarification(
    session_id: int,
    payload: ClarificationAnswerRequest,
    db: DbSession = Depends(get_db),
    client: ServiceBClient = Depends(get_service_b_client),
):
    session = repo.get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    exchange = repo.get_open_clarification(db, session_id)
    if exchange is None:
        raise HTTPException(status_code=409, detail="no open clarification to answer")
    repo.answer_clarification(db, exchange.id, payload.answer)
    repo.set_session_status(db, session_id, "active")
    return _advance_turn(db, session, client)


@app.get("/sessions/{session_id}/recommendations", response_model=RecommendTurnResponse)
def read_recommendations(session_id: int, db: DbSession = Depends(get_db)) -> RecommendTurnResponse:
    recommendation = repo.get_latest_recommendation(db, session_id)
    if recommendation is None:
        raise HTTPException(status_code=404, detail="no recommendations yet")
    return RecommendTurnResponse(
        session_id=session_id,
        recommendation_id=recommendation.id,
        items=_recommendation_items_out(db, recommendation.items),
    )


@app.get("/users/{user_id}/history", response_model=list[HistorySessionOut])
def read_history(user_id: int, db: DbSession = Depends(get_db)) -> list[HistorySessionOut]:
    sessions = repo.get_user_history(db, user_id)
    history: list[HistorySessionOut] = []
    for session in sessions:
        latest = max(session.recommendations, key=lambda rec: rec.turn_index, default=None)
        items = _recommendation_items_out(db, latest.items) if latest is not None else None
        history.append(
            HistorySessionOut(
                session_id=session.id, status=session.status, created_at=session.created_at, latest_recommendation=items
            )
        )
    return history
