"""Service A API tests, driven through FastAPI's TestClient against a
throwaway in-memory SQLite database and the default MockServiceBClient
(SERVICE_B_MODE is unset/mock). Run: python -m unittest discover -s tests"""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.app.db.database import get_db
from src.app.db.models import Base, Movie
from src.app.service_a.main import app


def _ratings(movie_ids: list[int]) -> list[dict]:
    return [{"movie_id": movie_id, "title": f"Movie {movie_id}", "rating": 5.0} for movie_id in movie_ids]


class ServiceATests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
        Base.metadata.create_all(self.engine)
        session_factory = sessionmaker(bind=self.engine, future=True)
        self.session_factory = session_factory

        def override_get_db():
            db = session_factory()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db
        self.client = TestClient(app)

        seed_db = session_factory()
        for movie_id in range(1, 15):
            seed_db.add(
                Movie(movie_id=movie_id, title=f"Movie {movie_id}", genres="Drama", text=f"Movie {movie_id}.")
            )
        seed_db.commit()
        seed_db.close()

    def tearDown(self) -> None:
        app.dependency_overrides.clear()
        self.engine.dispose()

    def test_full_lifecycle_with_clarification(self) -> None:
        created = self.client.post("/sessions", json={"ratings": _ratings([1])})
        self.assertEqual(created.status_code, 201)
        session = created.json()
        self.assertEqual(session["status"], "active")

        turn = self.client.post(f"/sessions/{session['id']}/turn")
        self.assertEqual(turn.status_code, 200)
        self.assertEqual(turn.json()["action"], "ask")

        answered = self.client.post(f"/sessions/{session['id']}/clarifications", json={"answer": "Something fun"})
        self.assertEqual(answered.status_code, 200)
        body = answered.json()
        self.assertEqual(body["action"], "recommend")
        self.assertGreater(len(body["items"]), 0)
        self.assertNotIn(1, [item["movie_id"] for item in body["items"]])

        recs = self.client.get(f"/sessions/{session['id']}/recommendations")
        self.assertEqual(recs.status_code, 200)
        self.assertEqual(recs.json()["recommendation_id"], body["recommendation_id"])

        history = self.client.get(f"/users/{session['user_id']}/history")
        self.assertEqual(history.status_code, 200)
        self.assertEqual(len(history.json()), 1)
        self.assertEqual(history.json()[0]["status"], "completed")

    def test_sufficient_history_skips_clarification(self) -> None:
        created = self.client.post("/sessions", json={"ratings": _ratings([1, 2, 3, 4])})
        session = created.json()
        turn = self.client.post(f"/sessions/{session['id']}/turn")
        self.assertEqual(turn.json()["action"], "recommend")

    def test_turn_on_completed_session_conflicts(self) -> None:
        created = self.client.post("/sessions", json={"ratings": _ratings([1, 2, 3, 4])})
        session = created.json()
        self.client.post(f"/sessions/{session['id']}/turn")
        second = self.client.post(f"/sessions/{session['id']}/turn")
        self.assertEqual(second.status_code, 409)

    def test_turn_on_missing_session_is_404(self) -> None:
        resp = self.client.post("/sessions/99999/turn")
        self.assertEqual(resp.status_code, 404)

    def test_clarification_without_open_question_conflicts(self) -> None:
        created = self.client.post("/sessions", json={"ratings": _ratings([1, 2, 3, 4])})
        session = created.json()
        self.client.post(f"/sessions/{session['id']}/turn")  # sufficient history -> recommend directly
        answered = self.client.post(f"/sessions/{session['id']}/clarifications", json={"answer": "whatever"})
        self.assertEqual(answered.status_code, 409)


if __name__ == "__main__":
    unittest.main()
