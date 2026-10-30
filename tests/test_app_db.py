"""Repository-layer tests against a throwaway in-memory SQLite database, so
these run without Docker/Postgres. Run: python -m unittest discover -s tests"""

from __future__ import annotations

import time
import unittest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.app.db import repository as repo
from src.app.db.models import Base, Movie


class RepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = create_engine(
            "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
        )
        Base.metadata.create_all(self.engine)
        SessionFactory = sessionmaker(bind=self.engine, future=True)
        self.db = SessionFactory()

    def tearDown(self) -> None:
        self.db.close()
        self.engine.dispose()

    def test_create_session_defaults_to_active(self) -> None:
        user = repo.create_user(self.db)
        session = repo.create_session(self.db, user_id=user.id)
        self.assertEqual(session.status, "active")
        self.assertEqual(session.user_id, user.id)

    def test_record_and_fetch_watched_items_preserve_order(self) -> None:
        session = repo.create_session(self.db)
        repo.record_watched_items(
            self.db,
            session.id,
            [
                {"movie_id": 1, "title": "Toy Story", "rating": 5.0},
                {"movie_id": 2, "title": "Heat", "rating": 2.0},
            ],
        )
        items = repo.get_watched_items(self.db, session.id)
        self.assertEqual([item.movie_id for item in items], [1, 2])

    def test_clarification_round_trip_and_open_lookup(self) -> None:
        session = repo.create_session(self.db)
        exchange = repo.append_clarification(self.db, session.id, turn_index=1, question="What mood?")
        self.assertIsNone(exchange.answer)
        self.assertEqual(repo.get_open_clarification(self.db, session.id).id, exchange.id)

        repo.answer_clarification(self.db, exchange.id, "Light and fun")
        self.assertIsNone(repo.get_open_clarification(self.db, session.id))
        self.assertEqual(repo.get_clarifications(self.db, session.id)[0].answer, "Light and fun")

    def test_save_recommendations_and_latest_lookup(self) -> None:
        session = repo.create_session(self.db)
        repo.save_recommendations(
            self.db, session.id, turn_index=1, items=[{"movie_id": 1, "rank": 1, "score": 0.9, "explanation": "x"}]
        )
        second = repo.save_recommendations(
            self.db, session.id, turn_index=2, items=[{"movie_id": 2, "rank": 1, "score": 0.8, "explanation": "y"}]
        )
        latest = repo.get_latest_recommendation(self.db, session.id)
        self.assertEqual(latest.id, second.id)
        self.assertEqual(latest.items[0].movie_id, 2)

    def test_action_trace_ordering(self) -> None:
        session = repo.create_session(self.db)
        repo.append_action_trace(self.db, session.id, turn_index=1, action="ask", payload={"q": "a"})
        repo.append_action_trace(self.db, session.id, turn_index=2, action="recommend", payload={"n": 5})
        traces = repo.get_action_traces(self.db, session.id)
        self.assertEqual([trace.action for trace in traces], ["ask", "recommend"])

    def test_user_history_orders_newest_first(self) -> None:
        user = repo.create_user(self.db)
        first = repo.create_session(self.db, user_id=user.id)
        time.sleep(0.01)
        second = repo.create_session(self.db, user_id=user.id)
        history = repo.get_user_history(self.db, user.id)
        self.assertEqual([session.id for session in history], [second.id, first.id])

    def test_get_movies_by_ids_returns_only_known_ids(self) -> None:
        self.db.add(Movie(movie_id=1, title="Toy Story", genres="Animation", text="."))
        self.db.commit()
        movies = repo.get_movies_by_ids(self.db, [1, 2])
        self.assertEqual(movies[1].title, "Toy Story")
        self.assertNotIn(2, movies)

    def test_list_candidate_movies_excludes_given_ids(self) -> None:
        for movie_id in range(1, 6):
            self.db.add(Movie(movie_id=movie_id, title=f"Movie {movie_id}", genres="", text=""))
        self.db.commit()
        candidates = repo.list_candidate_movies(self.db, exclude_ids=[1, 2], limit=2)
        ids = [movie.movie_id for movie in candidates]
        self.assertEqual(ids, [3, 4])


if __name__ == "__main__":
    unittest.main()
