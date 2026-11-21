"""Isolated browser-test API: real Service A routes, SQLite, fixture movies.

Never connects to the developer's PostgreSQL database or downloads a model.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ['DATABASE_URL'] = 'sqlite://'
os.environ['FRONTEND_ORIGIN'] = 'http://127.0.0.1:4173'
os.environ['SERVICE_B_MODE'] = 'mock'

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool
from src.app.db.database import Base, get_db
from src.app.db.models import Movie
from src.app.service_a.main import app

engine = create_engine('sqlite://', connect_args={'check_same_thread': False}, poolclass=StaticPool)
Base.metadata.create_all(engine)
# Deliberately small fixture data for UI acceptance, not recommendation evaluation.
titles = ['Arrival (2016)', 'Contact (1997)', 'Interstellar (2014)', 'Moon (2009)',
          'The Truman Show (1998)', 'Her (2013)', 'The Matrix (1999)',
          'WALL-E (2008)', 'The Martian (2015)', 'Gattaca (1997)',
          'Blade Runner (1982)', 'Solaris (1972)', 'Toy Story (1995)']
with Session(engine) as db:
    for i, title in enumerate(titles, 1):
        db.add(Movie(movie_id=i, title=title, genres='Drama|Sci-Fi', text=title,
                     rating_avg=4.1, rating_count=1200))
    db.commit()

def fixture_db():
    with Session(engine) as db:
        yield db

app.dependency_overrides[get_db] = fixture_db

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8017)
