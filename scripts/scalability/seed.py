"""Seed only a dedicated local load-test database from frozen movie documents."""
import argparse
import json
import os
from urllib.parse import urlparse
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import Session
from src.app.db.database import Base
from src.app.db.models import Movie


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--documents',required=True)
    args=parser.parse_args()
    url=os.environ['BENCH_DATABASE_URL']
    parsed=urlparse(url)
    if parsed.hostname not in ('localhost','127.0.0.1') or parsed.path!='/movie_rec_bench':
        raise ValueError('Only the dedicated local movie_rec_bench database is allowed')
    engine=create_engine(url.replace('postgresql://','postgresql+psycopg://'))
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        if db.scalar(select(func.count()).select_from(Movie)):
            raise ValueError('Catalog is already populated; do not mix fixture versions')
        with open(args.documents,encoding='utf-8') as source:
            for line in source:
                row=json.loads(line)
                db.add(Movie(movie_id=row['movie_id'],title=row['title'],genres=row.get('genres',''),text=row['text']))
        db.commit()
        print(f'Seeded {db.scalar(select(func.count()).select_from(Movie))} catalog rows')
    engine.dispose()

if __name__=='__main__':main()
