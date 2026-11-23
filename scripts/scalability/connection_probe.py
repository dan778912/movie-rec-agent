"""Compare checked-out connections during a real HTTP fixture call.

Uses the disposable benchmark database; does not run models. Run while no load
matrix is active. The old handler is loaded from the specified git commit.
"""
import json
import os
from pathlib import Path
import subprocess
import time
import httpx
import psycopg
from sqlalchemy.orm import Session
from src.app.db.database import engine
from src.app.db import repository as repo
from src.app.service_a.main import _advance_turn


def main():
    root=Path(__file__).resolve().parents[2]
    source=subprocess.check_output(['git','show','737f83d:src/app/service_a/main.py'],cwd=root,text=True)
    old={};exec(compile(source,'baseline-main.py','exec'),old)
    measurements=[]
    class Client:
        def decide(self, **kwargs):
            with psycopg.connect(os.environ['BENCH_DATABASE_URL'],autocommit=True) as observer:
                states=observer.execute("SELECT state,count(*) FROM pg_stat_activity WHERE datname=current_database() AND pid<>pg_backend_pid() GROUP BY state").fetchall()
            measurements.append({'variant':variant,'checked_out_during_call':engine.pool.checkedout(),
                                 'database_states':dict(states)})
            start=time.monotonic()
            response=httpx.post('http://127.0.0.1:8028/decide',json=kwargs,timeout=5)
            response.raise_for_status()
            measurements[-1]['http_fixture_seconds']=time.monotonic()-start
            return response.json()
    for variant, handler in [('baseline',old['_advance_turn']),('fixed',_advance_turn)]:
        with Session(engine) as db:
            session=repo.create_session(db)
            result=handler(db,session,Client())
            if result.action!='ask':raise AssertionError('Unexpected fixture response')
    assert measurements[0]['checked_out_during_call']==1
    assert measurements[1]['checked_out_during_call']==0
    print(json.dumps(measurements,indent=2))

if __name__=='__main__':main()
