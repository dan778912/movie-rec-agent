"""Run isolated Locust stages and save database telemetry and recovery probes.

Starts/stops only its own Service A process. Requires an existing disposable
PostgreSQL database, canonical seeded catalog, Locust executable, and app Python.
"""
import argparse
import csv
import hashlib
import datetime
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import threading
import time
import urllib.request
from urllib.parse import urlparse
import psycopg


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--locust',required=True)
    p.add_argument('--app-python',required=True)
    p.add_argument('--workload',choices=['MixedUser','SearchUser','SessionUser'],default='MixedUser')
    p.add_argument('--levels',default='20,50,100,150,300')
    p.add_argument('--duration',type=int,default=60)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--limit',type=int,default=0)
    args=p.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[2]
    dburl=os.environ['BENCH_DATABASE_URL']
    parsed=urlparse(dburl)
    if parsed.hostname not in ('localhost','127.0.0.1') or parsed.path != '/movie_rec_bench':
        raise ValueError('Refusing database without the dedicated movie_rec_bench name')
    with psycopg.connect(dburl,autocommit=True) as reset:
        reset.execute('TRUNCATE TABLE users, sessions RESTART IDENTITY CASCADE')
    metadata={'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
              'dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True)),
              'source_sha256':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (root/'src/app').rglob('*.py')},
              'driver_sha256':hashlib.sha256((root/'tests/load/locustfile.py').read_bytes()).hexdigest(),
              'platform':platform.platform(),'python':platform.python_version(),'workers':args.workers,'admission_limit_per_worker':args.limit,
              'workload':args.workload,'levels':args.levels,'duration_seconds_per_stage_including_ramp':args.duration,
              'spawn_rate_per_second':100,'think_time_seconds':[0.5,1.5],'pool_size_per_worker':5,'max_overflow_per_worker':10,
              'service_b':'in_process_mock','catalog_rows':21333,'request_timeout_seconds':5,'locust_version':subprocess.check_output([args.locust,'--version'],text=True).strip()}
    (output/'environment.json').write_text(json.dumps(metadata,indent=2)+'\n')
    stop=threading.Event();stage={'users':0}
    def sample():
        with psycopg.connect(dburl,autocommit=True,application_name='scalability-observer') as db, (output/'pg-stat-activity.jsonl').open('w') as f:
            while not stop.is_set():
                records=db.execute("SELECT state, wait_event_type, count(*), max(extract(epoch FROM (clock_timestamp()-xact_start))) FROM pg_stat_activity WHERE datname=current_database() AND pid<>pg_backend_pid() AND backend_type='client backend' GROUP BY state,wait_event_type").fetchall()
                f.write(json.dumps({'time':time.time(),'users':stage['users'],'states':[{'state':s,'wait':w,'count':n,'max_transaction_seconds':float(age) if age is not None else None} for s,w,n,age in records]})+'\n');f.flush()
                stop.wait(.5)
    env={**os.environ,'DATABASE_URL':dburl.replace('postgresql://','postgresql+psycopg://'),
         'SERVICE_B_MODE':'mock','SERVICE_A_MAX_IN_FLIGHT':str(args.limit),'PYTHONUNBUFFERED':'1'}
    log=(output/'service-a.log').open('w')
    server=subprocess.Popen([args.app_python,'-m','uvicorn','src.app.service_a.main:app','--host','127.0.0.1','--port','8027','--workers',str(args.workers),'--no-access-log'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    watcher=None
    def probe():
        start=time.monotonic()
        try:
            with urllib.request.urlopen('http://127.0.0.1:8027/movies/search?q=Alien',timeout=5) as r:
                return {'status':r.status,'seconds':time.monotonic()-start}
        except Exception as e:return {'error':str(e),'seconds':time.monotonic()-start}
    try:
        for _ in range(30):
            if server.poll() is not None:raise RuntimeError('Service A failed to start; inspect log')
            if probe().get('status')==200:break
            time.sleep(.2)
        else:raise RuntimeError('Service A never became ready')
        watcher=threading.Thread(target=sample,daemon=True);watcher.start()
        results=[]
        for users in map(int,args.levels.split(',')):
            stage['users']=users;prefix=output/f'{users}-users'
            command=[args.locust,'-f',str(root/'tests/load/locustfile.py'),args.workload,'--headless','--host','http://127.0.0.1:8027','--users',str(users),'--spawn-rate','100','--run-time',f'{args.duration}s','--stop-timeout','6','--csv',str(prefix),'--csv-full-history','--html',str(prefix)+'.html','--exit-code-on-error','0','--only-summary']
            with (output/f'{users}-locust.log').open('w') as f:
                run=subprocess.run(command,cwd=root,stdout=f,stderr=subprocess.STDOUT,timeout=args.duration+30)
            recovery=probe()
            results.append({'users':users,'locust_exit':run.returncode,'recovery_probe':recovery})
            (output/'stages.json').write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps(results[-1]),flush=True)
            if recovery.get('status')!=200:
                print('Stopping escalation: recovery probe failed.',flush=True);break
            time.sleep(2)
    finally:
        stop.set()
        if watcher:watcher.join(timeout=3)
        os.killpg(server.pid,signal.SIGTERM)
        try:server.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(server.pid,signal.SIGKILL);server.wait()
        log.close()

if __name__=='__main__':main()
