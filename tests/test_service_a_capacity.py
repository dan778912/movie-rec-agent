"""Regression checks for DB connection lifetime and bounded admission."""
import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import QueuePool
from src.app.db.database import Base
from src.app.db import repository as repo
from src.app.service_a.main import _advance_turn

class ConnectionLifetimeTests(unittest.TestCase):
    def test_no_connection_is_held_during_downstream_call(self):
        engine = create_engine('sqlite://', poolclass=QueuePool, pool_size=1, max_overflow=0)
        Base.metadata.create_all(engine)
        observed = []
        class Client:
            def decide(self, **kwargs):
                observed.append(engine.pool.checkedout())
                return {'action':'ask','question':'Which genre?'}
        try:
            with Session(engine) as db:
                session = repo.create_session(db)
                sid = session.id
                result = _advance_turn(db, session, Client())
                self.assertEqual(result.session_id, sid)
                self.assertEqual(repo.get_session(db,sid).status,'awaiting_clarification')
            self.assertEqual(observed,[0], 'Read transactions must end before waiting for Service B')
            self.assertEqual(engine.pool.checkedout(),0)
        finally:
            engine.dispose()

    def test_downstream_failure_does_not_leave_connection_checked_out(self):
        engine = create_engine('sqlite://', poolclass=QueuePool, pool_size=1, max_overflow=0)
        Base.metadata.create_all(engine)
        class Client:
            def decide(self, **kwargs):
                raise TimeoutError('downstream timed out')
        try:
            with Session(engine) as db:
                session = repo.create_session(db)
                with self.assertRaises(TimeoutError): _advance_turn(db,session,Client())
                self.assertEqual(engine.pool.checkedout(),0)
        finally:engine.dispose()

import asyncio
from src.app.service_a.capacity import AdmissionLimitMiddleware

class AdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_overload_rejected_before_downstream_and_capacity_recovers(self):
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []
        async def app(scope, receive, send):
            calls.append(scope['path']); entered.set(); await release.wait()
        gate = AdmissionLimitMiddleware(app, limit=1)
        async def receive(): return {'type':'http.request','body':b''}
        messages = []
        async def send(message): messages.append(message)
        first = asyncio.create_task(gate({'type':'http','path':'/first'}, receive, send))
        await entered.wait()
        await gate({'type':'http','path':'/overload'}, receive, send)
        self.assertEqual(calls,['/first'])
        self.assertEqual(messages[0]['status'],503)
        self.assertIn((b'retry-after',b'1'),messages[0]['headers'])
        release.set(); await first
        await gate({'type':'http','path':'/after'}, receive, send)
        self.assertEqual(calls,['/first','/after'])

    async def test_cancellation_returns_admission_slot(self):
        entered = asyncio.Event()
        calls = []
        async def app(scope,receive,send):
            calls.append(scope['path'])
            if scope['path']=='/cancel':entered.set();await asyncio.Event().wait()
        gate=AdmissionLimitMiddleware(app,limit=1)
        async def unused(*args): pass
        request=asyncio.create_task(gate({'type':'http','path':'/cancel'},unused,unused))
        await entered.wait();request.cancel()
        with self.assertRaises(asyncio.CancelledError):await request
        await gate({'type':'http','path':'/next'},unused,unused)
        self.assertEqual(calls,['/cancel','/next'])

    async def test_exception_returns_admission_slot(self):
        calls=[]
        async def app(scope,receive,send):
            calls.append(scope['path'])
            if scope['path']=='/fail':raise RuntimeError('test failure')
        gate=AdmissionLimitMiddleware(app,limit=1)
        async def unused(*args):pass
        with self.assertRaises(RuntimeError):await gate({'type':'http','path':'/fail'},unused,unused)
        await gate({'type':'http','path':'/next'},unused,unused)
        self.assertEqual(calls,['/fail','/next'])
