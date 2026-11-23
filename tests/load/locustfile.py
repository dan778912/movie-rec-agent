"""Service A HTTP workloads. Every HTTP request is followed by 0.5-1.5s think time.

Run against a disposable database. Complete sessions create persistent rows.
Use MixedUser for the historical 5:2 task ratio, or SearchUser/SessionUser
for endpoint isolation. No model, held-out labels, or production data is needed.
"""
import random
import gevent
from locust import HttpUser


class WorkloadUser(HttpUser):
    abstract = True
    # Think time lives in request(), rather than once per multi-request task.
    wait_time = lambda self: 0

    def request(self, method, path, name, expected=200, **kwargs):
        result = None
        with self.client.request(method, path, name=name, timeout=5,
                                 catch_response=True, **kwargs) as response:
            if response.status_code == expected:
                try:
                    result = response.json()
                except ValueError:
                    response.failure('Invalid JSON response')
            else:
                response.failure(f'HTTP {response.status_code}')
        gevent.sleep(random.uniform(0.5, 1.5))
        return result

    def search(self):
        self.request('GET', '/movies/search', '/movies/search',
                     params={'q': random.choice(['Star', 'Love', 'Alien', 'Toy', 'Man'])})

    def session_flow(self):
        session = self.request('POST', '/sessions', '/sessions', expected=201,
                               json={'ratings':[{'movie_id':1,'title':'Toy Story (1995)','rating':4.5}]})
        if not session:
            return
        sid = session['id']
        decision = self.request('POST', f'/sessions/{sid}/turn', '/sessions/:id/turn')
        if not decision:
            return
        if decision.get('action') == 'ask':
            decision = self.request('POST', f'/sessions/{sid}/clarifications',
                                    '/sessions/:id/clarifications', json={'answer':'Science fiction'})
        if not decision or decision.get('action') != 'recommend':
            return
        self.request('GET', f'/sessions/{sid}/recommendations', '/sessions/:id/recommendations')
        self.request('GET', f"/users/{session['user_id']}/history", '/users/:id/history')


class MixedUser(WorkloadUser):
    tasks = {WorkloadUser.search:5, WorkloadUser.session_flow:2}


class SearchUser(WorkloadUser):
    tasks = [WorkloadUser.search]


class SessionUser(WorkloadUser):
    tasks = [WorkloadUser.session_flow]
