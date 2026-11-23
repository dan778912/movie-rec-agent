"""Per-process admission before sync dependencies can occupy worker threads.

This is load shedding, not a queue or a global rate limit. A default of 12
leaves headroom under the unchanged SQLAlchemy pool capacity (5 + 10).
Each Uvicorn worker has its own counter and pool; budget total DB connections
when increasing workers. Zero explicitly disables protection for baselines.
"""
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


class AdmissionLimitMiddleware:
    def __init__(self, app: ASGIApp, limit: int = 12):
        if type(limit) is not int or limit < 0:
            raise ValueError('Admission limit must be a nonnegative integer')
        self.app = app
        self.limit = limit
        self.active = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope['type'] != 'http' or self.limit == 0:
            await self.app(scope, receive, send)
            return
        # No await between checking and reserving: atomic on the worker's loop.
        if self.active >= self.limit:
            response = JSONResponse({'detail':'Service temporarily at capacity. Retry later.'},
                                    status_code=503, headers={'Retry-After':'1'})
            await response(scope, receive, send)
            return
        self.active += 1
        try:
            # Hold the slot through response delivery and dependency cleanup.
            await self.app(scope, receive, send)
        finally:
            self.active -= 1
