"""Service A-compatible HTTP entrypoint; one admitted inference request at a time."""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from src.agent.decide import InitialRetrievalError, RequestError, decide, resume
from src.agent.trace import telemetry_record
from src.agent.controller import TransitionError, MissingCandidateError
from .continuation import Continuations, InvalidContinuation
from .runtime import Timed, load_resources

LOG = logging.getLogger("service_b")


class Clarification(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class DecideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    profile_text: str
    history_movie_ids: list[StrictInt]
    clarifications: list[Clarification] = Field(default_factory=list)


class ResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    continuation: str = Field(min_length=1, max_length=2_000_000)
    answer: str = Field(min_length=1)


def create_app(resources=None, config_path=None, state_secret=None):
    @asynccontextmanager
    async def lifespan(app):
        owned = resources is None
        app.state.resources = resources if not owned else await asyncio.to_thread(
            load_resources, config_path or os.environ.get("SERVICE_B_CONFIG", "configs/service_b_mock_v1.json"))
        app.state.stopping = False
        app.state.background = set()
        app.state.inference = set()
        try:
            yield
        finally:
            app.state.stopping = True
            await asyncio.gather(*list(app.state.inference), return_exceptions=True)
            await asyncio.gather(*list(app.state.background), return_exceptions=True)
            if owned:
                await asyncio.to_thread(app.state.resources.close)

    app = FastAPI(title="Movie Recommendation Service B", lifespan=lifespan)
    admission = threading.Lock()

    @app.get("/health/live")
    def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    async def ready(response: Response):
        r = app.state.resources
        index_ready = await asyncio.to_thread(r.retriever.check_health) if hasattr(r.retriever, "check_health") else True
        is_ready = r.ready and index_ready
        response.status_code = 200 if is_ready else 503
        return {"ready": is_ready, "model_ready": r.ready, "index_ready": index_ready,
                "mode": r.mode, "index_version": r.retriever.index_version,
                "policy": r.agent_config.label}

    def codec(r):
        try:
            return Continuations(state_secret or os.environ.get("SERVICE_B_STATE_SECRET"), r.identity)
        except ValueError as error:
            raise HTTPException(503, {"code": "continuation_not_configured", "reason": str(error)}) from error

    def run(r, payload, request_id, timeout, *, state=None, trace_id=None, versioned=False):
        start = time.perf_counter()
        model, retrieval = Timed(r.model), Timed(r.retriever)
        if hasattr(r.model, "deadline"):
            r.model.deadline = time.monotonic() + timeout
        try:
            if state is None:
                decision = decide(payload, model=model, retriever=retrieval, catalog=r.catalog, config=r.agent_config)
            else:
                decision = resume(state, payload["answer"], "client_supplied", model=model,
                                  retriever=retrieval, catalog=r.catalog)
            result = decision.response
            s = decision.state
            live_trace = {"schema_version": "service-b-live-episode-v1", "trace_id": trace_id,
                "initial_candidate_movie_ids": s.initial_pool, "events": s.events,
                "final_candidate_movie_ids": s.pool, "termination_reason": s.termination_reason,
                "index_version": s.initial_index_version}
            LOG.info(json.dumps({"request_id": request_id, "mode": r.mode,
                "trace_segment": decision.state.events, "telemetry": telemetry_record(decision.state),
                "model_call_audit": model.model_call_audit,
                "latency_ms": {"model": model.seconds * 1000, "retrieval": retrieval.seconds * 1000,
                               "end_to_end": (time.perf_counter() - start) * 1000}}))
            if result["action"] == "error":
                details = {"code": "agent_failed", "reason": result["reason"]}
                if versioned:
                    details.update(trace_id=trace_id, decision_trace=live_trace, telemetry=telemetry_record(s))
                raise HTTPException(503, details)
            if result["action"] == "recommend":
                items = result["items"]
                if (len(items) != 10 or len({i["movie_id"] for i in items}) != 10
                        or any(i["movie_id"] in decision.state.history_movie_ids or i["movie_id"] not in r.catalog
                               or not math.isfinite(i["score"]) or not i["explanation"].strip()
                               or i["rank"] != n for n, i in enumerate(items, 1))):
                    raise HTTPException(503, {"code": "invalid_agent_response"})
            if versioned:
                return {**result, "contract_version": "service-b-v1", "request_id": request_id,
                    "trace_id": trace_id, "mode": r.mode,
                    "continuation": codec(r).encode(s, trace_id) if result["action"] == "ask" else None,
                    "decision_trace": live_trace, "telemetry": telemetry_record(s),
                    "model_call_audit": model.model_call_audit}
            return result
        except (RequestError, TransitionError) as error:
            raise HTTPException(422, str(error)) from error
        except InitialRetrievalError as error:
            raise HTTPException(503, {"code": "initial_retrieval_failed"}) from error
        except MissingCandidateError as error:
            raise HTTPException(503, {"code": "catalog_gap"}) from error
        finally:
            if hasattr(r.model, "deadline"):
                r.model.deadline = None
            admission.release()

    async def dispatch(payload, response, *, state=None, trace_id=None, versioned=False):
        r = app.state.resources
        if not r.ready:
            raise HTTPException(503, {"code": "not_ready"})
        if not admission.acquire(blocking=False):
            raise HTTPException(429, {"code": "inference_busy"}, headers={"Retry-After": "1"})
        request_id = uuid.uuid4().hex
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Service-B-Mode"] = r.mode
        timeout = r.config.get("request_timeout_seconds", 25)
        task = asyncio.create_task(asyncio.to_thread(run, r, payload, request_id, timeout,
            state=state, trace_id=trace_id or request_id, versioned=versioned))
        app.state.inference.add(task)
        async def recover():
            try:
                restored = await asyncio.to_thread(r.model.recover)
                LOG.warning(json.dumps({"code": "inference_recovery", "restored": restored}))
            except Exception as error:
                LOG.error(json.dumps({"code": "inference_recovery_failed", "reason": str(error)}))
        def finished(t):
            app.state.inference.discard(t)
            if not t.cancelled():
                t.exception()
            if not app.state.stopping and not r.ready and hasattr(r.model, "recover"):
                recovery = asyncio.create_task(recover())
                app.state.background.add(recovery)
                recovery.add_done_callback(app.state.background.discard)
        # Shield keeps admission held until the actual worker finishes after HTTP cancellation.
        task.add_done_callback(finished)
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout)
        except HTTPException as error:
            error.headers = {**(error.headers or {}), "X-Request-ID": request_id, "X-Service-B-Mode": r.mode}
            raise
        except asyncio.TimeoutError as error:
            LOG.warning(json.dumps({"request_id": request_id, "code": "request_timeout"}))
            raise HTTPException(504, {"code": "request_timeout", "request_id": request_id}) from error

    @app.post("/decide")
    async def route(payload: DecideRequest, response: Response):
        return await dispatch(payload.model_dump(), response)

    @app.post("/v1/decide")
    async def start(payload: DecideRequest, response: Response):
        codec(app.state.resources)
        if payload.clarifications:
            raise HTTPException(422, "Start a new v1 episode without prior turns; use /v1/resume to continue")
        return await dispatch(payload.model_dump(), response, versioned=True)

    @app.post("/v1/resume")
    async def continue_episode(payload: ResumeRequest, response: Response):
        try:
            state, trace_id = codec(app.state.resources).decode(payload.continuation)
            if not payload.answer.strip():
                raise InvalidContinuation("answer must be nonempty")
        except InvalidContinuation as error:
            raise HTTPException(422, {"code": "invalid_continuation", "reason": str(error)}) from error
        return await dispatch(payload.model_dump(), response, state=state, trace_id=trace_id, versioned=True)

    return app


app = create_app()
