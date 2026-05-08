from __future__ import annotations

import asyncio
import hmac
from dataclasses import asdict, replace
from typing import Callable, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from pydantic import BaseModel, Field

from .models import RunEvent
from .runner import AsyncRunSession, RunResult
from .session_factory import RunConfig, build_session, default_run_config


SessionBuilder = Callable[[RunConfig], AsyncRunSession]


class RunCreateRequest(BaseModel):
    task: str = Field(min_length=1)
    backend: Literal["openai", "openrouter", "lmstudio", "scripted"] | None = None
    script: str | None = None
    input_commander_url: str | None = None
    input_commander_width: int | None = Field(default=None, gt=0)
    input_commander_height: int | None = Field(default=None, gt=0)
    model: str | None = None
    lmstudio_url: str | None = None
    lmstudio_model: str | None = None
    lmstudio_api_key: str | None = None
    lmstudio_timeout: float | None = Field(default=None, gt=0)
    openrouter_url: str | None = None
    openrouter_model: str | None = None
    openrouter_api_key: str | None = None
    openrouter_http_referer: str | None = None
    openrouter_app_title: str | None = None
    openrouter_timeout: float | None = Field(default=None, gt=0)
    environment: str | None = None
    capture_backend: Literal["auto", "wayland", "spectacle", "x11", "windows", "mac"] | None = None
    spectacle_bin: str | None = None
    x11_capture_bin: str | None = None
    windows_capture_bin: str | None = None
    mac_capture_bin: str | None = None
    capture_mode: Literal["fullscreen", "current", "activewindow"] | None = None
    capture_delay_ms: int | None = Field(default=None, ge=0)
    tool_shape: Literal["ga", "ga_minimal", "preview"] | None = None
    max_steps: int | None = Field(default=None, gt=0)
    debug: bool | None = None
    safety_mode: Literal["confirm_risky", "step", "yes"] | None = None


class ApprovalDecision(BaseModel):
    approved: bool


class RunStore:
    def __init__(self, session_builder: SessionBuilder = build_session) -> None:
        self._session_builder = session_builder
        self._sessions: dict[str, AsyncRunSession] = {}
        self._tasks: dict[str, asyncio.Task[RunResult]] = {}
        self._active_session_id: str | None = None
        self._lock = asyncio.Lock()

    async def create_run(self, config: RunConfig) -> AsyncRunSession:
        async with self._lock:
            if self._active_session_id is not None:
                active = self._tasks.get(self._active_session_id)
                if active is not None and not active.done():
                    raise ActiveRunError(self._active_session_id)
                self._active_session_id = None

            session = self._session_builder(config)
            self._sessions[session.session_id] = session
            task = asyncio.create_task(session.run())
            self._tasks[session.session_id] = task
            self._active_session_id = session.session_id
            task.add_done_callback(lambda done: self._finish_run(session.session_id, done))
            return session

    def session(self, session_id: str) -> AsyncRunSession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise UnknownSessionError(session_id) from exc

    def task(self, session_id: str) -> asyncio.Task[RunResult] | None:
        return self._tasks.get(session_id)

    def _finish_run(self, session_id: str, task: asyncio.Task[RunResult]) -> None:
        if self._active_session_id == session_id:
            self._active_session_id = None
        if not task.cancelled():
            task.exception()


class ActiveRunError(RuntimeError):
    def __init__(self, session_id: str) -> None:
        super().__init__(f"run already active: {session_id}")
        self.session_id = session_id


class UnknownSessionError(KeyError):
    pass


def create_app(
    *,
    server_token: str | None = None,
    allow_no_auth: bool = False,
    session_builder: SessionBuilder = build_session,
) -> FastAPI:
    store = RunStore(session_builder=session_builder)
    app = FastAPI(title="cua-agents server")

    configured_token = server_token.strip() if server_token is not None else None

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        if allow_no_auth:
            return
        if not configured_token:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="server token is not configured",
            )
        if not _is_authorized(configured_token, authorization):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    auth = Depends(require_auth)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/runs", status_code=status.HTTP_202_ACCEPTED, dependencies=[auth])
    async def create_run(request: RunCreateRequest) -> dict:
        config = _config_from_request(request)
        try:
            session = await store.create_run(config)
        except ActiveRunError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={"error": "run already active", "session_id": exc.session_id},
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        return {
            "session_id": session.session_id,
            "status": "running",
            "events_url": f"/runs/{session.session_id}/events",
            "status_url": f"/runs/{session.session_id}",
        }

    @app.get("/runs/{session_id}", dependencies=[auth])
    async def get_run(session_id: str) -> dict:
        session = _get_session(store, session_id)
        return _session_status(session, store.task(session_id))

    @app.get("/runs/{session_id}/events", dependencies=[auth])
    async def get_events(session_id: str, after: int = Query(default=0, ge=0)) -> dict:
        session = _get_session(store, session_id)
        events = session.events_after(after)
        return {
            "session_id": session.session_id,
            "events": [_event_payload(event) for event in events],
            "result": _result_payload(session.result),
        }

    @app.post("/runs/{session_id}/approvals/{approval_id}", dependencies=[auth])
    async def resolve_approval(session_id: str, approval_id: str, decision: ApprovalDecision) -> dict:
        session = _get_session(store, session_id)
        if approval_id not in session.pending_approval_ids:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown approval_id")
        if decision.approved:
            await session.approve(approval_id)
        else:
            await session.reject(approval_id)
        return {"session_id": session.session_id, "approval_id": approval_id, "approved": decision.approved}

    @app.post("/runs/{session_id}/stop", dependencies=[auth])
    async def stop_run(session_id: str) -> dict:
        session = _get_session(store, session_id)
        await session.stop()
        return {"session_id": session.session_id, "status": "stop_requested"}

    app.state.cua_run_store = store
    return app


def _config_from_request(request: RunCreateRequest) -> RunConfig:
    config = default_run_config(request.task)
    if hasattr(request, "model_dump"):
        overrides = request.model_dump(exclude_unset=True)
    else:
        overrides = request.dict(exclude_unset=True)
    overrides.pop("task", None)
    return replace(config, **overrides)


def _is_authorized(
    configured_token: str,
    authorization: str | None,
) -> bool:
    if authorization is None:
        return False

    parts = authorization.strip().split(None, 1)
    if len(parts) != 2:
        return False
    scheme, token = parts
    return scheme.lower() == "bearer" and hmac.compare_digest(token.strip(), configured_token)


def _get_session(store: RunStore, session_id: str) -> AsyncRunSession:
    try:
        return store.session(session_id)
    except UnknownSessionError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="unknown session_id") from exc


def _session_status(session: AsyncRunSession, task: asyncio.Task[RunResult] | None) -> dict:
    running = task is not None and not task.done()
    result = session.result
    if running:
        run_status = "running"
    elif result is not None:
        run_status = result.status
    else:
        run_status = "unknown"
    return {
        "session_id": session.session_id,
        "status": run_status,
        "result": _result_payload(result),
        "event_count": session.event_count,
        "pending_approval_ids": session.pending_approval_ids,
    }


def _event_payload(event: RunEvent) -> dict:
    return asdict(event)


def _result_payload(result: RunResult | None) -> dict | None:
    if result is None:
        return None
    return asdict(result)
