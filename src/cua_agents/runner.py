from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass, field
from typing import TextIO
from uuid import uuid4

from .backend import AgentBackend
from .capture import CaptureBackend
from .executor import ActionExecutor
from .models import AgentTurn, ComputerAction, ComputerCall, RunEvent, Screenshot
from .safety import SafetyPolicy


@dataclass(frozen=True)
class RunResult:
    status: str
    final_text: str | None = None
    steps: int = 0


@dataclass(frozen=True)
class ApprovalRequest:
    approval_id: str
    reason: str
    message: str
    payload: dict


@dataclass
class AsyncRunSession:
    task: str
    backend: AgentBackend
    capture: CaptureBackend
    executor: ActionExecutor
    safety: SafetyPolicy
    max_steps: int = 25
    debug: bool = False
    session_id: str = field(default_factory=lambda: uuid4().hex)

    def __post_init__(self) -> None:
        self._sequence = 0
        self._history: list[RunEvent] = []
        self._subscribers: list[asyncio.Queue[RunEvent | None]] = []
        self._pending_approvals: dict[str, asyncio.Future[bool]] = {}
        self._stop_requested = False
        self._result: RunResult | None = None

    @property
    def result(self) -> RunResult | None:
        return self._result

    @property
    def pending_approval_ids(self) -> list[str]:
        return list(self._pending_approvals)

    @property
    def event_count(self) -> int:
        return len(self._history)

    def events_after(self, sequence: int) -> list[RunEvent]:
        return [event for event in self._history if event.sequence > sequence]

    async def events(self):
        queue: asyncio.Queue[RunEvent | None] = asyncio.Queue()
        for event in self._history:
            await queue.put(event)
        if self._result is not None:
            await queue.put(None)
        self._subscribers.append(queue)
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
        finally:
            if queue in self._subscribers:
                self._subscribers.remove(queue)

    async def run(self) -> RunResult:
        await self._emit("run_started", {"task": self.task})
        try:
            result = await self._run_loop()
        except asyncio.CancelledError:
            result = RunResult(status="stopped")
            await self._emit("run_stopped", {"status": result.status})
            raise
        except Exception as exc:
            result = RunResult(status="failed")
            await self._emit("run_failed", {"error": str(exc), "error_type": type(exc).__name__})

        self._result = result
        if result.status == "completed":
            await self._emit(
                "run_completed",
                {"status": result.status, "final_text": result.final_text, "steps": result.steps},
            )
        elif result.status != "failed":
            await self._emit(
                "run_stopped",
                {"status": result.status, "final_text": result.final_text, "steps": result.steps},
            )

        for subscriber in list(self._subscribers):
            await subscriber.put(None)
        return result

    async def approve(self, approval_id: str) -> None:
        await self._resolve_approval(approval_id, True)

    async def reject(self, approval_id: str) -> None:
        await self._resolve_approval(approval_id, False)

    async def stop(self) -> None:
        self._stop_requested = True
        for approval_id in list(self._pending_approvals):
            await self.reject(approval_id)

    async def _run_loop(self) -> RunResult:
        screenshot = await self._capture()
        turn = await asyncio.to_thread(self.backend.start, self.task, screenshot)

        for step in range(1, self.max_steps + 1):
            await self._emit_turn(step, turn)
            if self._stop_requested:
                return RunResult(status="stopped", steps=step - 1)
            if turn.is_done:
                return RunResult(status="completed", final_text=turn.final_text, steps=step - 1)

            call = turn.calls[0]
            acknowledged = await self._handle_safety_checks(call)
            if acknowledged is None:
                return RunResult(status="rejected_safety_check", steps=step - 1)

            for action in call.actions:
                approved = await self._approve_action(action)
                if not approved:
                    return RunResult(status="rejected_action", steps=step - 1)
                await self._emit("action_executing", {"action": _action_payload(action)})
                await asyncio.to_thread(self.executor.execute, action, screenshot)
                await self._emit("action_executed", {"action": _action_payload(action)})

            screenshot = await self._capture()
            turn = await asyncio.to_thread(
                self.backend.continue_after_call,
                call.call_id,
                screenshot,
                acknowledged,
            )

        return RunResult(status="max_steps_exceeded", steps=self.max_steps)

    async def _capture(self) -> Screenshot:
        screenshot = await asyncio.to_thread(self.capture.capture)
        await self._emit(
            "screenshot",
            {
                "width": screenshot.width,
                "height": screenshot.height,
                "image_url": screenshot.data_url,
            },
        )
        return screenshot

    async def _emit_turn(self, step: int, turn: AgentTurn) -> None:
        await self._emit(
            "turn",
            {
                "step": step,
                "response_id": turn.response_id,
                "summaries": turn.summaries,
                "final_text": turn.final_text,
                "has_calls": bool(turn.calls),
            },
        )
        if self.debug and turn.usage and turn.usage.has_values:
            await self._emit(
                "tokens",
                {
                    "input": turn.usage.input_tokens,
                    "cached": turn.usage.cached_tokens,
                    "output": turn.usage.output_tokens,
                    "label": turn.usage.label(),
                },
            )
        for call in turn.calls:
            for action in call.actions:
                await self._emit(
                    "proposed_action",
                    {"call_id": call.call_id, "action": _action_payload(action)},
                )

    async def _handle_safety_checks(self, call: ComputerCall) -> list[dict] | None:
        if not call.pending_safety_checks:
            return []
        approved = await self._request_approval(
            reason="safety_check",
            message="Acknowledge safety checks and continue?",
            payload={"call_id": call.call_id, "safety_checks": call.pending_safety_checks},
        )
        if approved:
            return call.pending_safety_checks
        return None

    async def _approve_action(self, action: ComputerAction) -> bool:
        if not self.safety.needs_confirmation(action):
            return True
        return await self._request_approval(
            reason="risky_action",
            message=f"Execute {action.label()}?",
            payload={"action": _action_payload(action)},
        )

    async def _request_approval(self, reason: str, message: str, payload: dict) -> bool:
        approval_id = uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self._pending_approvals[approval_id] = future
        await self._emit(
            "approval_required",
            {
                "approval_id": approval_id,
                "reason": reason,
                "message": message,
                **payload,
            },
        )
        try:
            return await future
        finally:
            self._pending_approvals.pop(approval_id, None)

    async def _resolve_approval(self, approval_id: str, approved: bool) -> None:
        future = self._pending_approvals.get(approval_id)
        if future is None or future.done():
            return
        future.set_result(approved)
        await self._emit(
            "approval_resolved",
            {"approval_id": approval_id, "approved": approved},
        )

    async def _emit(self, event_type: str, payload: dict) -> RunEvent:
        self._sequence += 1
        event = RunEvent(
            session_id=self.session_id,
            sequence=self._sequence,
            type=event_type,
            payload=payload,
        )
        self._history.append(event)
        for subscriber in list(self._subscribers):
            await subscriber.put(event)
        return event


@dataclass
class SessionManager:
    sessions: dict[str, AsyncRunSession] = field(default_factory=dict)

    def create(
        self,
        *,
        task: str,
        backend: AgentBackend,
        capture: CaptureBackend,
        executor: ActionExecutor,
        safety: SafetyPolicy,
        max_steps: int = 25,
        debug: bool = False,
        session_id: str | None = None,
    ) -> AsyncRunSession:
        session = AsyncRunSession(
            task=task,
            backend=backend,
            capture=capture,
            executor=executor,
            safety=safety,
            max_steps=max_steps,
            debug=debug,
            session_id=session_id or uuid4().hex,
        )
        self.sessions[session.session_id] = session
        return session

    async def approve(self, session_id: str, approval_id: str) -> None:
        await self._session(session_id).approve(approval_id)

    async def reject(self, session_id: str, approval_id: str) -> None:
        await self._session(session_id).reject(approval_id)

    async def stop(self, session_id: str) -> None:
        await self._session(session_id).stop()

    def _session(self, session_id: str) -> AsyncRunSession:
        try:
            return self.sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"Unknown session_id: {session_id}") from exc


def terminal_confirm(prompt: str) -> bool:
    answer = input(f"{prompt} [y/N] ").strip().lower()
    return answer in {"y", "yes"}


async def run_terminal_session(session: AsyncRunSession, out: TextIO = sys.stdout) -> RunResult:
    task = asyncio.create_task(session.run())
    async for event in session.events():
        await _handle_terminal_event(session, event, out)
    return await task


async def _handle_terminal_event(session: AsyncRunSession, event: RunEvent, out: TextIO) -> None:
    payload = event.payload
    if event.type == "turn":
        print(f"\nstep {payload['step']}", file=out)
        for summary in payload.get("summaries", []):
            print(f"summary: {summary}", file=out)
        if payload.get("final_text") and not payload.get("has_calls"):
            print(payload["final_text"], file=out)
    elif event.type == "tokens":
        print(f"tokens: {payload['label']}", file=out)
    elif event.type == "proposed_action":
        print(f"proposed: {payload['action']['label']}", file=out)
    elif event.type == "action_executing":
        print(f"execute: {payload['action']['label']}", file=out)
    elif event.type == "approval_required":
        if event.payload["reason"] == "safety_check":
            print("pending safety checks:", file=out)
            for check in event.payload.get("safety_checks", []):
                code = check.get("code", "unknown")
                message = check.get("message", "")
                print(f"- {code}: {message}", file=out)
        approved = await asyncio.to_thread(terminal_confirm, event.payload["message"])
        if approved:
            await session.approve(event.payload["approval_id"])
        else:
            await session.reject(event.payload["approval_id"])
    elif event.type == "run_failed":
        print(f"error: {payload['error']}", file=out)


def _action_payload(action: ComputerAction) -> dict:
    return {"type": action.type, "data": action.data, "label": action.label()}
