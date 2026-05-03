from __future__ import annotations

import asyncio

from cua_agents.executor import ActionExecutor
from cua_agents.models import AgentTurn, ComputerAction, ComputerCall, RunEvent, Screenshot, TokenUsage
from cua_agents.runner import AsyncRunSession, SessionManager
from cua_agents.safety import SafetyPolicy


class FakeCapture:
    def capture(self) -> Screenshot:
        return Screenshot(png=b"png", width=100, height=100)


class FakeBackend:
    def __init__(self, turns: list[AgentTurn]) -> None:
        self.turns = turns
        self.index = 0
        self.acknowledged: list[dict] | None = None

    def start(self, task: str, screenshot: Screenshot) -> AgentTurn:
        return self._next()

    def continue_after_call(
        self,
        call_id: str,
        screenshot: Screenshot,
        acknowledged_safety_checks: list[dict] | None = None,
    ) -> AgentTurn:
        self.acknowledged = acknowledged_safety_checks
        return self._next()

    def _next(self) -> AgentTurn:
        turn = self.turns[self.index]
        self.index += 1
        return turn


class FakeExecutor(ActionExecutor):
    def __init__(self) -> None:
        self.actions: list[ComputerAction] = []

    def execute(self, action: ComputerAction, screenshot: Screenshot) -> None:
        self.actions.append(action)


async def collect_events(session: AsyncRunSession, approve: bool = True) -> list[RunEvent]:
    events: list[RunEvent] = []
    async for event in session.events():
        events.append(event)
        if event.type == "approval_required":
            if approve:
                await session.approve(event.payload["approval_id"])
            else:
                await session.reject(event.payload["approval_id"])
    return events


def test_session_completes_after_one_call() -> None:
    async def scenario() -> None:
        action = ComputerAction(type="wait", data={})
        backend = FakeBackend(
            [
                AgentTurn(response_id="r1", calls=[ComputerCall(call_id="c1", actions=[action])]),
                AgentTurn(response_id="r2", final_text="done"),
            ]
        )
        executor = FakeExecutor()
        session = AsyncRunSession(
            task="task",
            backend=backend,
            capture=FakeCapture(),
            executor=executor,
            safety=SafetyPolicy(mode="yes"),
            max_steps=3,
        )

        events_task = asyncio.create_task(collect_events(session))
        result = await session.run()
        events = await events_task

        assert result.status == "completed"
        assert result.final_text == "done"
        assert executor.actions == [action]
        assert [event.type for event in events if event.type in {"run_started", "run_completed"}] == [
            "run_started",
            "run_completed",
        ]
        assert any(event.type == "proposed_action" for event in events)

    asyncio.run(scenario())


def test_session_pauses_for_risky_action_and_resumes_after_approval() -> None:
    async def scenario() -> None:
        action = ComputerAction(type="type", data={"text": "password"})
        backend = FakeBackend(
            [
                AgentTurn(response_id="r1", calls=[ComputerCall(call_id="c1", actions=[action])]),
                AgentTurn(response_id="r2", final_text="done"),
            ]
        )
        executor = FakeExecutor()
        session = AsyncRunSession(
            task="task",
            backend=backend,
            capture=FakeCapture(),
            executor=executor,
            safety=SafetyPolicy(mode="confirm_risky"),
        )

        events_task = asyncio.create_task(collect_events(session))
        result = await session.run()
        events = await events_task

        approval = next(event for event in events if event.type == "approval_required")
        assert approval.payload["reason"] == "risky_action"
        assert result.status == "completed"
        assert executor.actions == [action]

    asyncio.run(scenario())


def test_session_stops_when_safety_check_is_rejected() -> None:
    async def scenario() -> None:
        backend = FakeBackend(
            [
                AgentTurn(
                    response_id="r1",
                    calls=[
                        ComputerCall(
                            call_id="c1",
                            actions=[ComputerAction(type="wait", data={})],
                            pending_safety_checks=[{"id": "s1", "code": "sensitive_domain"}],
                        )
                    ],
                )
            ]
        )
        session = AsyncRunSession(
            task="task",
            backend=backend,
            capture=FakeCapture(),
            executor=FakeExecutor(),
            safety=SafetyPolicy(mode="yes"),
        )

        events_task = asyncio.create_task(collect_events(session, approve=False))
        result = await session.run()
        events = await events_task

        assert result.status == "rejected_safety_check"
        assert any(event.type == "approval_required" for event in events)

    asyncio.run(scenario())


def test_session_stops_at_max_steps() -> None:
    async def scenario() -> None:
        backend = FakeBackend(
            [
                AgentTurn(
                    response_id="r1",
                    calls=[ComputerCall(call_id="c1", actions=[ComputerAction(type="screenshot")])],
                ),
                AgentTurn(
                    response_id="r2",
                    calls=[ComputerCall(call_id="c2", actions=[ComputerAction(type="screenshot")])],
                ),
            ]
        )
        session = AsyncRunSession(
            task="task",
            backend=backend,
            capture=FakeCapture(),
            executor=FakeExecutor(),
            safety=SafetyPolicy(mode="yes"),
            max_steps=1,
        )

        events_task = asyncio.create_task(collect_events(session))
        result = await session.run()
        await events_task

        assert result.status == "max_steps_exceeded"

    asyncio.run(scenario())


def test_session_emits_token_usage_when_debug_enabled() -> None:
    async def scenario() -> None:
        session = AsyncRunSession(
            task="task",
            backend=FakeBackend(
                [
                    AgentTurn(
                        response_id="r1",
                        final_text="done",
                        usage=TokenUsage(input_tokens=10, cached_tokens=4, output_tokens=2),
                    )
                ]
            ),
            capture=FakeCapture(),
            executor=FakeExecutor(),
            safety=SafetyPolicy(mode="yes"),
            debug=True,
        )

        events_task = asyncio.create_task(collect_events(session))
        result = await session.run()
        events = await events_task

        assert result.status == "completed"
        assert any(
            event.type == "tokens" and event.payload["label"] == "input=10, cached=4, output=2"
            for event in events
        )

    asyncio.run(scenario())


def test_session_hides_token_usage_without_debug() -> None:
    async def scenario() -> None:
        session = AsyncRunSession(
            task="task",
            backend=FakeBackend(
                [
                    AgentTurn(
                        response_id="r1",
                        final_text="done",
                        usage=TokenUsage(input_tokens=10, cached_tokens=4, output_tokens=2),
                    )
                ]
            ),
            capture=FakeCapture(),
            executor=FakeExecutor(),
            safety=SafetyPolicy(mode="yes"),
        )

        events_task = asyncio.create_task(collect_events(session))
        result = await session.run()
        events = await events_task

        assert result.status == "completed"
        assert not any(event.type == "tokens" for event in events)

    asyncio.run(scenario())


def test_session_manager_creates_and_controls_sessions() -> None:
    async def scenario() -> None:
        manager = SessionManager()
        session = manager.create(
            task="task",
            backend=FakeBackend(
                [
                    AgentTurn(
                        response_id="r1",
                        calls=[
                            ComputerCall(
                                call_id="c1",
                                actions=[ComputerAction(type="type", data={"text": "password"})],
                            )
                        ],
                    ),
                    AgentTurn(response_id="r2", final_text="done"),
                ]
            ),
            capture=FakeCapture(),
            executor=FakeExecutor(),
            safety=SafetyPolicy(mode="confirm_risky"),
        )

        async def approve_with_manager() -> None:
            async for event in session.events():
                if event.type == "approval_required":
                    await manager.approve(session.session_id, event.payload["approval_id"])

        events_task = asyncio.create_task(approve_with_manager())
        result = await session.run()
        await events_task

        assert result.status == "completed"
        assert manager.sessions[session.session_id] is session

    asyncio.run(scenario())


def test_two_sessions_can_run_concurrently() -> None:
    async def scenario() -> None:
        manager = SessionManager()
        session_a = manager.create(
            task="a",
            backend=FakeBackend([AgentTurn(response_id="a", final_text="done a")]),
            capture=FakeCapture(),
            executor=FakeExecutor(),
            safety=SafetyPolicy(mode="yes"),
        )
        session_b = manager.create(
            task="b",
            backend=FakeBackend([AgentTurn(response_id="b", final_text="done b")]),
            capture=FakeCapture(),
            executor=FakeExecutor(),
            safety=SafetyPolicy(mode="yes"),
        )

        result_a, result_b = await asyncio.gather(session_a.run(), session_b.run())

        assert result_a.status == "completed"
        assert result_b.status == "completed"
        assert session_a.session_id != session_b.session_id

    asyncio.run(scenario())
