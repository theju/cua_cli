from __future__ import annotations

import time
from collections.abc import Callable

from fastapi.testclient import TestClient

from cua_agents.executor import ActionExecutor
from cua_agents.models import AgentTurn, ComputerAction, ComputerCall, Screenshot
from cua_agents.runner import AsyncRunSession
from cua_agents.safety import SafetyPolicy
from cua_agents.server import create_app
from cua_agents.session_factory import RunConfig


class FakeCapture:
    def capture(self) -> Screenshot:
        return Screenshot(png=b"png", width=100, height=100)


class FakeBackend:
    def __init__(self, turns: list[AgentTurn]) -> None:
        self.turns = list(turns)
        self.index = 0

    def start(self, task: str, screenshot: Screenshot) -> AgentTurn:
        return self._next()

    def continue_after_call(
        self,
        call_id: str,
        screenshot: Screenshot,
        acknowledged_safety_checks: list[dict] | None = None,
    ) -> AgentTurn:
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


def auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-token"}


def wait_until(predicate: Callable[[], dict], condition: Callable[[dict], bool]) -> dict:
    last: dict = {}
    for _ in range(100):
        last = predicate()
        if condition(last):
            return last
        time.sleep(0.01)
    raise AssertionError(f"condition was not met; last payload: {last}")


def build_fake_session(config: RunConfig, turns: list[AgentTurn]) -> AsyncRunSession:
    return AsyncRunSession(
        task=config.task,
        backend=FakeBackend(turns),
        capture=FakeCapture(),
        executor=FakeExecutor(),
        safety=SafetyPolicy(mode=config.safety_mode),
        max_steps=config.max_steps,
        debug=config.debug,
    )


def test_health_does_not_require_auth() -> None:
    app = create_app(server_token="test-token")

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_runs_require_bearer_token() -> None:
    app = create_app(server_token="test-token")

    with TestClient(app) as client:
        response = client.post("/runs", json={"task": "hello"})

    assert response.status_code == 401


def test_bearer_auth_accepts_whitespace() -> None:
    def builder(config: RunConfig) -> AsyncRunSession:
        return build_fake_session(config, [AgentTurn(response_id="r1", final_text="done")])

    app = create_app(server_token=" test-token\n", session_builder=builder)

    with TestClient(app) as client:
        bearer = client.post(
            "/runs",
            json={"task": "bearer"},
            headers={"Authorization": "Bearer   test-token  "},
        )
        session_id = bearer.json()["session_id"]
        wait_until(
            lambda: client.get(f"/runs/{session_id}", headers=auth_headers()).json(),
            lambda payload: payload["status"] == "completed",
        )

    assert bearer.status_code == 202


def test_create_run_and_poll_events() -> None:
    def builder(config: RunConfig) -> AsyncRunSession:
        return build_fake_session(config, [AgentTurn(response_id="r1", final_text="done")])

    app = create_app(server_token="test-token", session_builder=builder)

    with TestClient(app) as client:
        created = client.post("/runs", json={"task": "hello"}, headers=auth_headers())
        session_id = created.json()["session_id"]
        status_payload = wait_until(
            lambda: client.get(f"/runs/{session_id}", headers=auth_headers()).json(),
            lambda payload: payload["status"] == "completed",
        )
        events = client.get(f"/runs/{session_id}/events", headers=auth_headers()).json()

    assert created.status_code == 202
    assert status_payload["result"]["final_text"] == "done"
    assert [event["type"] for event in events["events"]] == [
        "run_started",
        "screenshot",
        "turn",
        "run_completed",
    ]


def test_rejects_second_active_run() -> None:
    action = ComputerAction(type="type", data={"text": "password"})

    def builder(config: RunConfig) -> AsyncRunSession:
        return build_fake_session(
            config,
            [
                AgentTurn(response_id="r1", calls=[ComputerCall(call_id="c1", actions=[action])]),
                AgentTurn(response_id="r2", final_text="done"),
            ],
        )

    app = create_app(server_token="test-token", session_builder=builder)

    with TestClient(app) as client:
        first = client.post(
            "/runs",
            json={"task": "first", "safety_mode": "confirm_risky"},
            headers=auth_headers(),
        )
        session_id = first.json()["session_id"]
        wait_until(
            lambda: client.get(f"/runs/{session_id}", headers=auth_headers()).json(),
            lambda payload: bool(payload["pending_approval_ids"]),
        )
        second = client.post("/runs", json={"task": "second"}, headers=auth_headers())
        approval_id = client.get(f"/runs/{session_id}", headers=auth_headers()).json()[
            "pending_approval_ids"
        ][0]
        client.post(
            f"/runs/{session_id}/approvals/{approval_id}",
            json={"approved": False},
            headers=auth_headers(),
        )

    assert first.status_code == 202
    assert second.status_code == 409


def test_approval_endpoint_resumes_paused_run() -> None:
    action = ComputerAction(type="type", data={"text": "password"})

    def builder(config: RunConfig) -> AsyncRunSession:
        return build_fake_session(
            config,
            [
                AgentTurn(response_id="r1", calls=[ComputerCall(call_id="c1", actions=[action])]),
                AgentTurn(response_id="r2", final_text="done"),
            ],
        )

    app = create_app(server_token="test-token", session_builder=builder)

    with TestClient(app) as client:
        created = client.post(
            "/runs",
            json={"task": "type secret", "safety_mode": "confirm_risky"},
            headers=auth_headers(),
        )
        session_id = created.json()["session_id"]
        paused = wait_until(
            lambda: client.get(f"/runs/{session_id}", headers=auth_headers()).json(),
            lambda payload: bool(payload["pending_approval_ids"]),
        )
        approval_id = paused["pending_approval_ids"][0]
        approval = client.post(
            f"/runs/{session_id}/approvals/{approval_id}",
            json={"approved": True},
            headers=auth_headers(),
        )
        completed = wait_until(
            lambda: client.get(f"/runs/{session_id}", headers=auth_headers()).json(),
            lambda payload: payload["status"] == "completed",
        )
        events = client.get(f"/runs/{session_id}/events", headers=auth_headers()).json()

    assert approval.status_code == 200
    assert completed["result"]["final_text"] == "done"
    assert "approval_resolved" in [event["type"] for event in events["events"]]
    assert "action_executed" in [event["type"] for event in events["events"]]


def test_completed_run_releases_active_lock() -> None:
    def builder(config: RunConfig) -> AsyncRunSession:
        return build_fake_session(config, [AgentTurn(response_id="r1", final_text="done")])

    app = create_app(server_token="test-token", session_builder=builder)

    with TestClient(app) as client:
        first = client.post("/runs", json={"task": "first"}, headers=auth_headers())
        first_id = first.json()["session_id"]
        wait_until(
            lambda: client.get(f"/runs/{first_id}", headers=auth_headers()).json(),
            lambda payload: payload["status"] == "completed",
        )
        second = client.post("/runs", json={"task": "second"}, headers=auth_headers())

    assert first.status_code == 202
    assert second.status_code == 202


def test_stop_endpoint_requests_stop() -> None:
    action = ComputerAction(type="type", data={"text": "password"})

    def builder(config: RunConfig) -> AsyncRunSession:
        return build_fake_session(
            config,
            [
                AgentTurn(response_id="r1", calls=[ComputerCall(call_id="c1", actions=[action])]),
                AgentTurn(response_id="r2", final_text="done"),
            ],
        )

    app = create_app(server_token="test-token", session_builder=builder)

    with TestClient(app) as client:
        created = client.post(
            "/runs",
            json={"task": "stop me", "safety_mode": "confirm_risky"},
            headers=auth_headers(),
        )
        session_id = created.json()["session_id"]
        wait_until(
            lambda: client.get(f"/runs/{session_id}", headers=auth_headers()).json(),
            lambda payload: bool(payload["pending_approval_ids"]),
        )
        stopped = client.post(f"/runs/{session_id}/stop", headers=auth_headers())
        final = wait_until(
            lambda: client.get(f"/runs/{session_id}", headers=auth_headers()).json(),
            lambda payload: payload["status"] == "rejected_action",
        )

    assert stopped.status_code == 200
    assert final["result"]["status"] == "rejected_action"
