from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import AgentTurn, ComputerAction, ComputerCall, Screenshot


class ScriptedBackend:
    def __init__(self, turns: list[AgentTurn]) -> None:
        self._turns = turns
        self._index = 0

    @classmethod
    def from_file(cls, path: str | Path) -> "ScriptedBackend":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        turns = [_parse_turn(turn) for turn in data.get("turns", [])]
        return cls(turns)

    def start(self, task: str, screenshot: Screenshot) -> AgentTurn:
        return self._next_turn()

    def continue_after_call(
        self,
        call_id: str,
        screenshot: Screenshot,
        acknowledged_safety_checks: list[dict] | None = None,
    ) -> AgentTurn:
        return self._next_turn()

    def _next_turn(self) -> AgentTurn:
        if self._index >= len(self._turns):
            return AgentTurn(response_id=None, final_text="Script completed.")
        turn = self._turns[self._index]
        self._index += 1
        return turn


def _parse_turn(data: dict[str, Any]) -> AgentTurn:
    calls: list[ComputerCall] = []
    for call in data.get("calls", []):
        actions = [
            ComputerAction(type=str(action["type"]), data={k: v for k, v in action.items() if k != "type"})
            for action in call.get("actions", [])
        ]
        calls.append(
            ComputerCall(
                call_id=str(call.get("call_id", "scripted_call")),
                actions=actions,
                pending_safety_checks=list(call.get("pending_safety_checks", [])),
            )
        )
    return AgentTurn(
        response_id=data.get("response_id"),
        calls=calls,
        final_text=data.get("final_text"),
        summaries=list(data.get("summaries", [])),
    )
