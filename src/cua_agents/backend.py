from __future__ import annotations

from typing import Protocol

from .models import AgentTurn, Screenshot


class AgentBackend(Protocol):
    def start(self, task: str, screenshot: Screenshot) -> AgentTurn:
        """Start an agent run from a user task and initial screenshot."""

    def continue_after_call(
        self,
        call_id: str,
        screenshot: Screenshot,
        acknowledged_safety_checks: list[dict] | None = None,
    ) -> AgentTurn:
        """Continue after a computer call has been executed."""
