from __future__ import annotations

import base64
from dataclasses import dataclass, field
from time import time
from typing import Any


@dataclass(frozen=True)
class Screenshot:
    png: bytes
    width: int
    height: int

    @property
    def data_url(self) -> str:
        encoded = base64.b64encode(self.png).decode("ascii")
        return f"data:image/png;base64,{encoded}"


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int | None = None
    cached_tokens: int | None = None
    output_tokens: int | None = None

    def label(self) -> str:
        parts: list[str] = []
        if self.input_tokens is not None:
            parts.append(f"input={self.input_tokens}")
        if self.cached_tokens is not None:
            parts.append(f"cached={self.cached_tokens}")
        if self.output_tokens is not None:
            parts.append(f"output={self.output_tokens}")
        return ", ".join(parts)

    @property
    def has_values(self) -> bool:
        return any(
            value is not None
            for value in (self.input_tokens, self.cached_tokens, self.output_tokens)
        )


@dataclass(frozen=True)
class ComputerAction:
    type: str
    data: dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        if self.data:
            args = ", ".join(f"{key}={value!r}" for key, value in sorted(self.data.items()))
            return f"{self.type}({args})"
        return f"{self.type}()"


@dataclass(frozen=True)
class ComputerCall:
    call_id: str
    actions: list[ComputerAction]
    pending_safety_checks: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class AgentTurn:
    response_id: str | None
    calls: list[ComputerCall] = field(default_factory=list)
    final_text: str | None = None
    summaries: list[str] = field(default_factory=list)
    usage: TokenUsage | None = None

    @property
    def is_done(self) -> bool:
        return not self.calls


@dataclass(frozen=True)
class RunEvent:
    session_id: str
    sequence: int
    type: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time)
