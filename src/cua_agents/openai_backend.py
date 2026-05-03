from __future__ import annotations

from typing import Any

from .models import AgentTurn, ComputerAction, ComputerCall, Screenshot, TokenUsage


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _to_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)
    if hasattr(obj, "dict"):
        return obj.dict()
    data = getattr(obj, "__dict__", None)
    if isinstance(data, dict):
        return {key: value for key, value in data.items() if not key.startswith("_")}
    return {"value": obj}


class OpenAIComputerBackend:
    def __init__(
        self,
        *,
        model: str = "gpt-5.5",
        environment: str = "linux",
        tool_shape: str = "ga",
        client: Any | None = None,
    ) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self.client = client
        self.model = model
        self.environment = environment
        self.tool_shape = tool_shape
        self._previous_response_id: str | None = None
        self._last_dimensions: tuple[int, int] | None = None

    def start(self, task: str, screenshot: Screenshot) -> AgentTurn:
        self._last_dimensions = (screenshot.width, screenshot.height)
        response = self.client.responses.create(
            model=self.model,
            tools=[self._tool(screenshot)],
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": task},
                        {"type": "input_image", "image_url": screenshot.data_url},
                    ],
                }
            ],
        )
        return self._parse_response(response)

    def continue_after_call(
        self,
        call_id: str,
        screenshot: Screenshot,
        acknowledged_safety_checks: list[dict] | None = None,
    ) -> AgentTurn:
        self._last_dimensions = (screenshot.width, screenshot.height)
        response = self.client.responses.create(
            model=self.model,
            previous_response_id=self._previous_response_id,
            tools=[self._tool(screenshot)],
            input=[
                {
                    "type": "computer_call_output",
                    "call_id": call_id,
                    "acknowledged_safety_checks": acknowledged_safety_checks or [],
                    "output": {
                        "type": "computer_screenshot",
                        "image_url": screenshot.data_url,
                    },
                }
            ],
        )
        return self._parse_response(response)

    def _tool(self, screenshot: Screenshot) -> dict[str, Any]:
        if self.tool_shape == "preview":
            return {
                "type": "computer_use_preview",
                "display_width": screenshot.width,
                "display_height": screenshot.height,
                "environment": self.environment,
            }
        if self.tool_shape in {"ga", "ga_minimal"}:
            return {"type": "computer"}
        raise ValueError(f"Unsupported OpenAI computer tool shape: {self.tool_shape}")

    def _parse_response(self, response: Any) -> AgentTurn:
        response_id = _get(response, "id")
        self._previous_response_id = response_id

        calls: list[ComputerCall] = []
        summaries: list[str] = []
        output = _get(response, "output", []) or []
        for item in output:
            item_type = _get(item, "type")
            if item_type == "reasoning":
                summaries.extend(self._reasoning_summary(item))
                continue
            if item_type != "computer_call":
                continue

            raw_actions = _get(item, "actions")
            if raw_actions is None:
                raw_action = _get(item, "action")
                raw_actions = [raw_action] if raw_action is not None else []

            actions = [self._parse_action(action) for action in raw_actions]
            safety_checks = _get(item, "pending_safety_checks", None)
            if safety_checks is None:
                safety_checks = _get(item, "pending_safety_check", [])
            calls.append(
                ComputerCall(
                    call_id=str(_get(item, "call_id")),
                    actions=actions,
                    pending_safety_checks=[_to_dict(check) for check in safety_checks or []],
                )
            )

        final_text = _get(response, "output_text")
        if not final_text:
            final_text = self._message_text(output)

        return AgentTurn(
            response_id=response_id,
            calls=calls,
            final_text=final_text,
            summaries=summaries,
            usage=_parse_usage(_get(response, "usage")),
        )

    def _parse_action(self, action: Any) -> ComputerAction:
        data = _to_dict(action)
        action_type = str(data.pop("type", _get(action, "type", "")))
        return ComputerAction(type=action_type, data=data)

    def _reasoning_summary(self, item: Any) -> list[str]:
        summary = _get(item, "summary", []) or []
        values: list[str] = []
        for entry in summary:
            text = _get(entry, "text")
            if text is None:
                text = _get(entry, "summary_text")
            if text:
                values.append(str(text))
        return values

    def _message_text(self, output: list[Any]) -> str | None:
        chunks: list[str] = []
        for item in output:
            if _get(item, "type") != "message":
                continue
            for content in _get(item, "content", []) or []:
                text = _get(content, "text")
                if text:
                    chunks.append(str(text))
        if chunks:
            return "\n".join(chunks)
        return None


def _parse_usage(usage: Any) -> TokenUsage | None:
    if usage is None:
        return None

    input_tokens = _get(usage, "input_tokens")
    if input_tokens is None:
        input_tokens = _get(usage, "prompt_tokens")

    output_tokens = _get(usage, "output_tokens")
    if output_tokens is None:
        output_tokens = _get(usage, "completion_tokens")

    input_details = _get(usage, "input_tokens_details")
    if input_details is None:
        input_details = _get(usage, "prompt_tokens_details")
    cached_tokens = _get(input_details, "cached_tokens") if input_details is not None else None

    parsed = TokenUsage(
        input_tokens=_int_or_none(input_tokens),
        cached_tokens=_int_or_none(cached_tokens),
        output_tokens=_int_or_none(output_tokens),
    )
    return parsed if parsed.has_values else None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
