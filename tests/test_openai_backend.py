from __future__ import annotations

from types import SimpleNamespace

from cua_agents.models import Screenshot
from cua_agents.openai_backend import OpenAIComputerBackend


class FakeResponses:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeClient:
    def __init__(self, response: object) -> None:
        self.responses = FakeResponses(response)


def test_parses_batched_ga_actions() -> None:
    response = SimpleNamespace(
        id="resp_1",
        output=[
            SimpleNamespace(type="reasoning", summary=[SimpleNamespace(text="Need to click.")]),
            SimpleNamespace(
                type="computer_call",
                call_id="call_1",
                actions=[
                    SimpleNamespace(type="move", x=10, y=20),
                    SimpleNamespace(type="click", x=10, y=20, button="left"),
                ],
                pending_safety_checks=[],
            ),
        ],
        output_text=None,
    )
    client = FakeClient(response)
    backend = OpenAIComputerBackend(client=client)

    turn = backend.start("task", Screenshot(png=b"png", width=100, height=50))

    assert turn.response_id == "resp_1"
    assert turn.summaries == ["Need to click."]
    assert turn.calls[0].call_id == "call_1"
    assert [action.type for action in turn.calls[0].actions] == ["move", "click"]
    assert client.responses.calls[0]["tools"][0] == {"type": "computer"}


def test_parses_responses_token_usage() -> None:
    response = SimpleNamespace(
        id="resp_1",
        output=[],
        output_text="done",
        usage=SimpleNamespace(
            input_tokens=100,
            input_tokens_details=SimpleNamespace(cached_tokens=25),
            output_tokens=12,
        ),
    )
    backend = OpenAIComputerBackend(client=FakeClient(response))

    turn = backend.start("task", Screenshot(png=b"png", width=100, height=50))

    assert turn.usage is not None
    assert turn.usage.input_tokens == 100
    assert turn.usage.cached_tokens == 25
    assert turn.usage.output_tokens == 12


def test_ga_minimal_alias_uses_current_computer_tool_shape() -> None:
    response = SimpleNamespace(id="resp_1", output=[], output_text="done")
    client = FakeClient(response)
    backend = OpenAIComputerBackend(client=client, tool_shape="ga_minimal")

    backend.start("task", Screenshot(png=b"png", width=100, height=50))

    assert client.responses.calls[0]["tools"][0] == {"type": "computer"}


def test_preview_tool_shape_keeps_legacy_display_fields() -> None:
    response = SimpleNamespace(id="resp_1", output=[], output_text="done")
    client = FakeClient(response)
    backend = OpenAIComputerBackend(client=client, tool_shape="preview")

    backend.start("task", Screenshot(png=b"png", width=100, height=50))

    assert client.responses.calls[0]["tools"][0] == {
        "type": "computer_use_preview",
        "display_width": 100,
        "display_height": 50,
        "environment": "linux",
    }


def test_parses_legacy_single_action() -> None:
    response = {
        "id": "resp_1",
        "output": [
            {
                "type": "computer_call",
                "call_id": "call_1",
                "action": {"type": "keypress", "keys": ["ENTER"]},
                "pending_safety_checks": [{"id": "safe_1", "code": "malicious_instructions"}],
            }
        ],
    }
    backend = OpenAIComputerBackend(client=FakeClient(response))

    turn = backend.start("task", Screenshot(png=b"png", width=100, height=50))

    assert turn.calls[0].actions[0].type == "keypress"
    assert turn.calls[0].actions[0].data == {"keys": ["ENTER"]}
    assert turn.calls[0].pending_safety_checks[0]["code"] == "malicious_instructions"


def test_continue_sends_computer_call_output_with_previous_response_id() -> None:
    response = SimpleNamespace(id="resp_2", output=[], output_text="done")
    client = FakeClient(response)
    backend = OpenAIComputerBackend(client=client)
    backend._previous_response_id = "resp_1"

    turn = backend.continue_after_call(
        "call_1",
        Screenshot(png=b"png", width=100, height=50),
        acknowledged_safety_checks=[{"id": "safe_1"}],
    )

    payload = client.responses.calls[0]
    assert payload["previous_response_id"] == "resp_1"
    assert payload["input"][0]["type"] == "computer_call_output"
    assert payload["input"][0]["call_id"] == "call_1"
    assert payload["input"][0]["acknowledged_safety_checks"] == [{"id": "safe_1"}]
    assert turn.final_text == "done"
