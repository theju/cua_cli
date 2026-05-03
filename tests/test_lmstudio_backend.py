from __future__ import annotations

from cua_agents.lmstudio_backend import LMStudioBackend, _extract_json_object
from cua_agents.models import Screenshot


def screenshot() -> Screenshot:
    return Screenshot(png=b"png", width=100, height=50)


def response(content: str, usage: dict | None = None) -> dict:
    data = {
        "id": "chatcmpl_1",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                }
            }
        ],
    }
    if usage is not None:
        data["usage"] = usage
    return data


def test_start_posts_openai_compatible_lmstudio_payload() -> None:
    requests: list[tuple[str, dict, dict, float]] = []

    def http_post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
        requests.append((url, payload, headers, timeout))
        return response(
            '{"calls":[{"call_id":"c1","actions":[{"type":"click","x":10,"y":20}]}],'
            '"summaries":["click target"],"final_text":null}'
        )

    backend = LMStudioBackend(
        base_url="http://localhost:1234/v1",
        model="local-vision",
        api_key="secret",
        timeout=3,
        http_post=http_post,
    )

    turn = backend.start("click the target", screenshot())

    url, payload, headers, timeout = requests[0]
    assert url == "http://localhost:1234/v1/chat/completions"
    assert payload["model"] == "local-vision"
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert headers["Authorization"] == "Bearer secret"
    assert timeout == 3
    assert turn.summaries == ["click target"]
    assert turn.calls[0].call_id == "c1"
    assert turn.calls[0].actions[0].type == "click"
    assert turn.calls[0].actions[0].data == {"x": 10, "y": 20}


def test_continue_preserves_message_history() -> None:
    requests: list[dict] = []

    def http_post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
        requests.append(payload)
        if len(requests) == 1:
            return response('{"calls":[{"call_id":"c1","actions":[{"type":"wait","ms":1}]}]}')
        return response('{"calls":[],"final_text":"done"}')

    backend = LMStudioBackend(http_post=http_post)

    backend.start("wait", screenshot())
    turn = backend.continue_after_call("c1", screenshot())

    assert len(requests[1]["messages"]) == 4
    assert requests[1]["messages"][2]["role"] == "assistant"
    assert requests[1]["messages"][3]["role"] == "user"
    assert turn.final_text == "done"
    assert turn.calls == []


def test_plain_text_response_is_treated_as_final_text() -> None:
    backend = LMStudioBackend(http_post=lambda url, payload, headers, timeout: response("I am done."))

    turn = backend.start("task", screenshot())

    assert turn.final_text == "I am done."
    assert turn.calls == []


def test_parses_lmstudio_token_usage() -> None:
    backend = LMStudioBackend(
        http_post=lambda url, payload, headers, timeout: response(
            '{"calls":[],"final_text":"done"}',
            usage={
                "prompt_tokens": 42,
                "prompt_tokens_details": {"cached_tokens": 7},
                "completion_tokens": 9,
            },
        )
    )

    turn = backend.start("task", screenshot())

    assert turn.usage is not None
    assert turn.usage.input_tokens == 42
    assert turn.usage.cached_tokens == 7
    assert turn.usage.output_tokens == 9


def test_extract_json_object_handles_markdown_fence() -> None:
    parsed = _extract_json_object(
        """
```json
{"calls": [], "final_text": "done"}
```
"""
    )

    assert parsed == {"calls": [], "final_text": "done"}
