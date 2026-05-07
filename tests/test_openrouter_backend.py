from __future__ import annotations

from cua_agents.models import Screenshot
from cua_agents.openrouter_backend import OpenRouterBackend


def screenshot() -> Screenshot:
    return Screenshot(png=b"png", width=100, height=50)


def response(content: str, usage: dict | None = None) -> dict:
    data = {
        "id": "gen_1",
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


def test_start_posts_openrouter_chat_payload() -> None:
    requests: list[tuple[str, dict, dict, float]] = []

    def http_post(url: str, payload: dict, headers: dict, timeout: float) -> dict:
        requests.append((url, payload, headers, timeout))
        return response(
            '{"calls":[{"actions":[{"type":"click","x":10,"y":20}]}],'
            '"summaries":["click target"],"final_text":null}'
        )

    backend = OpenRouterBackend(
        base_url="https://openrouter.ai/api/v1",
        model="anthropic/claude-3.5-sonnet",
        api_key="secret",
        http_referer="https://example.test",
        app_title="test app",
        timeout=3,
        http_post=http_post,
    )

    turn = backend.start("click the target", screenshot())

    url, payload, headers, timeout = requests[0]
    assert url == "https://openrouter.ai/api/v1/chat/completions"
    assert payload["model"] == "anthropic/claude-3.5-sonnet"
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert headers["Authorization"] == "Bearer secret"
    assert headers["HTTP-Referer"] == "https://example.test"
    assert headers["X-Title"] == "test app"
    assert timeout == 3
    assert turn.calls[0].call_id == "openrouter_call_1"
    assert turn.calls[0].actions[0].type == "click"


def test_parses_openrouter_token_usage() -> None:
    backend = OpenRouterBackend(
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
