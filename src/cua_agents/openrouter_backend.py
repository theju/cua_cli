from __future__ import annotations

from dataclasses import dataclass

from .lmstudio_backend import LMStudioBackend


class OpenRouterError(RuntimeError):
    pass


@dataclass
class OpenRouterBackend(LMStudioBackend):
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "openrouter/auto"
    api_key: str | None = None
    http_referer: str | None = None
    app_title: str | None = "cua-agents"
    provider_name: str = "OpenRouter"
    call_id_prefix: str = "openrouter_call"
    error_type: type[RuntimeError] = OpenRouterError

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.app_title:
            headers["X-Title"] = self.app_title
        return headers
