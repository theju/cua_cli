from __future__ import annotations

import os
from dataclasses import dataclass

from .capture import CaptureConfig, create_capture_backend
from .executor import ActionExecutor
from .input_commander import InputCommanderClient
from .lmstudio_backend import LMStudioBackend
from .openai_backend import OpenAIComputerBackend
from .openrouter_backend import OpenRouterBackend
from .runner import AsyncRunSession
from .safety import SafetyPolicy
from .scripted_backend import ScriptedBackend


@dataclass(frozen=True)
class RunConfig:
    task: str
    backend: str = "openai"
    script: str | None = None
    input_commander_url: str = "http://localhost:8080"
    input_commander_width: int = 1920
    input_commander_height: int = 1080
    model: str = "gpt-5.5"
    lmstudio_url: str = "http://localhost:1234/v1"
    lmstudio_model: str = "local-model"
    lmstudio_api_key: str | None = None
    lmstudio_timeout: float = 120.0
    openrouter_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "openrouter/auto"
    openrouter_api_key: str | None = None
    openrouter_http_referer: str | None = None
    openrouter_app_title: str | None = "cua-agents"
    openrouter_timeout: float = 120.0
    environment: str = "linux"
    capture_backend: str = "auto"
    spectacle_bin: str = "spectacle"
    x11_capture_bin: str = "import"
    windows_capture_bin: str = "powershell"
    mac_capture_bin: str = "screencapture"
    capture_mode: str = "fullscreen"
    capture_delay_ms: int = 0
    tool_shape: str = "ga"
    max_steps: int = 25
    debug: bool = False
    safety_mode: str = "confirm_risky"


def default_run_config(task: str) -> RunConfig:
    return RunConfig(
        task=task,
        input_commander_url=os.getenv("INPUT_COMMANDER_URL", "http://localhost:8080"),
        input_commander_width=_int_env("INPUT_COMMANDER_WIDTH", 1920),
        input_commander_height=_int_env("INPUT_COMMANDER_HEIGHT", 1080),
        model=os.getenv("CUA_MODEL", "gpt-5.5"),
        lmstudio_url=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
        lmstudio_model=os.getenv("LMSTUDIO_MODEL", "local-model"),
        lmstudio_api_key=os.getenv("LMSTUDIO_API_KEY"),
        lmstudio_timeout=_float_env("LMSTUDIO_TIMEOUT", 120.0),
        openrouter_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "openrouter/auto"),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
        openrouter_http_referer=os.getenv("OPENROUTER_HTTP_REFERER"),
        openrouter_app_title=os.getenv("OPENROUTER_APP_TITLE", "cua-agents"),
        openrouter_timeout=_float_env("OPENROUTER_TIMEOUT", 120.0),
        environment=os.getenv("CUA_ENVIRONMENT", "linux"),
        capture_backend=os.getenv("CUA_CAPTURE_BACKEND", "auto"),
        spectacle_bin=os.getenv("CUA_SPECTACLE_BIN", "spectacle"),
        x11_capture_bin=os.getenv("CUA_X11_CAPTURE_BIN", "import"),
        windows_capture_bin=os.getenv("CUA_WINDOWS_CAPTURE_BIN", "powershell"),
        mac_capture_bin=os.getenv("CUA_MAC_CAPTURE_BIN", "screencapture"),
        capture_mode=os.getenv("CUA_CAPTURE_MODE", "fullscreen"),
        capture_delay_ms=_int_env("CUA_CAPTURE_DELAY_MS", 0),
        tool_shape=os.getenv("CUA_OPENAI_TOOL_SHAPE", "ga"),
        max_steps=_int_env("CUA_MAX_STEPS", 25),
    )


def build_session(config: RunConfig) -> AsyncRunSession:
    backend = _build_backend(config)
    commander = InputCommanderClient(
        base_url=config.input_commander_url,
        absolute_width=config.input_commander_width,
        absolute_height=config.input_commander_height,
    )
    return AsyncRunSession(
        task=config.task,
        backend=backend,
        capture=create_capture_backend(
            CaptureConfig(
                backend=config.capture_backend,
                spectacle_bin=config.spectacle_bin,
                x11_bin=config.x11_capture_bin,
                windows_bin=config.windows_capture_bin,
                mac_bin=config.mac_capture_bin,
                mode=config.capture_mode,
                delay_ms=config.capture_delay_ms,
            )
        ),
        executor=ActionExecutor(commander=commander),
        safety=SafetyPolicy(mode=config.safety_mode),
        max_steps=config.max_steps,
        debug=config.debug,
    )


def _build_backend(config: RunConfig):
    if config.backend == "scripted":
        if not config.script:
            raise ValueError("script is required with scripted backend")
        return ScriptedBackend.from_file(config.script)
    if config.backend == "lmstudio":
        return LMStudioBackend(
            base_url=config.lmstudio_url,
            model=config.lmstudio_model,
            api_key=config.lmstudio_api_key,
            timeout=config.lmstudio_timeout,
        )
    if config.backend == "openrouter":
        return OpenRouterBackend(
            base_url=config.openrouter_url,
            model=config.openrouter_model,
            api_key=config.openrouter_api_key,
            http_referer=config.openrouter_http_referer,
            app_title=config.openrouter_app_title,
            timeout=config.openrouter_timeout,
        )
    if config.backend == "openai":
        return OpenAIComputerBackend(
            model=config.model,
            environment=config.environment,
            tool_shape=config.tool_shape,
        )
    raise ValueError(f"unsupported backend: {config.backend}")


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)
