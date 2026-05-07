from __future__ import annotations

import argparse
import asyncio
import os

from .capture import CaptureConfig, create_capture_backend
from .input_commander import InputCommanderClient
from .executor import ActionExecutor
from .lmstudio_backend import LMStudioBackend
from .openai_backend import OpenAIComputerBackend
from .openrouter_backend import OpenRouterBackend
from .runner import AsyncRunSession, run_terminal_session
from .safety import SafetyPolicy
from .scripted_backend import ScriptedBackend


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cua")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run a computer-use task")
    run.add_argument("task", help="task for the agent to perform")
    run.add_argument(
        "--backend",
        choices=["openai", "openrouter", "lmstudio", "scripted"],
        default="openai",
    )
    run.add_argument("--script", help="scripted backend JSON file")
    run.add_argument(
        "--input-commander-url",
        default=os.getenv("INPUT_COMMANDER_URL", "http://localhost:8080"),
    )
    run.add_argument("--model", default=os.getenv("CUA_MODEL", "gpt-5.5"))
    run.add_argument("--lmstudio-url", default=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"))
    run.add_argument("--lmstudio-model", default=os.getenv("LMSTUDIO_MODEL", "local-model"))
    run.add_argument("--lmstudio-api-key", default=os.getenv("LMSTUDIO_API_KEY"))
    run.add_argument(
        "--lmstudio-timeout",
        type=float,
        default=float(os.getenv("LMSTUDIO_TIMEOUT", "120")),
        help="LM Studio request timeout in seconds",
    )
    run.add_argument(
        "--openrouter-url",
        default=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    run.add_argument("--openrouter-model", default=os.getenv("OPENROUTER_MODEL", "openrouter/auto"))
    run.add_argument("--openrouter-api-key", default=os.getenv("OPENROUTER_API_KEY"))
    run.add_argument("--openrouter-http-referer", default=os.getenv("OPENROUTER_HTTP_REFERER"))
    run.add_argument("--openrouter-app-title", default=os.getenv("OPENROUTER_APP_TITLE", "cua-agents"))
    run.add_argument(
        "--openrouter-timeout",
        type=float,
        default=float(os.getenv("OPENROUTER_TIMEOUT", "120")),
        help="OpenRouter request timeout in seconds",
    )
    run.add_argument("--environment", default=os.getenv("CUA_ENVIRONMENT", "linux"))
    run.add_argument(
        "--capture-backend",
        choices=["auto", "wayland", "spectacle", "x11", "windows", "mac"],
        default=os.getenv("CUA_CAPTURE_BACKEND", "auto"),
        help="screenshot backend",
    )
    run.add_argument("--spectacle-bin", default=os.getenv("CUA_SPECTACLE_BIN", "spectacle"))
    run.add_argument("--x11-capture-bin", default=os.getenv("CUA_X11_CAPTURE_BIN", "import"))
    run.add_argument("--windows-capture-bin", default=os.getenv("CUA_WINDOWS_CAPTURE_BIN", "powershell"))
    run.add_argument("--mac-capture-bin", default=os.getenv("CUA_MAC_CAPTURE_BIN", "screencapture"))
    run.add_argument(
        "--capture-mode",
        choices=["fullscreen", "current", "activewindow"],
        default=os.getenv("CUA_CAPTURE_MODE", "fullscreen"),
        help="capture mode for backends that support it",
    )
    run.add_argument(
        "--capture-delay-ms",
        type=int,
        default=int(os.getenv("CUA_CAPTURE_DELAY_MS", "0")),
        help="screenshot delay in milliseconds for backends that support it",
    )
    run.add_argument(
        "--tool-shape",
        choices=["ga", "ga_minimal", "preview"],
        default=os.getenv("CUA_OPENAI_TOOL_SHAPE", "ga"),
        help="OpenAI computer tool wire shape",
    )
    run.add_argument("--max-steps", type=int, default=int(os.getenv("CUA_MAX_STEPS", "25")))
    run.add_argument("--debug", action="store_true", help="print debug details such as token usage")
    run.add_argument("--step", action="store_true", help="ask before every action")
    run.add_argument("--yes", action="store_true", help="skip local risky-action confirmations")
    run.add_argument(
        "--input-commander-width",
        type=int,
        default=int(os.getenv("INPUT_COMMANDER_WIDTH", "1920")),
        help="absolute X range configured in input_commander",
    )
    run.add_argument(
        "--input-commander-height",
        type=int,
        default=int(os.getenv("INPUT_COMMANDER_HEIGHT", "1080")),
        help="absolute Y range configured in input_commander",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return run_command(args)
    parser.error(f"unknown command {args.command}")
    return 2


def run_command(args: argparse.Namespace) -> int:
    return asyncio.run(run_command_async(args))


async def run_command_async(args: argparse.Namespace) -> int:
    if args.backend == "scripted":
        if not args.script:
            raise SystemExit("--script is required with --backend scripted")
        backend = ScriptedBackend.from_file(args.script)
    elif args.backend == "lmstudio":
        backend = LMStudioBackend(
            base_url=args.lmstudio_url,
            model=args.lmstudio_model,
            api_key=args.lmstudio_api_key,
            timeout=args.lmstudio_timeout,
        )
    elif args.backend == "openrouter":
        backend = OpenRouterBackend(
            base_url=args.openrouter_url,
            model=args.openrouter_model,
            api_key=args.openrouter_api_key,
            http_referer=args.openrouter_http_referer,
            app_title=args.openrouter_app_title,
            timeout=args.openrouter_timeout,
        )
    else:
        backend = OpenAIComputerBackend(
            model=args.model,
            environment=args.environment,
            tool_shape=args.tool_shape,
        )

    if args.step:
        safety_mode = "step"
    elif args.yes:
        safety_mode = "yes"
    else:
        safety_mode = "confirm_risky"

    commander = InputCommanderClient(
        base_url=args.input_commander_url,
        absolute_width=args.input_commander_width,
        absolute_height=args.input_commander_height,
    )
    session = AsyncRunSession(
        task=args.task,
        backend=backend,
        capture=create_capture_backend(
            CaptureConfig(
                backend=args.capture_backend,
                spectacle_bin=args.spectacle_bin,
                x11_bin=args.x11_capture_bin,
                windows_bin=args.windows_capture_bin,
                mac_bin=args.mac_capture_bin,
                mode=args.capture_mode,
                delay_ms=args.capture_delay_ms,
            )
        ),
        executor=ActionExecutor(commander=commander),
        safety=SafetyPolicy(mode=safety_mode),
        max_steps=args.max_steps,
        debug=args.debug,
    )
    result = await run_terminal_session(session)
    print(f"\nstatus: {result.status}")
    return 0 if result.status == "completed" else 1
