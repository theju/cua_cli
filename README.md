# cua-agents

Computer Use Agent harness that captures the local desktop, asks an agent backend for
computer actions, and executes supported actions through the running
`input_commander` server.

The goal of this project is to keep the local computer-control layer small and
inspectable. The harness is responsible for the loop, screenshots, safety
confirmations, coordinate scaling, and HTTP calls to `input_commander`. The agent
backend is swappable: use OpenAI for native computer-use tasks, OpenRouter or
LM Studio for vision-chat models that emit JSON actions, or the scripted backend
for deterministic local testing.

The run loop is asynchronous, event-emitting, and resumable. Multiple sessions
can run at the same time, and sessions pause on approval requests until a caller
approves or rejects the specific approval ID.

## Disclaimer

Contains AI-generated code.

## Requirements

- Python 3.12+
- `input_commander` already running at `http://localhost:8080`
- A screenshot tool for your platform: Spectacle for KDE Wayland, ImageMagick
  `import` for X11, PowerShell for Windows, or `screencapture` for macOS
- `OPENAI_API_KEY` for the OpenAI backend, or `OPENROUTER_API_KEY` for the
  OpenRouter backend

`input_commander` is available at https://github.com/theju/input_commander and
is expected to expose these endpoints:

- `POST /mouse` for `move`, `click`, `scroll`, and `drag`
- `POST /keyboard` for `tap`, `combo`, and `type`

The default mouse coordinate target is `1920x1080`, matching the current
`input_commander` virtual device configuration. If your server is configured
differently, set `INPUT_COMMANDER_WIDTH` and `INPUT_COMMANDER_HEIGHT` or pass the
matching CLI flags.

Screenshots are captured through a pluggable backend. `auto` chooses based on
the platform: X11 on Linux X11 sessions, Spectacle-backed Wayland on other Linux
sessions, PowerShell on Windows, and `screencapture` on macOS.

## Installation

This project uses `uv` for dependency management during development:

```bash
uv sync
```

You can then run the CLI through `uv`:

```bash
uv run cua --help
```

If you install the package into another Python environment, the console command
is still `cua`.

## Quick start

Run a task with the OpenAI backend:

```bash
uv run cua run "open a terminal and type hello"
```

Run with approval before every computer action:

```bash
uv run cua run --step "inspect the current screen"
```

Run with local risky-action confirmations disabled:

```bash
uv run cua run --yes --max-steps 10 "close the active window"
```

Run with debug metadata, including token usage when the provider returns it:

```bash
uv run cua run --debug "inspect the current screen"
```

Run the deterministic scripted backend:

```bash
uv run cua run --backend scripted --script tests/fixtures/simple_actions.json "ignored"
```

Run against a local LM Studio model:

```bash
uv run cua run --backend lmstudio --lmstudio-model local-vision-model "inspect the current screen"
```

Run against an OpenRouter vision-capable chat model:

```bash
OPENROUTER_API_KEY=... uv run cua run \
  --backend openrouter \
  --openrouter-model provider/model-name \
  "inspect the current screen"
```

## Python library usage

The harness can be used directly from Python. The CLI is only a thin wrapper
around the same async session classes.

OpenAI example:

```python
import asyncio

from cua_agents.capture import CaptureConfig, create_capture_backend
from cua_agents.input_commander import InputCommanderClient
from cua_agents.executor import ActionExecutor
from cua_agents.openai_backend import OpenAIComputerBackend
from cua_agents.runner import AsyncRunSession
from cua_agents.safety import SafetyPolicy


async def main():
    backend = OpenAIComputerBackend(model="gpt-5.5")
    commander = InputCommanderClient(
        base_url="http://localhost:8080",
        absolute_width=1920,
        absolute_height=1080,
    )
    session = AsyncRunSession(
        task="inspect the current screen",
        backend=backend,
        capture=create_capture_backend(CaptureConfig(backend="auto")),
        executor=ActionExecutor(commander=commander),
        safety=SafetyPolicy(mode="confirm_risky"),
        max_steps=25,
        debug=True,
    )

    result = await session.run()
    print(result.status, result.final_text)


asyncio.run(main())
```

LM Studio example:

```python
import asyncio

from cua_agents.capture import CaptureConfig, create_capture_backend
from cua_agents.input_commander import InputCommanderClient
from cua_agents.executor import ActionExecutor
from cua_agents.lmstudio_backend import LMStudioBackend
from cua_agents.runner import AsyncRunSession
from cua_agents.safety import SafetyPolicy


async def main():
    backend = LMStudioBackend(
        base_url="http://localhost:1234/v1",
        model="local-vision-model",
    )
    session = AsyncRunSession(
        task="summarize what is visible on screen",
        backend=backend,
        capture=create_capture_backend(CaptureConfig(backend="wayland")),
        executor=ActionExecutor(
            commander=InputCommanderClient(base_url="http://localhost:8080")
        ),
        safety=SafetyPolicy(mode="step"),
        max_steps=10,
        debug=True,
    )

    result = await session.run()
    print(result.status)


asyncio.run(main())
```

Scripted backend example for tests or deterministic demos:

```python
from cua_agents.scripted_backend import ScriptedBackend

backend = ScriptedBackend.from_file("tests/fixtures/simple_actions.json")
```

For web or GUI usage, consume session events while the session runs:

```python
run_task = asyncio.create_task(session.run())

async for event in session.events():
    if event.type == "approval_required":
        # Show event.payload["message"] in your UI, then resume the session.
        await session.approve(event.payload["approval_id"])
    elif event.type == "proposed_action":
        print(event.payload["action"]["label"])

result = await run_task
```

The main reusable classes are:

- `AsyncRunSession`: owns one async, resumable computer-use loop.
- `SessionManager`: creates sessions and routes approval/stop requests.
- `create_capture_backend`: creates a platform-specific screenshot backend.
- `InputCommanderClient`: sends mouse and keyboard requests to `input_commander`.
- `ActionExecutor`: maps normalized actions to commander calls.
- `SafetyPolicy`: decides which local actions need confirmation.
- `OpenAIComputerBackend`, `LMStudioBackend`, and `ScriptedBackend`: agent
  backends.

## How the loop works

Each async session follows a standard computer-use loop:

1. Capture the current screen as a PNG with the configured capture backend.
2. Send the user task and screenshot to the selected backend.
3. Read the backend's proposed computer actions.
4. Confirm safety checks or risky local actions when required.
5. Execute supported actions through `input_commander`.
6. Capture a fresh screenshot and send it back to the backend.
7. Repeat until the backend returns a final message or `--max-steps` is reached.

Sessions emit structured `RunEvent` objects for screenshots, turns, token usage,
proposed actions, approvals, executed actions, completion, failure, and stops.
When an approval is needed, the session emits `approval_required` with an
`approval_id` and waits until `approve()` or `reject()` is called.

The OpenAI backend uses the Responses API computer tool by default. It supports
the current batched `actions[]` response shape and also accepts the older
single-`action` shape so the harness remains compatible with preview-style
responses.

## Supported actions

The harness currently maps these computer actions:

- `move`: move the mouse to a screen coordinate.
- `click`: move if coordinates are present, then left-click.
- `double_click`: move if coordinates are present, then click twice.
- `scroll`: move if coordinates are present, then send wheel units.
- `type`: type text through the keyboard endpoint.
- `keypress`: tap a single key or send a key combo.
- `wait`: sleep without touching the UI.
- `screenshot`: no-op locally; the runner captures after each model turn.
- `drag`: drag from a start point to an end point through `input_commander`.

Coordinates from screenshots are scaled into the absolute coordinate range used
by `input_commander`. By default, this scales from the captured display size into
`1920x1080`.

## Drag and drop

The harness maps `drag` actions to the `input_commander` `/mouse` `drag`
endpoint. It supports two input shapes:

OpenAI-style path actions:

```json
{
  "type": "drag",
  "path": [
    {"x": 200, "y": 200},
    {"x": 800, "y": 600}
  ]
}
```

Explicit scripted actions:

```json
{
  "type": "drag",
  "x": 200,
  "y": 200,
  "to_x": 800,
  "to_y": 600,
  "button": "left",
  "duration": 0.25,
  "steps": 20
}
```

For path-style actions, the harness uses the first and last points as the drag
start and end. The server performs interpolation.

## Backends

### OpenAI backend

The OpenAI backend is selected by default:

```bash
uv run cua run "summarize what is visible on screen"
```

Useful options:

```bash
uv run cua run --model gpt-5.5 "task"
uv run cua run --environment linux "task"
uv run cua run --tool-shape ga "task"
uv run cua run --debug "task"
```

`--tool-shape` accepts:

- `ga`: send the current `computer` tool payload: `{"type": "computer"}`.
- `ga_minimal`: accepted as an alias for `ga` for older local configs.
- `preview`: send the older `computer_use_preview` tool shape.

### Scripted backend

The scripted backend reads a JSON file containing turns and actions. It is useful
for testing the executor without calling a model:

```bash
uv run cua run --backend scripted --script tests/fixtures/simple_actions.json "ignored"
```

Script files use this shape:

```json
{
  "turns": [
    {
      "response_id": "script_1",
      "summaries": ["Move to the center and wait."],
      "calls": [
        {
          "call_id": "script_call_1",
          "actions": [
            {"type": "move", "x": 50, "y": 50},
            {"type": "wait", "ms": 100}
          ]
        }
      ]
    },
    {
      "response_id": "script_2",
      "final_text": "Script completed."
    }
  ]
}
```

### LM Studio backend

The LM Studio backend talks to LM Studio's local OpenAI-compatible chat
completions server. It is useful when you want to run a local vision-capable
model instead of an OpenAI-hosted computer-use model.

Start LM Studio, load a vision-capable model, enable the local server, then run:

```bash
uv run cua run \
  --backend lmstudio \
  --lmstudio-url http://localhost:1234/v1 \
  --lmstudio-model local-vision-model \
  "summarize what is visible on screen"
```

The default LM Studio settings are:

- `LMSTUDIO_BASE_URL=http://localhost:1234/v1`
- `LMSTUDIO_MODEL=local-model`
- `LMSTUDIO_TIMEOUT=120`
- `LMSTUDIO_API_KEY` unset

If your LM Studio server requires a bearer token, set `LMSTUDIO_API_KEY` or pass
`--lmstudio-api-key`.

The LM Studio backend sends each screenshot as a data URL image in the chat
messages and prompts the model to return JSON actions. It does not use a native
computer-use API, so quality depends heavily on the model's vision ability and
instruction following.

Use `--debug` to print token usage if LM Studio includes an OpenAI-compatible
`usage` object in its response.

Expected model output:

```json
{
  "calls": [
    {
      "call_id": "call_1",
      "actions": [
        {"type": "click", "x": 400, "y": 300, "button": "left"}
      ]
    }
  ],
  "final_text": null,
  "summaries": ["Click the visible target."]
}
```

When the task is complete, the model should return:

```json
{
  "calls": [],
  "final_text": "Done.",
  "summaries": []
}
```

If the local model returns plain text instead of JSON, the harness treats that
text as a final message and stops. This is intentionally conservative: the
harness will not infer desktop actions from ambiguous prose.

### OpenRouter backend

The OpenRouter backend talks to OpenRouter's OpenAI-compatible chat completions
API. It is useful for trying hosted vision-capable models that do not expose a
native computer-use API.

Set an API key, choose a model that accepts image input, then run:

```bash
OPENROUTER_API_KEY=... uv run cua run \
  --backend openrouter \
  --openrouter-model provider/model-name \
  "summarize what is visible on screen"
```

The default OpenRouter settings are:

- `OPENROUTER_BASE_URL=https://openrouter.ai/api/v1`
- `OPENROUTER_MODEL=openrouter/auto`
- `OPENROUTER_TIMEOUT=120`
- `OPENROUTER_API_KEY` unset
- `OPENROUTER_HTTP_REFERER` unset
- `OPENROUTER_APP_TITLE=cua-agents`

The OpenRouter backend uses the same screenshot and JSON action protocol as the
LM Studio backend. Quality depends on the selected model's vision ability,
instruction following, and ability to produce the requested JSON without extra
prose.

### Other model providers

The harness is provider-neutral below the backend layer. OpenAI is the only live
native computer-use provider implemented today. OpenRouter and LM Studio are
implemented for OpenAI-compatible chat models that emit JSON actions. Another
provider can be added without changing screenshot capture, safety confirmations,
coordinate scaling, or `input_commander` execution.

To add another model provider:

1. Create a backend class that implements `AgentBackend` from
   `src/cua_agents/backend.py`.
2. Convert the provider's response format into the harness models from
   `src/cua_agents/models.py`.
3. Add the backend choice to `src/cua_agents/cli.py`.
4. Add tests with mocked provider responses before using it against the real
   desktop.

The backend interface is:

```python
class AgentBackend(Protocol):
    def start(self, task: str, screenshot: Screenshot) -> AgentTurn:
        ...

    def continue_after_call(
        self,
        call_id: str,
        screenshot: Screenshot,
        acknowledged_safety_checks: list[dict] | None = None,
    ) -> AgentTurn:
        ...
```

The backend should return an `AgentTurn` containing zero or more `ComputerCall`
objects. Each `ComputerCall` contains normalized `ComputerAction` objects such as
`click`, `type`, `keypress`, `scroll`, `drag`, `wait`, or `screenshot`.

For providers that do not have a built-in computer-use API, use a structured
output prompt that asks the model to emit this action schema:

```json
{
  "calls": [
    {
      "call_id": "provider_call_1",
      "actions": [
        {"type": "click", "x": 400, "y": 300, "button": "left"},
        {"type": "type", "text": "hello"}
      ]
    }
  ],
  "final_text": null
}
```

When no more actions are needed, return an `AgentTurn` with no calls and a
`final_text` message.

Important provider requirements:

- The model needs vision input, because each turn is driven by a screenshot.
- Action coordinates must be in screenshot pixel coordinates; the executor
  handles scaling into the `input_commander` coordinate range.
- Preserve a provider conversation id or message history inside the backend so
  `continue_after_call()` can send the next screenshot in context.
- If the provider has its own safety checks, map them into
  `pending_safety_checks` so the runner can pause for user confirmation.
- Keep provider-specific action names out of the executor; normalize them in the
  backend.

The scripted backend is the easiest reference implementation for the normalized
schema. The OpenAI backend is the reference implementation for a native
computer-use provider. The LM Studio and OpenRouter backends are reference
implementations for generic vision-chat providers that emit JSON actions.

## Configuration

Configuration can be supplied by environment variables or CLI flags.

Defaults:

- `INPUT_COMMANDER_URL=http://localhost:8080`
- `CUA_MODEL=gpt-5.5`
- `CUA_MAX_STEPS=25`
- `CUA_ENVIRONMENT=linux`
- `CUA_OPENAI_TOOL_SHAPE=ga`
- `INPUT_COMMANDER_WIDTH=1920`
- `INPUT_COMMANDER_HEIGHT=1080`
- `LMSTUDIO_BASE_URL=http://localhost:1234/v1`
- `LMSTUDIO_MODEL=local-model`
- `LMSTUDIO_TIMEOUT=120`
- `OPENROUTER_BASE_URL=https://openrouter.ai/api/v1`
- `OPENROUTER_MODEL=openrouter/auto`
- `OPENROUTER_TIMEOUT=120`
- `OPENROUTER_APP_TITLE=cua-agents`

CLI flags override environment variables:

```bash
uv run cua run \
  --input-commander-url http://localhost:8080 \
  --capture-backend auto \
  --capture-mode fullscreen \
  --input-commander-width 1920 \
  --input-commander-height 1080 \
  --max-steps 25 \
  "task"
```

Debug output:

- `--debug` prints provider metadata that is useful while tuning prompts and
  local models.
- When available in the model response, token usage is printed as
  `input=<n>, cached=<n>, output=<n>`.
- Cached token counts are omitted when the provider does not report them.

Screenshot-related defaults:

- `CUA_CAPTURE_BACKEND=auto`
- `CUA_SPECTACLE_BIN=spectacle`
- `CUA_X11_CAPTURE_BIN=import`
- `CUA_WINDOWS_CAPTURE_BIN=powershell`
- `CUA_MAC_CAPTURE_BIN=screencapture`
- `CUA_CAPTURE_MODE=fullscreen`
- `CUA_CAPTURE_DELAY_MS=0`

`CUA_CAPTURE_BACKEND` accepts:

- `auto`: select a backend from the current platform/session.
- `wayland`: use KDE Spectacle, intended for KDE Plasma Wayland.
- `spectacle`: alias for `wayland`.
- `x11`: use ImageMagick `import -window root`.
- `windows`: use PowerShell with .NET screen capture APIs.
- `mac`: use macOS `screencapture`.

`CUA_CAPTURE_MODE` accepts:

- `fullscreen`: capture the entire desktop.
- `current`: capture the current monitor.
- `activewindow`: capture the active window.

`CUA_CAPTURE_MODE` is currently used by the Spectacle/Wayland backend. X11,
Windows, and macOS backends capture the primary/full desktop area in this
version.

## Safety

The default mode confirms OpenAI safety checks and locally risky actions before
executing them. Use `--step` to approve every action. Use `--yes` only for
low-risk unattended runs; safety checks still require confirmation.

Safety modes:

- Default: confirm locally risky actions such as sensitive typing or submit-like
  clicks.
- `--step`: confirm every proposed action.
- `--yes`: skip local risky-action confirmations for low-risk unattended runs.

OpenAI safety checks are always shown and require confirmation before the harness
acknowledges them and continues.

Because this harness controls the active desktop, prefer running it in a VM,
containerized desktop, or disposable test session for risky workflows.

## Development

Run tests:

```bash
uv run pytest
```

Run a compile check:

```bash
uv run python -m compileall src tests
```

The tests mock the HTTP boundary and do not move the real mouse or type on the
real keyboard.

## Project layout

- `src/cua_agents/cli.py`: command-line interface.
- `src/cua_agents/runner.py`: computer-use loop and safety confirmations.
- `src/cua_agents/openai_backend.py`: OpenAI Responses API integration.
- `src/cua_agents/openrouter_backend.py`: OpenRouter chat-completions integration.
- `src/cua_agents/lmstudio_backend.py`: LM Studio chat-completions integration.
- `src/cua_agents/scripted_backend.py`: deterministic backend for tests.
- `src/cua_agents/input_commander.py`: HTTP client for `input_commander`.
- `src/cua_agents/executor.py`: action mapping and coordinate scaling.
- `src/cua_agents/capture.py`: local screenshot capture.
- `tests/`: unit tests and scripted fixtures.

## Troubleshooting

If the CLI cannot connect to `input_commander`, verify the server is running and
that `INPUT_COMMANDER_URL` points to the right host and port.

If actions land in the wrong place, confirm the virtual mouse absolute range in
`input_commander` and update `INPUT_COMMANDER_WIDTH` / `INPUT_COMMANDER_HEIGHT`.

If the OpenAI backend fails before any action is proposed, confirm
`OPENAI_API_KEY` is set in the environment where `uv run cua ...` is executed.

If screenshot capture fails, first check which backend is active:

```bash
uv run cua run --capture-backend auto --debug "inspect the current screen"
```

Then test the platform command from the same desktop session:

- Wayland/KDE: `spectacle --background --nonotify --fullscreen --output /tmp/cua-test.png`
- X11: `import -window root /tmp/cua-test.png`
- Windows: verify `powershell` is available.
- macOS: `screencapture -x /tmp/cua-test.png`

If a tool is installed under a different name or path, set the matching
`CUA_*_CAPTURE_BIN` variable or CLI flag.

If `uv run` fails due to cache permissions in a sandboxed environment, rerun with
appropriate permissions or pre-create the virtual environment outside the
sandbox.
