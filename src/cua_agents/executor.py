from __future__ import annotations

import math
import time
from dataclasses import dataclass

from .input_commander import InputCommanderClient
from .models import ComputerAction, Screenshot


class UnsupportedAction(RuntimeError):
    pass


def _number(data: dict, key: str, default: float = 0) -> float:
    value = data.get(key, default)
    if value is None:
        return default
    return float(value)


def _scroll_units(value: float) -> int:
    if value == 0:
        return 0
    magnitude = max(1, round(abs(value) / 120))
    return int(math.copysign(magnitude, value))


def _normalize_key(key: str) -> str:
    key = key.strip()
    aliases = {
        "ARROWDOWN": "down",
        "ARROWLEFT": "left",
        "ARROWRIGHT": "right",
        "ARROWUP": "up",
        "CTRL": "ctrl",
        "CONTROL": "ctrl",
        "CMD": "cmd",
        "COMMAND": "cmd",
        "ESCAPE": "esc",
        "RETURN": "enter",
    }
    compact = key.replace("_", "").replace("-", "").replace(" ", "").upper()
    return aliases.get(compact, key.lower())


def _normalize_keys(raw_keys: object) -> list[str]:
    if isinstance(raw_keys, str):
        parts = raw_keys.replace("+", " ").split()
        return [_normalize_key(part) for part in parts]
    if isinstance(raw_keys, list):
        return [_normalize_key(str(key)) for key in raw_keys]
    raise UnsupportedAction(f"keypress requires keys, got {raw_keys!r}")


@dataclass
class ActionExecutor:
    commander: InputCommanderClient
    post_action_delay: float = 0.2

    def execute(self, action: ComputerAction, screenshot: Screenshot) -> None:
        action_type = action.type
        data = action.data

        if action_type == "move":
            x, y = self._point(data, screenshot)
            self.commander.move(x, y)
        elif action_type == "click":
            self._click(data, screenshot, click_count=1)
        elif action_type == "double_click":
            self._click(data, screenshot, click_count=2)
        elif action_type == "scroll":
            self._scroll(data, screenshot)
        elif action_type == "type":
            text = str(data.get("text", ""))
            self.commander.type_text(text)
        elif action_type == "keypress":
            keys = _normalize_keys(data.get("keys"))
            if len(keys) == 1:
                self.commander.tap_key(keys[0])
            else:
                self.commander.press_combo(keys)
        elif action_type == "wait":
            time.sleep(float(data.get("ms", 1000)) / 1000)
        elif action_type == "screenshot":
            return
        elif action_type == "drag":
            self._drag(data, screenshot)
        else:
            raise UnsupportedAction(f"unsupported computer action: {action_type}")

        if self.post_action_delay:
            time.sleep(self.post_action_delay)

    def _point(self, data: dict, screenshot: Screenshot) -> tuple[int, int]:
        return self.commander.scale_point(
            _number(data, "x"),
            _number(data, "y"),
            screenshot.width,
            screenshot.height,
        )

    def _click(self, data: dict, screenshot: Screenshot, click_count: int) -> None:
        button = str(data.get("button", "left")).lower()
        if button not in {"left", "right"}:
            raise UnsupportedAction(f"input_commander only supports left/right click, got {button!r}")
        if "x" in data and "y" in data:
            x, y = self._point(data, screenshot)
            self.commander.move(x, y)
        for _ in range(click_count):
            self.commander.click(button=button)

    def _scroll(self, data: dict, screenshot: Screenshot) -> None:
        if "x" in data and "y" in data:
            x, y = self._point(data, screenshot)
            self.commander.move(x, y)

        scroll_x = _number(data, "scroll_x", _number(data, "dx", 0))
        scroll_y = _number(data, "scroll_y", _number(data, "dy", 0))
        wheel_x = _scroll_units(scroll_x)
        wheel_y = -_scroll_units(scroll_y)
        self.commander.scroll(wheel_x, wheel_y)

    def _drag(self, data: dict, screenshot: Screenshot) -> None:
        path = data.get("path")
        if path is not None:
            start, end = self._drag_path_endpoints(path)
            start_x, start_y = self.commander.scale_point(
                start[0],
                start[1],
                screenshot.width,
                screenshot.height,
            )
            end_x, end_y = self.commander.scale_point(
                end[0],
                end[1],
                screenshot.width,
                screenshot.height,
            )
        else:
            start_x, start_y = self._point(data, screenshot)
            end_x, end_y = self.commander.scale_point(
                _number(data, "to_x", _number(data, "end_x", _number(data, "x"))),
                _number(data, "to_y", _number(data, "end_y", _number(data, "y"))),
                screenshot.width,
                screenshot.height,
            )

        button = str(data.get("button", "left")).lower()
        if button not in {"left", "right"}:
            raise UnsupportedAction(f"input_commander only supports left/right drag, got {button!r}")

        duration = data.get("duration")
        steps = data.get("steps")
        self.commander.drag(
            start_x,
            start_y,
            end_x,
            end_y,
            button=button,
            duration=float(duration) if duration is not None else None,
            steps=int(steps) if steps is not None else None,
        )

    def _drag_path_endpoints(self, path: object) -> tuple[tuple[float, float], tuple[float, float]]:
        if not isinstance(path, list) or len(path) < 2:
            raise UnsupportedAction("drag path must contain at least two points")
        return self._drag_point(path[0]), self._drag_point(path[-1])

    def _drag_point(self, point: object) -> tuple[float, float]:
        if isinstance(point, dict):
            return _number(point, "x"), _number(point, "y")
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            return float(point[0]), float(point[1])
        raise UnsupportedAction(f"invalid drag path point: {point!r}")
