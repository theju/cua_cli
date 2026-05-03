from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class InputCommanderError(RuntimeError):
    pass


@dataclass(frozen=True)
class InputCommanderClient:
    base_url: str = "http://localhost:8080"
    timeout: float = 5.0
    absolute_width: int = 1920
    absolute_height: int = 1080

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        body = json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise InputCommanderError(f"{path} failed with HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise InputCommanderError(f"{path} failed: {exc.reason}") from exc

        if not data:
            return {}
        try:
            return json.loads(data)
        except json.JSONDecodeError as exc:
            raise InputCommanderError(f"{path} returned invalid JSON: {data!r}") from exc

    def scale_point(self, x: float, y: float, source_width: int, source_height: int) -> tuple[int, int]:
        if source_width <= 0 or source_height <= 0:
            raise InputCommanderError("source screenshot dimensions must be positive")
        scaled_x = round((float(x) / source_width) * self.absolute_width)
        scaled_y = round((float(y) / source_height) * self.absolute_height)
        return (
            max(0, min(self.absolute_width, scaled_x)),
            max(0, min(self.absolute_height, scaled_y)),
        )

    def move(self, x: int, y: int) -> None:
        self._post("mouse", {"action": "move", "x": x, "y": y})

    def click(self, button: str = "left") -> None:
        self._post("mouse", {"action": "click", "button": button})

    def scroll(self, x: int, y: int) -> None:
        self._post("mouse", {"action": "scroll", "x": x, "y": y})

    def drag(
        self,
        x: int,
        y: int,
        to_x: int,
        to_y: int,
        *,
        button: str = "left",
        duration: float | None = None,
        steps: int | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "action": "drag",
            "x": x,
            "y": y,
            "to_x": to_x,
            "to_y": to_y,
            "button": button,
        }
        if duration is not None:
            payload["duration"] = duration
        if steps is not None:
            payload["steps"] = steps
        self._post("mouse", payload)

    def tap_key(self, key: str) -> None:
        self._post("keyboard", {"action": "tap", "key": key})

    def press_combo(self, keys: list[str]) -> None:
        self._post("keyboard", {"action": "combo", "keys": keys})

    def type_text(self, text: str) -> None:
        self._post("keyboard", {"action": "type", "text": text})
