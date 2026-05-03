from __future__ import annotations

from cua_agents.input_commander import InputCommanderClient
from cua_agents.executor import ActionExecutor
from cua_agents.models import ComputerAction, Screenshot


class RecordingCommander(InputCommanderClient):
    def __init__(self) -> None:
        super().__init__(absolute_width=1920, absolute_height=1080)
        self.calls: list[tuple[str, dict]] = []

    def _post(self, path: str, payload: dict) -> dict:
        self.calls.append((path, payload))
        return {"ok": True}


def screenshot(width: int = 960, height: int = 540) -> Screenshot:
    return Screenshot(png=b"png", width=width, height=height)


def test_scale_point_maps_screenshot_to_commander_range() -> None:
    client = InputCommanderClient(absolute_width=1920, absolute_height=1080)

    assert client.scale_point(480, 270, 960, 540) == (960, 540)
    assert client.scale_point(9999, -10, 960, 540) == (1920, 0)


def test_click_moves_before_left_click() -> None:
    commander = RecordingCommander()
    executor = ActionExecutor(commander=commander, post_action_delay=0)

    executor.execute(
        ComputerAction(type="click", data={"x": 480, "y": 270, "button": "left"}),
        screenshot(),
    )

    assert commander.calls == [
        ("mouse", {"action": "move", "x": 960, "y": 540}),
        ("mouse", {"action": "click", "button": "left"}),
    ]


def test_right_click_moves_before_clicking_right_button() -> None:
    commander = RecordingCommander()
    executor = ActionExecutor(commander=commander, post_action_delay=0)

    executor.execute(
        ComputerAction(type="click", data={"x": 480, "y": 270, "button": "right"}),
        screenshot(),
    )

    assert commander.calls == [
        ("mouse", {"action": "move", "x": 960, "y": 540}),
        ("mouse", {"action": "click", "button": "right"}),
    ]


def test_right_double_click_uses_right_button_twice() -> None:
    commander = RecordingCommander()
    executor = ActionExecutor(commander=commander, post_action_delay=0)

    executor.execute(
        ComputerAction(type="double_click", data={"x": 480, "y": 270, "button": "right"}),
        screenshot(),
    )

    assert commander.calls == [
        ("mouse", {"action": "move", "x": 960, "y": 540}),
        ("mouse", {"action": "click", "button": "right"}),
        ("mouse", {"action": "click", "button": "right"}),
    ]


def test_keypress_combo_uses_keyboard_combo_endpoint() -> None:
    commander = RecordingCommander()
    executor = ActionExecutor(commander=commander, post_action_delay=0)

    executor.execute(
        ComputerAction(type="keypress", data={"keys": ["CTRL", "L"]}),
        screenshot(),
    )

    assert commander.calls == [
        ("keyboard", {"action": "combo", "keys": ["ctrl", "l"]}),
    ]


def test_scroll_converts_pixel_scroll_to_wheel_units() -> None:
    commander = RecordingCommander()
    executor = ActionExecutor(commander=commander, post_action_delay=0)

    executor.execute(
        ComputerAction(type="scroll", data={"x": 10, "y": 20, "scroll_y": 240}),
        screenshot(),
    )

    assert commander.calls == [
        ("mouse", {"action": "move", "x": 20, "y": 40}),
        ("mouse", {"action": "scroll", "x": 0, "y": -2}),
    ]


def test_drag_path_maps_to_mouse_drag_endpoint() -> None:
    commander = RecordingCommander()
    executor = ActionExecutor(commander=commander, post_action_delay=0)

    executor.execute(
        ComputerAction(
            type="drag",
            data={
                "path": [
                    {"x": 100, "y": 50},
                    {"x": 300, "y": 250},
                ],
                "duration": 0.4,
                "steps": 8,
            },
        ),
        screenshot(),
    )

    assert commander.calls == [
        (
            "mouse",
            {
                "action": "drag",
                "x": 200,
                "y": 100,
                "to_x": 600,
                "to_y": 500,
                "button": "left",
                "duration": 0.4,
                "steps": 8,
            },
        )
    ]


def test_drag_explicit_endpoints_maps_to_mouse_drag_endpoint() -> None:
    commander = RecordingCommander()
    executor = ActionExecutor(commander=commander, post_action_delay=0)

    executor.execute(
        ComputerAction(
            type="drag",
            data={"x": 100, "y": 50, "to_x": 300, "to_y": 250, "button": "right"},
        ),
        screenshot(),
    )

    assert commander.calls == [
        (
            "mouse",
            {
                "action": "drag",
                "x": 200,
                "y": 100,
                "to_x": 600,
                "to_y": 500,
                "button": "right",
            },
        )
    ]
