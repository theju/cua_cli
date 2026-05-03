from __future__ import annotations

import struct

import pytest

from cua_agents.capture import (
    CaptureConfig,
    MacCapture,
    ScreenCapture,
    ScreenCaptureError,
    SpectacleCapture,
    WaylandCapture,
    WindowsCapture,
    X11Capture,
    create_capture_backend,
    detect_capture_backend,
    png_dimensions,
)


def minimal_png(width: int, height: int) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height)


def test_png_dimensions_reads_ihdr_size() -> None:
    assert png_dimensions(minimal_png(123, 45)) == (123, 45)


def test_png_dimensions_rejects_non_png() -> None:
    with pytest.raises(ScreenCaptureError):
        png_dimensions(b"not a png")


def test_spectacle_mode_args() -> None:
    assert SpectacleCapture(mode="fullscreen")._mode_args() == ["--fullscreen"]
    assert SpectacleCapture(mode="current")._mode_args() == ["--current"]
    assert SpectacleCapture(mode="activewindow")._mode_args() == ["--activewindow"]


def test_spectacle_mode_args_rejects_unknown_mode() -> None:
    with pytest.raises(ScreenCaptureError):
        SpectacleCapture(mode="region")._mode_args()


def test_screen_capture_alias_preserves_spectacle_behavior() -> None:
    assert ScreenCapture is WaylandCapture


def test_create_capture_backend_selects_explicit_backends() -> None:
    assert isinstance(create_capture_backend(CaptureConfig(backend="wayland")), WaylandCapture)
    assert isinstance(create_capture_backend(CaptureConfig(backend="spectacle")), WaylandCapture)
    assert isinstance(create_capture_backend(CaptureConfig(backend="x11")), X11Capture)
    assert isinstance(create_capture_backend(CaptureConfig(backend="windows")), WindowsCapture)
    assert isinstance(create_capture_backend(CaptureConfig(backend="mac")), MacCapture)


def test_create_capture_backend_rejects_unknown_backend() -> None:
    with pytest.raises(ScreenCaptureError):
        create_capture_backend(CaptureConfig(backend="unknown"))


def test_detect_capture_backend_linux_x11(monkeypatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setenv("XDG_SESSION_TYPE", "x11")

    assert detect_capture_backend() == "x11"


def test_detect_capture_backend_linux_defaults_to_wayland(monkeypatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.delenv("XDG_SESSION_TYPE", raising=False)

    assert detect_capture_backend() == "wayland"


def test_detect_capture_backend_windows(monkeypatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Windows")

    assert detect_capture_backend() == "windows"


def test_detect_capture_backend_mac(monkeypatch) -> None:
    monkeypatch.setattr("platform.system", lambda: "Darwin")

    assert detect_capture_backend() == "mac"
