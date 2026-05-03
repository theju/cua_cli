from __future__ import annotations

import os
import platform
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .models import Screenshot


class ScreenCaptureError(RuntimeError):
    pass


class CaptureBackend(Protocol):
    def capture(self) -> Screenshot:
        """Capture the current screen as a PNG screenshot."""


def png_dimensions(png: bytes) -> tuple[int, int]:
    if len(png) < 24 or png[:8] != b"\x89PNG\r\n\x1a\n" or png[12:16] != b"IHDR":
        raise ScreenCaptureError("Screenshot output is not a valid PNG image")
    width, height = struct.unpack(">II", png[16:24])
    if width <= 0 or height <= 0:
        raise ScreenCaptureError("Screenshot output has invalid PNG dimensions")
    return width, height


def _run_png_command(command: list[str], *, missing_message: str) -> Screenshot:
    with tempfile.TemporaryDirectory(prefix="cua-screenshot-") as directory:
        output = Path(directory) / "screen.png"
        try:
            completed = subprocess.run(
                [part if part != "{output}" else str(output) for part in command],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except FileNotFoundError as exc:
            raise ScreenCaptureError(missing_message) from exc

        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            detail = stderr or stdout or f"exit code {completed.returncode}"
            raise ScreenCaptureError(f"Screenshot command failed: {detail}")
        if not output.exists():
            raise ScreenCaptureError("Screenshot command did not create the requested file")

        png = output.read_bytes()
        width, height = png_dimensions(png)
        return Screenshot(png=png, width=width, height=height)


@dataclass(frozen=True)
class SpectacleCapture:
    spectacle: str = "spectacle"
    mode: str = "fullscreen"
    delay_ms: int = 0

    def capture(self) -> Screenshot:
        return _run_png_command(
            [
                self.spectacle,
                "--background",
                "--nonotify",
                *self._mode_args(),
                "--delay",
                str(self.delay_ms),
                "--output",
                "{output}",
            ],
            missing_message="Spectacle was not found. Install KDE Spectacle or set CUA_SPECTACLE_BIN.",
        )

    def _mode_args(self) -> list[str]:
        if self.mode == "fullscreen":
            return ["--fullscreen"]
        if self.mode == "current":
            return ["--current"]
        if self.mode == "activewindow":
            return ["--activewindow"]
        raise ScreenCaptureError(
            f"Unsupported Spectacle capture mode {self.mode!r}; use fullscreen, current, or activewindow"
        )


@dataclass(frozen=True)
class WaylandCapture:
    """Wayland capture backend.

    Uses KDE Spectacle because it works through compositor-supported screenshot
    paths on KDE Plasma Wayland and is already the project default.
    """

    spectacle: str = "spectacle"
    mode: str = "fullscreen"
    delay_ms: int = 0

    def capture(self) -> Screenshot:
        return SpectacleCapture(
            spectacle=self.spectacle,
            mode=self.mode,
            delay_ms=self.delay_ms,
        ).capture()


@dataclass(frozen=True)
class X11Capture:
    import_bin: str = "import"
    root: bool = True

    def capture(self) -> Screenshot:
        command = [self.import_bin]
        if self.root:
            command.append("-window")
            command.append("root")
        command.append("{output}")
        return _run_png_command(
            command,
            missing_message=(
                "ImageMagick import was not found. Install ImageMagick or set CUA_X11_CAPTURE_BIN."
            ),
        )


@dataclass(frozen=True)
class WindowsCapture:
    powershell: str = "powershell"

    def capture(self) -> Screenshot:
        script = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "Add-Type -AssemblyName System.Drawing;"
            "$b=[System.Windows.Forms.Screen]::PrimaryScreen.Bounds;"
            "$bmp=New-Object System.Drawing.Bitmap $b.Width,$b.Height;"
            "$g=[System.Drawing.Graphics]::FromImage($bmp);"
            "$g.CopyFromScreen($b.Location,[System.Drawing.Point]::Empty,$b.Size);"
            "$bmp.Save('{output}',[System.Drawing.Imaging.ImageFormat]::Png);"
            "$g.Dispose();$bmp.Dispose();"
        )
        return _run_png_command(
            [
                self.powershell,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                script,
            ],
            missing_message="PowerShell was not found. Set CUA_WINDOWS_CAPTURE_BIN to a usable PowerShell path.",
        )


@dataclass(frozen=True)
class MacCapture:
    screencapture: str = "screencapture"

    def capture(self) -> Screenshot:
        return _run_png_command(
            [self.screencapture, "-x", "{output}"],
            missing_message="screencapture was not found. Set CUA_MAC_CAPTURE_BIN to a usable path.",
        )


@dataclass(frozen=True)
class CaptureConfig:
    backend: str = "auto"
    mode: str = "fullscreen"
    delay_ms: int = 0
    spectacle_bin: str = "spectacle"
    x11_bin: str = "import"
    windows_bin: str = "powershell"
    mac_bin: str = "screencapture"


def create_capture_backend(config: CaptureConfig | None = None) -> CaptureBackend:
    config = config or CaptureConfig()
    backend = config.backend.lower()
    if backend == "auto":
        backend = detect_capture_backend()

    if backend in {"wayland", "spectacle"}:
        return WaylandCapture(
            spectacle=config.spectacle_bin,
            mode=config.mode,
            delay_ms=config.delay_ms,
        )
    if backend == "x11":
        return X11Capture(import_bin=config.x11_bin)
    if backend == "windows":
        return WindowsCapture(powershell=config.windows_bin)
    if backend in {"mac", "macos", "darwin"}:
        return MacCapture(screencapture=config.mac_bin)
    raise ScreenCaptureError(
        f"Unsupported capture backend {config.backend!r}; use auto, wayland, x11, windows, mac, or spectacle"
    )


def detect_capture_backend() -> str:
    system = platform.system().lower()
    if system == "windows":
        return "windows"
    if system == "darwin":
        return "mac"
    if system == "linux":
        session_type = os.getenv("XDG_SESSION_TYPE", "").lower()
        if session_type == "x11":
            return "x11"
        return "wayland"
    return "wayland"


# Backwards-compatible alias for older library callers.
ScreenCapture = WaylandCapture
