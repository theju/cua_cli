from __future__ import annotations

from dataclasses import dataclass

from .models import ComputerAction


RISKY_TEXT_MARKERS = {
    "password",
    "passcode",
    "otp",
    "2fa",
    "auth code",
    "credit card",
    "cvv",
    "ssn",
    "social security",
    "bank",
    "wire transfer",
    "api key",
    "token",
    "secret",
}

RISKY_ACTION_MARKERS = {
    "buy",
    "purchase",
    "pay",
    "submit",
    "send",
    "upload",
    "delete",
    "remove",
    "transfer",
    "share",
    "authorize",
    "confirm",
}


@dataclass(frozen=True)
class SafetyPolicy:
    mode: str = "confirm_risky"

    def needs_confirmation(self, action: ComputerAction) -> bool:
        if self.mode == "step":
            return True
        if self.mode == "yes":
            return False
        if action.type in {"type", "click", "double_click", "keypress"}:
            label = action.label().lower()
            if any(marker in label for marker in RISKY_ACTION_MARKERS):
                return True
        if action.type == "type":
            text = str(action.data.get("text", "")).lower()
            if any(marker in text for marker in RISKY_TEXT_MARKERS):
                return True
        return False
