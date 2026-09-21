"""Last-known model connectivity, separate from process/perception health."""

from __future__ import annotations

import time
from dataclasses import dataclass


def classify(exc: Exception) -> str:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if "auth" in name or "401" in text or "api key" in text:
        return "authentication"
    if "rate" in name or "429" in text or "rate limit" in text:
        return "rate_limit"
    if "timeout" in name or "timed out" in text:
        return "timeout"
    if "connect" in name or "connection" in text:
        return "connection"
    if "not found" in text or "model" in name and "not" in text:
        return "model"
    return "provider"


@dataclass
class ModelStatus:
    state: str = "unknown"
    detail: str = "No model call has completed since startup"
    checked_at: float | None = None

    def success(self) -> None:
        self.state = "ok"
        self.detail = "Model connection verified"
        self.checked_at = time.time()

    def failure(self, exc: Exception) -> None:
        category = classify(exc)
        self.state = "error"
        self.detail = f"{category}: {str(exc)[:180]}"
        self.checked_at = time.time()

    def view(self) -> dict:
        return {
            "state": self.state,
            "detail": self.detail,
            "checked_at": self.checked_at,
        }


MODEL_STATUS = ModelStatus()
