"""pan_to: guide the view through a turn, and know when it has arrived.

The surveyor. "Turn 45 degrees left and count the people you see" is this
behaviour followed by a count: the arrow tells a person which way to swing the
camera, odometry measures how far it has actually gone, and `reached` tells the
agent it may now ask its question.

    GUIDING --within tolerance--> REACHED

The subject is not used to decide anything here -- the turn is about the view,
not about an object. It is still filtered and counted, so the operator can
watch the tally build while panning, which is the whole point of the item.
"""

from __future__ import annotations

import numpy as np

from actuator.odometry import PanOdometry
from behaviors.base import Behavior, Frame, Outcome
from contracts import MotorCommand
from render.layers import Alert, Boxes

#: How close is close enough. Tighter than this and a human cannot stop
#: accurately enough to ever satisfy it.
TOLERANCE_DEG = 5.0
#: Degrees per second to ask for, scaled by how far is left.
MAX_RATE_DEG_S = 30.0
#: How long REACHED keeps its banner up.
FLASH_S = 2.0


class PanTo(Behavior):
    kind = "pan_to"
    states = ("GUIDING", "REACHED", "PAUSED")
    initial = "GUIDING"
    emits = ("reached",)

    def __init__(self, *a, **kw) -> None:
        self.odometry: PanOdometry | None = None
        self._reached_at = 0.0
        super().__init__(*a, **kw)

    def validate(self) -> None:
        try:
            self.target_deg = float(self.param("deg", 0.0))
        except (TypeError, ValueError):
            raise ValueError("pan_to needs 'deg': how far to turn, negative for left") from None
        if self.target_deg == 0.0:
            raise ValueError("pan_to needs a non-zero 'deg'")
        self.hfov_deg = float(self.param("hfov_deg", 70.0))
        self.tolerance_deg = float(self.param("tolerance_deg", TOLERANCE_DEG))
        if self.hfov_deg <= 0:
            raise ValueError("pan_to needs a positive 'hfov_deg'")

    # 1. Per frame ------------------------------------------------------
    def on_frame(self, frame: Frame, idx: np.ndarray, outcome: Outcome) -> None:
        if self.odometry is None:
            # Measuring starts when the behaviour does: "turn 45 degrees" means
            # from here, not from wherever the camera was pointed at boot.
            self.odometry = PanOdometry(self.hfov_deg)

        turned = self.odometry.update(frame.image)
        remaining = self.target_deg - turned
        self.data.update(turned_deg=round(turned, 1),
                         remaining_deg=round(remaining, 1),
                         target_deg=self.target_deg,
                         samples=self.odometry.samples,
                         matches=int(len(idx)))

        if self.state == "REACHED":
            if frame.now - self._reached_at < FLASH_S:
                outcome.layers.append(Alert(
                    text=f"reached {self.target_deg:+.0f}°  ·  {len(idx)} in view",
                    color=self.color, intensity=0.5))
            self._draw(frame, idx, outcome)
            return

        if abs(remaining) <= self.tolerance_deg:
            self._reached_at = frame.now
            self.to("REACHED", f"turned {turned:+.0f}°")
            outcome.events.append(self.event(
                "reached", frame, None,
                detail=f"{self.label}: turned {turned:+.0f}° of {self.target_deg:+.0f}°",
                turned_deg=round(turned, 1), matches=int(len(idx))))
        else:
            outcome.motor = self._motor(remaining)

        self._draw(frame, idx, outcome)

    def _motor(self, remaining: float) -> MotorCommand:
        """Ask for a turn proportional to what is left, so the arrow eases off
        as the target approaches rather than snapping from full to nothing."""
        scale = min(1.0, abs(remaining) / max(self.target_deg, 1e-6))
        rate = MAX_RATE_DEG_S * scale * (1.0 if remaining > 0 else -1.0)
        side = "right" if remaining > 0 else "left"
        return MotorCommand(rate_deg_s=rate,
                            reason=f"pan {side} {abs(remaining):.0f}°")

    def _draw(self, frame: Frame, idx: np.ndarray, outcome: Outcome) -> None:
        if len(idx) and self.spec.render.boxes:
            outcome.layers.append(Boxes(
                boxes=frame.tracks[idx].xyxy,
                labels=[self.label] + [""] * (len(idx) - 1),
                color=self.color))
