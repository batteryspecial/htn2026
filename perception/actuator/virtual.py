"""The virtual motor: the human is the actuator.

Draws an arrow telling a person which way to pan. Three intensities, because
the difference between "drifting off-centre" and "gone" has to be obvious from
across a room:

- **Near the edge** — a faint arrow, early warning, nothing has gone wrong yet.
- **Gone** — a bold pulsing arrow toward the exit edge, held.
- **Searching** — after about ten seconds, a banner instead, because an arrow
  nobody is following stops meaning anything.

Holding the arrow is the tracker's job, not this one's: it keeps issuing an
urgent command until the *same instance* reacquires. This module only draws
what it is told.
"""

from __future__ import annotations

import math

from contracts import MotorCommand
from render.layers import Alert, Arrow, Layer

#: How long an urgent command runs before the arrow becomes a banner.
SEARCH_AFTER_S = 10.0


class VirtualMotor:
    def __init__(self, search_after_s: float = SEARCH_AFTER_S) -> None:
        self.search_after_s = search_after_s
        self.last: MotorCommand | None = None
        self._urgent_since: float | None = None

    def command(self, motor: MotorCommand | None, shape: tuple[int, int],
                now: float) -> list[Layer]:
        self.last = motor
        if motor is None or motor.rate_deg_s == 0.0:
            self._urgent_since = None
            return []

        direction = "right" if motor.rate_deg_s > 0 else "left"
        if not motor.urgent:
            self._urgent_since = None
            # Faint: the subject is still visible, just drifting.
            return [Arrow(direction=direction, strength=0.35, pulse=0.6,
                          label=motor.reason)]

        if self._urgent_since is None:
            self._urgent_since = now
        if now - self._urgent_since > self.search_after_s:
            return [Alert(text="target lost — searching", color=(60, 190, 250),
                          intensity=0.5)]

        # Bold and pulsing, so it reads as "do something" rather than "noted".
        pulse = 0.55 + 0.45 * math.sin(now * 6.0)
        return [Arrow(direction=direction, strength=1.0, pulse=pulse,
                      label=f"pan {direction}")]
