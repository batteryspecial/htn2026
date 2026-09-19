"""The actuator: what turns "the target is over there" into the view moving.

The abstraction that survives the hardware. Today the motor driver is an arrow
renderer telling a person how to pan; later it is a stepper on a serial port.
The tracking logic is identical either way, so none of the work here is wasted
if the stepper arrives.
"""

from __future__ import annotations

from typing import Protocol

from contracts import MotorCommand
from render.layers import Layer


class Actuator(Protocol):
    """Takes what the tracker wants, returns whatever should be drawn."""

    def command(self, motor: MotorCommand | None, shape: tuple[int, int],
                now: float) -> list[Layer]: ...
