"""Choosing which of two targets to publish.

The controller can only chase one thing at a time, so with a two-target spec
the pipeline alternates. Plain alternation is not enough on its own: if the
active target walks out of frame, waiting out the full turn means the car sits
still while the other target is plainly visible. So a target that has been
missing for a while yields its turn early.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from config import CFG

log = logging.getLogger("perception.arbitrate")


@dataclass
class Arbiter:
    n_targets: int
    alternate_s: float = 3.0
    starve_s: float = CFG.ARBITRATE_STARVE_S
    index: int = 0
    last_flip: float = 0.0
    last_seen: dict[int, float] = field(default_factory=dict)

    def choose(self, visible: list[bool], now: float) -> int:
        """Index of the target to publish this frame."""
        for i, v in enumerate(visible):
            if v:
                self.last_seen[i] = now
        if self.n_targets < 2:
            return 0
        if not self.last_flip:
            self.last_flip = now

        if now - self.last_flip >= self.alternate_s:
            return self._flip(now, "turn")

        # Early yield: the active one is gone and another is right there.
        gone = now - self.last_seen.get(self.index, 0.0) > self.starve_s
        if gone and any(v for i, v in enumerate(visible) if i != self.index):
            return self._flip(now, "starved")
        return self.index

    def _flip(self, now: float, why: str) -> int:
        self.index = (self.index + 1) % self.n_targets
        self.last_flip = now
        log.debug("arbiter -> target %d (%s)", self.index, why)
        return self.index
