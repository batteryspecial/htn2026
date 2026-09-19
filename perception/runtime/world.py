"""The world: everything the loop is currently doing, as one swappable object.

What survives a change is the part that matters:

- Adding or removing a behaviour rebuilds the world but **keeps the tracker,
  the attribute cache and the trails**, because starting a privacy blur must
  not make the follow-cam lose its target.
- Changing the model **does** reset them, because tracker ids and cached crops
  from a different detector mean nothing.

Behaviours already running are carried across a rebuild by identity, so only a
new one starts from its initial state.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import supervision as sv
from supervision.tracker.byte_tracker.core import ByteTrack

from attributes.clip_cache import AttributeBank
from behaviors.base import Behavior
from detectors.base import Detector

log = logging.getLogger("perception.world")

TRAIL_LEN = 40


@dataclass
class World:
    model_name: str
    detector: Detector
    prepared: Any                 # opaque detector payload from prepare()
    behaviors: dict[str, Behavior] = field(default_factory=dict)
    text_vectors: dict[str, Any] = field(default_factory=dict)
    baseline_vectors: dict[str, Any] = field(default_factory=dict)
    created_ts: float = field(default_factory=time.time)

    @staticmethod
    def prompt_union(behaviors) -> list[str]:
        """Every class any behaviour needs. One detector pass serves them all.

        Paused behaviours are included: they are waiting for a model that can
        see them, and leaving them out would stop them ever resuming.
        """
        names: list[str] = []
        for b in behaviors:
            for n in b.subject.prompts():
                if n not in names:
                    names.append(n)
        return names

    @staticmethod
    def attribute_texts(behaviors) -> list[str]:
        texts: list[str] = []
        for b in behaviors:
            for t in b.subject.attribute_texts():
                if t not in texts:
                    texts.append(t)
        return texts

    def revalidate(self) -> list[str]:
        """Pause behaviours this model cannot see; resume the ones it can.

        A fixed-vocabulary model cannot be asked for "duck". Rather than drop
        the behaviour, it is held with a reason and comes back by itself when a
        model that can see it returns.
        """
        vocab = self.detector.classes
        changed = []
        for b in self.behaviors.values():
            missing = [] if vocab is None else [
                c for c in b.subject.prompts() if c not in set(vocab)]
            if missing and b.state != "PAUSED":
                b.pause(f"{self.model_name} cannot detect {missing}")
                changed.append(b.id)
            elif not missing and b.state == "PAUSED":
                b.resume()
                changed.append(b.id)
        return changed

    def summary(self) -> str:
        if not self.behaviors:
            return f"{self.model_name} [idle]"
        bits = ", ".join(f"{b.kind}:{b.subject.summary()}" for b in self.behaviors.values())
        return f"{self.model_name} [{bits}]"


@dataclass
class Shared:
    """State that outlives a world rebuild. Owned by the loop."""

    tracker: ByteTrack = field(default_factory=ByteTrack)
    bank: AttributeBank = field(default_factory=AttributeBank)
    trails: dict[int, deque] = field(default_factory=lambda: defaultdict(
        lambda: deque(maxlen=TRAIL_LEN)))

    def reset(self, why: str) -> None:
        log.info("resetting tracker, attributes and trails: %s", why)
        self.tracker = ByteTrack()
        self.bank = AttributeBank(self.bank.encoder)
        self.trails.clear()

    def track(self, dets: sv.Detections) -> sv.Detections:
        out = self.tracker.update_with_detections(dets)
        self._trails(out)
        return out

    def _trails(self, dets: sv.Detections) -> None:
        ids = dets.tracker_id
        if ids is None:
            return
        for i in range(len(dets)):
            x1, y1, x2, y2 = dets.xyxy[i]
            self.trails[int(ids[i])].append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
        if len(self.trails) > 512:  # a live stream never ends
            live = {int(v) for v in ids}
            for tid in [t for t in self.trails if t not in live]:
                del self.trails[tid]

    def paths(self) -> dict[int, list[tuple[int, int]]]:
        return {tid: list(pts) for tid, pts in self.trails.items()}
