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
from config import CFG
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
    ref_vectors: dict[str, Any] = field(default_factory=dict)
    created_ts: float = field(default_factory=time.time)

    @staticmethod
    def prompt_union(behaviors) -> list[str]:
        """Every class any behaviour needs. One detector pass serves them all.

        Paused behaviours are included: they are waiting for a model that can
        see them, and leaving them out would stop them ever resuming.
        """
        names: list[str] = []
        for b in behaviors:
            for sel in b.selectors():
                for n in sel.prompts():
                    if n not in names:
                        names.append(n)
        return names

    @staticmethod
    def attribute_texts(behaviors) -> list[str]:
        texts: list[str] = []
        for b in behaviors:
            for sel in b.selectors():
                for t in sel.attribute_texts():
                    if t not in texts:
                        texts.append(t)
        return texts

    def revalidate(self, registry=None) -> list[str]:
        """Pause behaviours this setup cannot serve; resume the ones it can.

        Two reasons a behaviour cannot run: the active detector has no word for
        its subject, or a model role it needs is unfilled. Either way it is
        held with a reason rather than dropped, and comes back by itself when
        the missing piece returns.
        """
        vocab = self.detector.classes
        changed = []
        for b in self.behaviors.values():
            reason = None
            if registry is not None:
                unfilled = [r for r in getattr(b, "needs_roles", ())
                            if not registry.has_role(r)]
                if unfilled:
                    reason = f"no {', '.join(unfilled)} model available"
            if reason is None and vocab is not None:
                wanted = [c for sel in b.selectors() for c in sel.prompts()]
                missing = [c for c in wanted if c not in set(vocab)]
                if missing:
                    reason = f"{self.model_name} cannot detect {missing}"

            if reason and b.state != "PAUSED":
                b.pause(reason)
                changed.append(b.id)
            elif not reason and b.state == "PAUSED":
                b.resume()
                changed.append(b.id)
        return changed

    def roles_needed(self) -> set[str]:
        """Aux roles some active behaviour wants this frame."""
        return {r for b in self.behaviors.values() if b.state != "PAUSED"
                for r in getattr(b, "needs_roles", ())}

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
    #: Set by the loop from the real frame, so the size floor is a fraction of
    #: the frame rather than a pixel count tied to one camera.
    frame_area: int = 640 * 480

    def reset(self, why: str) -> None:
        log.info("resetting tracker, attributes and trails: %s", why)
        self.tracker = ByteTrack()
        self.bank = AttributeBank(self.bank.encoder)
        self.trails.clear()

    def track(self, dets: sv.Detections) -> sv.Detections:
        out = self.tracker.update_with_detections(self._winnow(dets))
        self._trails(out)
        return out

    def _winnow(self, dets: sv.Detections) -> sv.Detections:
        """Drop detections too small to be anything.

        Open-vocabulary detectors produce boxes on texture. On real footage
        three of four "people without dark jackets" were a patterned wall, and
        a junk box passes every exclusion by definition.
        """
        if len(dets) == 0 or CFG.MIN_BOX_FRAC <= 0:
            return dets
        keep = dets.box_area >= CFG.MIN_BOX_FRAC * self.frame_area
        return dets[keep] if not keep.all() else dets

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
