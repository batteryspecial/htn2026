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

import copy
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import supervision as sv
from supervision.tracker.byte_tracker.core import ByteTrack

from attributes.clip_cache import AttributeBank
from config import CFG
from behaviors.base import Behavior
from detectors.base import Detector

log = logging.getLogger("perception.world")

TRAIL_LEN = 40


def new_tracker() -> ByteTrack:
    """A tracker whose activation gate matches the detector's.

    `ByteTrack()` gates new tracks at `track_activation_threshold + 0.1` = 0.35,
    which silently discards the 0.15-0.30 band open-vocabulary detections land
    in. Every construction of a tracker has to go through here, or the loop
    detects a subject on every frame and the behaviour layer never sees it.
    """
    return ByteTrack(
        track_activation_threshold=CFG.TRACK_ACTIVATION_THRESHOLD,
        lost_track_buffer=CFG.LOST_TRACK_BUFFER,
        minimum_matching_threshold=CFG.MIN_MATCHING_THRESHOLD,
        frame_rate=CFG.TRACK_FRAME_RATE,
    )


def _boost(dets: sv.Detections) -> sv.Detections:
    """Map detector confidence into the range ByteTrack expects.

    ByteTrack's gates assume a COCO-style detector where a real object scores
    0.8+; the hardest of them (`thresh=0.7` inline, for confirming a new
    track) demands 0.30. Open-vocabulary matching is a similarity on a
    different scale, where 0.15-0.30 is a correct detection, so every gate
    lands in the wrong place and correct detections flicker.

    `[CONF_THRESHOLD, 1]` is mapped onto `[TRACKER_CONF_FLOOR, 1]`, which is
    strictly increasing: relative ordering, and therefore every matching
    decision, is untouched. Only the absolute scale moves.
    """
    conf = dets.confidence
    if conf is None or len(dets) == 0:
        return dets
    lo, floor = CFG.CONF_THRESHOLD, CFG.TRACKER_CONF_FLOOR
    if floor <= lo or lo >= 1.0:
        return dets
    scaled = floor + (np.clip(conf, lo, 1.0) - lo) * (1.0 - floor) / (1.0 - lo)
    out = copy.copy(dets)
    out.confidence = scaled.astype(np.float32)
    return out


def _restore(dets: sv.Detections) -> sv.Detections:
    """Undo `_boost`, so everything downstream sees the detector's real score.

    Behaviours gate on confidence — the appearance EMA only updates above 0.6,
    queries report it, the agent reads it — and all of that must be the number
    the detector actually produced.
    """
    conf = dets.confidence
    if conf is None or len(dets) == 0:
        return dets
    lo, floor = CFG.CONF_THRESHOLD, CFG.TRACKER_CONF_FLOOR
    if floor <= lo or lo >= 1.0:
        return dets
    true = lo + (conf - floor) * (1.0 - lo) / (1.0 - floor)
    dets.confidence = np.clip(true, 0.0, 1.0).astype(np.float32)
    return dets


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

    tracker: ByteTrack = field(default_factory=new_tracker)
    bank: AttributeBank = field(default_factory=AttributeBank)
    trails: dict[int, deque] = field(default_factory=lambda: defaultdict(
        lambda: deque(maxlen=TRAIL_LEN)))
    #: Set by the loop from the real frame, so the size floor is a fraction of
    #: the frame rather than a pixel count tied to one camera.
    frame_area: int = 640 * 480

    def reset(self, why: str) -> None:
        log.info("resetting tracker, attributes and trails: %s", why)
        self.tracker = new_tracker()
        self.bank = AttributeBank(self.bank.encoder)
        self.trails.clear()

    def track(self, dets: sv.Detections) -> sv.Detections:
        # Boosted for the tracker, restored immediately afterwards, so the
        # rescaling is invisible to everything except ByteTrack's thresholds.
        out = self.tracker.update_with_detections(_boost(self._winnow(dets)))
        self._trails(out)
        return _restore(out)

    def _winnow(self, dets: sv.Detections) -> sv.Detections:
        """Drop boxes too degenerate to be anything.

        Deliberately close to nothing. The previous version used a fraction of
        frame area, which threw away every face past about 3.5 m at 1080p —
        measured — because the cutoff rose with resolution. A detection that is
        small is not a detection that is wrong; the confidence threshold is the
        tool for "probably not real", and this is only for boxes a few pixels
        across.
        """
        if len(dets) == 0:
            return dets
        w = dets.xyxy[:, 2] - dets.xyxy[:, 0]
        h = dets.xyxy[:, 3] - dets.xyxy[:, 1]
        keep = np.minimum(w, h) >= CFG.MIN_BOX_PX
        if CFG.MIN_BOX_FRAC > 0:
            keep &= dets.box_area >= CFG.MIN_BOX_FRAC * self.frame_area
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
