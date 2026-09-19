"""The world: everything the loop is currently doing, as one swappable object.

This replaces the old single-task model. The difference that matters is what
survives a change:

- Adding or removing a behaviour rebuilds the world but **keeps the tracker and
  the attribute cache**, because starting a privacy blur must not make the
  follow-cam lose its target.
- Changing the model **does** reset both, because tracker ids and cached crops
  from a different detector mean nothing.

Behaviours already running are carried across a rebuild by identity, so only
the new one starts from `arming`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import supervision as sv
from supervision.tracker.byte_tracker.core import ByteTrack

from behaviors.base import RuntimeBehavior
from contracts import Subject
from detectors.base import Detector
from stages.attributes import AttributeBank

log = logging.getLogger("perception.world")


@dataclass
class World:
    model_name: str
    detector: Detector
    prepared: Any                 # opaque detector payload from prepare()
    behaviors: dict[str, RuntimeBehavior] = field(default_factory=dict)
    #: Phrase and baseline vectors, encoded on the worker for the bank.
    text_vectors: dict[str, Any] = field(default_factory=dict)
    baseline_vectors: dict[str, Any] = field(default_factory=dict)
    created_ts: float = field(default_factory=time.time)

    # 1. What the detector must look for -------------------------------
    @staticmethod
    def prompt_union(behaviors) -> list[str]:
        """Every class any behaviour needs, deduplicated.

        One detector pass serves every behaviour, so two behaviours watching
        people cost one forward pass, not two.
        """
        names: list[str] = []
        for b in behaviors:
            for n in b.subject.prompts():
                if n not in names:
                    names.append(n)
        return names

    @staticmethod
    def attribute_texts(behaviors) -> list[str]:
        """Every CLIP phrase any behaviour needs, deduplicated."""
        texts: list[str] = []
        for b in behaviors:
            for t in b.subject.attribute_texts():
                if t not in texts:
                    texts.append(t)
        return texts

    @staticmethod
    def baseline_texts(behaviors) -> list[str]:
        """`"a {class}"` for every class, the contrastive comparison."""
        return [f"a {n}" for n in World.prompt_union(behaviors)]

    def subjects(self) -> list[Subject]:
        return [b.subject for b in self.behaviors.values()]

    def summary(self) -> str:
        if not self.behaviors:
            return f"{self.model_name} [idle]"
        bits = ", ".join(f"{b.kind}:{'+'.join(b.subject.detect)}"
                         for b in self.behaviors.values())
        return f"{self.model_name} [{bits}]"


@dataclass
class Shared:
    """State that outlives a world rebuild.

    Owned by the loop. Reset only when the detector changes, because that is
    the only change that invalidates tracker ids and cached crops.
    """

    tracker: ByteTrack = field(default_factory=ByteTrack)
    bank: AttributeBank = field(default_factory=AttributeBank)

    def reset(self, why: str) -> None:
        log.info("resetting tracker and attribute cache: %s", why)
        self.tracker = ByteTrack()
        self.bank = AttributeBank(self.bank.encoder)

    def track(self, dets: sv.Detections) -> sv.Detections:
        return self.tracker.update_with_detections(dets)
