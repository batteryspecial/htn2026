"""What the loop swaps in.

A PreparedTask is everything needed to run one spec, built entirely on the
loader worker and handed to the inference loop complete. Nothing in it is
filled in later, which is what makes the swap a single pointer assignment
between two frames.

Note the tracker and the per-target memory are created here, not reused. A new
spec therefore starts with no tracking history and no latched target: retasking
forgets the old target rather than trying to reconcile it with the new one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import supervision as sv
from supervision.tracker.byte_tracker.core import ByteTrack

from contracts import TaskSpec
from detectors.base import Detector
from stages.arbitrate import Arbiter
from stages.select import SelectState


@dataclass
class PreparedTask:
    instruction_id: str
    spec: TaskSpec
    model_name: str
    detector: Detector
    prepared: Any  # opaque; only the detector that built it knows the shape
    tracker: ByteTrack
    arbiter: Arbiter
    select_states: dict[str, SelectState]
    created_ts: float = field(default_factory=time.time)

    @classmethod
    def build(cls, instruction_id: str, spec: TaskSpec, model_name: str,
              detector: Detector, prepared: Any) -> PreparedTask:
        return cls(
            instruction_id=instruction_id,
            spec=spec,
            model_name=model_name,
            detector=detector,
            prepared=prepared,
            tracker=ByteTrack(),
            arbiter=Arbiter(n_targets=len(spec.targets),
                            alternate_s=spec.arbitration.alternate_s),
            select_states={t.ref: SelectState() for t in spec.targets},
        )

    def track(self, dets: sv.Detections) -> sv.Detections:
        return self.tracker.update_with_detections(dets)

    def summary(self) -> str:
        bits = "; ".join(f"{t.ref}:{'+'.join(t.detect)}" for t in self.spec.targets)
        return f"{self.model_name} [{bits}]"
