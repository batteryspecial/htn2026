"""A detector with no model behind it.

Two reasons it ships rather than living in tests:

1. The whole service has to be runnable on a laptop with no weights and no GPU,
   which is the only way the loop, the hot-swap and the endpoints get exercised
   before the GPU host exists.
2. It is open-vocabulary by construction, so it answers to any spec the
   orchestrator produces, which makes it the right target for a first end-to-end
   run against the real UI.

Default behaviour is a box per requested class drifting across the frame.
`set_script` replaces that with an exact per-frame answer for tests.
"""

from __future__ import annotations

import math

import numpy as np
import supervision as sv

from detectors.base import Detector, empty_detections, validate_vocab


class FakeDetector(Detector):
    open_vocab = True

    def __init__(self, name: str = "fake", period: int = 90) -> None:
        super().__init__(name)
        self._active: list[str] = []
        self._frame = 0
        self._script: list[sv.Detections] | None = None
        self._period = period

    @property
    def classes(self) -> list[str] | None:
        return None

    @property
    def active_classes(self) -> list[str]:
        return list(self._active)

    def load(self) -> None:
        self._loaded = True

    def warmup(self, imgsz: int = 320) -> None:
        pass

    def prepare(self, prompts: list[str]) -> list[str]:
        validate_vocab(prompts, None, self.name)
        return list(dict.fromkeys(prompts))

    def apply(self, prepared: list[str]) -> None:
        self._active = list(prepared)

    def set_script(self, frames: list[sv.Detections]) -> None:
        """Play back exact detections, then repeat the last one forever."""
        self._script = frames
        self._frame = 0

    def infer(self, frame: np.ndarray) -> sv.Detections:
        i, self._frame = self._frame, self._frame + 1
        if self._script is not None:
            return self._script[min(i, len(self._script) - 1)]
        if not self._active:
            return empty_detections()

        h, w = frame.shape[:2]
        boxes, confs, names = [], [], []
        for k, cls in enumerate(self._active):
            # Each class gets its own lane and phase, so two targets are
            # distinguishable and neither sits still.
            phase = 2 * math.pi * ((i / self._period) + k / max(len(self._active), 1))
            cx = w * (0.5 + 0.3 * math.sin(phase))
            cy = h * (0.5 + 0.15 * math.cos(phase))
            bw, bh = w * 0.18, h * 0.3
            boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
            confs.append(0.9 - 0.05 * k)
            names.append(cls)

        return sv.Detections(
            xyxy=np.array(boxes, dtype=np.float32),
            confidence=np.array(confs, dtype=np.float32),
            class_id=np.arange(len(names)),
            data={"class_name": np.array(names, dtype=object)},
        )
