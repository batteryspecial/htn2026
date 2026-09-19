"""Any Ultralytics model with a frozen class list: COCO weights, or a
TensorRT engine built from them.

The vocabulary is whatever the weights shipped with, so `prepare` is a lookup
from names to class ids and a hard failure on anything absent. Filtering at
inference is cheaper than filtering afterwards, and it is where the speed of a
compiled engine actually shows up.

TensorRT engines are tied to one GPU architecture and one TensorRT version.
They are built on the GPU host by scripts/build_engines.py and never committed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import supervision as sv

from config import CFG, quantize, resolve_device
from detectors.base import Detector, ensure_class_names, use_weights_dir, validate_vocab

log = logging.getLogger("perception.detector.fixed")


@dataclass(frozen=True)
class FixedPrepared:
    names: list[str]
    class_ids: list[int]


class UltralyticsFixedDetector(Detector):
    open_vocab = False

    def __init__(self, name: str, weights: str | Path) -> None:
        super().__init__(name)
        self.weights = Path(weights)
        self._model = None
        self._classes: list[str] | None = None
        self._id_of: dict[str, int] = {}
        self._active_ids: list[int] = []

    @property
    def classes(self) -> list[str] | None:
        return list(self._classes) if self._classes else None

    @property
    def is_engine(self) -> bool:
        return self.weights.suffix == ".engine"

    def load(self) -> None:
        from ultralytics import YOLO

        use_weights_dir()
        log.info("loading %s from %s", self.name, self.weights)
        self._model = YOLO(str(self.weights))
        if not self.is_engine and resolve_device() != "cpu":
            self._model.to(resolve_device())
        names = self._model.names
        self._classes = [names[i] for i in sorted(names)]
        self._id_of = {n: i for i, n in names.items()}
        self._loaded = True

    def warmup(self, imgsz: int = CFG.IMGSZ) -> None:
        if self._model is None:
            return
        blank = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        for _ in range(3):
            self.infer(blank)

    def prepare(self, prompts: list[str]) -> FixedPrepared:
        """Worker thread. Fails loudly and specifically on an unknown class."""
        if self._classes is None:
            raise RuntimeError(f"{self.name} used before load()")
        validate_vocab(prompts, self._classes, self.name)
        names = list(dict.fromkeys(prompts))
        return FixedPrepared(names=names, class_ids=[self._id_of[n] for n in names])

    def apply(self, prepared: FixedPrepared) -> None:
        self._active_ids = list(prepared.class_ids)

    def infer(self, frame: np.ndarray) -> sv.Detections:
        result = self._model.predict(
            frame,
            imgsz=CFG.IMGSZ,
            conf=CFG.CONF_THRESHOLD,
            classes=self._active_ids or None,
            # An engine has its precision compiled in, so never ask again.
            quantize=None if self.is_engine else quantize(),
            verbose=False,
        )[0]
        return ensure_class_names(sv.Detections.from_ultralytics(result))
