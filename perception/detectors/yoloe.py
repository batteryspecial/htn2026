"""YOLOE: the open-vocabulary detector.

This is the one model that must stay in PyTorch. Exporting YOLOE to TensorRT
bakes its prompt vocabulary into the engine, which removes the only reason we
run it. Fixed-vocabulary models are the ones worth compiling.

The prompt swap is split across the thread boundary on purpose:
`get_text_pe(names)` is a text-encoder forward pass and runs on the worker,
while `set_classes(names, pe)` mutates the model and so must run on the loop.
Because the embedding is already computed, the mutation is a tensor assignment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import supervision as sv

from config import CFG, quantize, resolve_device
from detectors.base import Detector, ensure_class_names, use_weights_dir, validate_vocab

log = logging.getLogger("perception.detector.yoloe")


@dataclass(frozen=True)
class YoloePrepared:
    """Immutable handoff from worker to loop."""

    names: list[str]
    text_pe: Any  # torch.Tensor, kept opaque so this module imports without torch


class YoloeDetector(Detector):
    open_vocab = True

    def __init__(self, name: str, weights: str | Path) -> None:
        super().__init__(name)
        self.weights = Path(weights)
        self._model = None
        self._active: list[str] = []
        self.device = "cpu"

    @property
    def classes(self) -> list[str] | None:
        return None  # open vocabulary: no frozen list to report

    @property
    def active_classes(self) -> list[str]:
        """What the model is currently set to detect. Diagnostics only."""
        return list(self._active)

    def load(self) -> None:
        from ultralytics import YOLOE

        use_weights_dir()
        log.info("loading YOLOE from %s", self.weights)
        self._model = YOLOE(str(self.weights))
        device = self.device = self._pick_device()
        if device != "cpu":
            self._model.to(device)
        self._loaded = True

    @staticmethod
    def _pick_device() -> str:
        """YOLOE runs on CPU on a Mac.

        Its MobileCLIP text encoder uses float64, which MPS cannot represent, so
        get_text_pe raises there and open-vocab swapping is the one thing we
        cannot lose. CPU at IMGSZ=320 is ~80ms/frame: slow, but the Mac only
        ever runs logic tests against recorded clips. CUDA is unaffected.
        """
        device = resolve_device()
        if device == "mps":
            log.warning("YOLOE forced to CPU: its text encoder needs float64, MPS has none")
            return "cpu"
        return device

    def warmup(self, imgsz: int = CFG.IMGSZ) -> None:
        if self._model is None:
            return
        # Needs a vocabulary before it will run at all.
        if not self._active:
            self.apply(self.prepare(["person"]))
        blank = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        for _ in range(3):
            self.infer(blank)

    def prepare(self, prompts: list[str]) -> YoloePrepared:
        """Worker thread. Encodes the prompts; touches no model state."""
        validate_vocab(prompts, None, self.name)
        if self._model is None:
            raise RuntimeError(f"{self.name} used before load()")
        names = list(dict.fromkeys(prompts))
        return YoloePrepared(names=names, text_pe=self._model.get_text_pe(names))

    def apply(self, prepared: YoloePrepared) -> None:
        """Loop thread. The only mutation, and it is a tensor assignment."""
        self._model.set_classes(prepared.names, prepared.text_pe)
        self._active = list(prepared.names)

    def infer(self, frame: np.ndarray, conf: float | None = None) -> sv.Detections:
        result = self._model.predict(
            frame,
            imgsz=CFG.IMGSZ,
            conf=CFG.CONF_THRESHOLD if conf is None else conf,
            quantize=quantize(),
            verbose=False,
        )[0]
        # Masks are kept: YOLOE-seg produces them anyway, the render layer
        # reads much better than boxes on a projector, and `render.mask`
        # defaults to on. They are the single largest thing in a Detections,
        # so if frame time suffers this is the first thing to drop.
        return ensure_class_names(sv.Detections.from_ultralytics(result))
