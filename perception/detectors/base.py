"""The detector boundary.

Everything above this line speaks one language: a list of class names in, a
`supervision.Detections` out. Everything below it is model-specific and stays
model-specific. That split is what makes "switch to RF-DETR" a config change
rather than a pipeline change.

Three calls, deliberately split across two threads:

1. `prepare(prompts)` runs on the loader worker. It translates a spec's word
   list into whatever this model actually needs, and it is the *only* place a
   spec is validated against a model. It must be pure: if it raises, the
   detector is exactly as it was, so a bad spec is a no-op.
2. `apply(prepared)` runs on the inference loop, between two frames. It only
   installs what `prepare` already built, so it has to be cheap.
3. `infer(frame)` runs on the inference loop, once per frame.

The vocabulary story is carried by two attributes and nothing else:

- `open_vocab=True,  classes=None`  -> accepts any text (YOLOE)
- `open_vocab=False, classes=[...]` -> frozen list, spec must be a subset
                                       (COCO, a TensorRT build, RF-DETR)

A future single-purpose model that ignores prompts entirely is just the second
case with a short list, so no third mode is needed.
"""

from __future__ import annotations

import functools
import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import supervision as sv

from config import CFG

log = logging.getLogger("perception.detector")


@functools.cache
def use_weights_dir() -> None:
    """Point ultralytics at WEIGHTS_DIR before any load.

    Ultralytics resolves assets against the working directory first, then this
    setting. YOLOE silently pulls a ~570MB MobileCLIP text encoder on first use,
    so without this it lands wherever the service happened to be started from
    and gets re-downloaded on the next deploy. Pinning it makes pre-caching for
    an offline venue actually work.
    """
    from ultralytics.utils import SETTINGS

    CFG.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS["weights_dir"] = str(CFG.WEIGHTS_DIR)


class UnknownClassError(ValueError):
    """A spec asked for a class this model cannot produce.

    Raised by `prepare`, so it lands in the loader worker and turns into a
    `failed` event. The running task is untouched.
    """


def validate_vocab(prompts: list[str], classes: list[str] | None, model: str) -> None:
    """Reject anything the model cannot name, with a message a human can act on."""
    if not prompts:
        raise UnknownClassError(f"{model}: spec asked for no classes at all")
    if classes is None:
        return
    known = set(classes)
    missing = [p for p in prompts if p not in known]
    if missing:
        raise UnknownClassError(
            f"{model} has a fixed vocabulary and does not know {missing}. "
            f"It knows {len(classes)} classes, e.g. {classes[:8]}"
        )


def empty_detections() -> sv.Detections:
    """An empty result that still carries `class_name`, so callers never branch."""
    d = sv.Detections.empty()
    d.data["class_name"] = np.array([], dtype=object)
    return d


def ensure_class_names(d: sv.Detections) -> sv.Detections:
    """Guarantee the `class_name` key exists. Some backends omit it when empty."""
    if "class_name" not in d.data:
        d.data["class_name"] = np.array([], dtype=object)
    return d


class Detector(ABC):
    """Base for every detector. Subclasses implement four methods."""

    #: Registry name, e.g. "yoloe". Matches TaskSpec.model.
    name: str = "base"
    #: True when the model accepts arbitrary text prompts at runtime.
    open_vocab: bool = False

    def __init__(self, name: str) -> None:
        self.name = name
        self._loaded = False

    # 1. Lifecycle -----------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    @abstractmethod
    def classes(self) -> list[str] | None:
        """Frozen vocabulary, or None when open. Unknown until loaded."""

    @abstractmethod
    def load(self) -> None:
        """Heavy. Worker thread only. Must set `self._loaded`."""

    @abstractmethod
    def warmup(self, imgsz: int) -> None:
        """A few throwaway inferences so the first real frame is not the slow one."""

    # 2. Per-spec ------------------------------------------------------
    @abstractmethod
    def prepare(self, prompts: list[str]) -> Any:
        """Worker thread. Pure. Raises UnknownClassError on a vocabulary miss."""

    @abstractmethod
    def apply(self, prepared: Any) -> None:
        """Loop thread. Cheap. Installs what `prepare` built."""

    # 3. Per-frame -----------------------------------------------------
    @abstractmethod
    def infer(self, frame: np.ndarray) -> sv.Detections:
        """Loop thread. Always returns Detections carrying `class_name`."""

    # 4. Introspection -------------------------------------------------
    def describe(self) -> dict[str, Any]:
        """Row for GET /models. The orchestrator's LLM builds its manifest from
        this, so a fixed-vocab model's class list has to be honest."""
        return {
            "name": self.name,
            "open_vocab": self.open_vocab,
            "classes": self.classes,
            "loaded": self.is_loaded,
        }
