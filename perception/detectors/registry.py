"""The model registry: every detector the service can switch to.

Adding a model is meant to be an edit to models.yaml plus, at most, one class
implementing the four methods in base.py. Nothing in the loop, the loader, the
state machine or the contracts should need to change. That is the whole point
of the registry sitting here.

Availability is decided at boot and never blocks it. A TensorRT engine on a
machine with no CUDA, or a weights file that is not there, marks the entry
`available: false` and the process still comes up. The service reports what it
can actually do rather than refusing to start.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from config import CFG, resolve_device
from detectors.base import Detector
from detectors.fake import FakeDetector
from detectors.ultralytics_fixed import UltralyticsFixedDetector
from detectors.yoloe import YoloeDetector

log = logging.getLogger("perception.registry")

BUILDERS = {
    "yoloe": lambda e: YoloeDetector(e.name, e.weights_path),
    "ultralytics_fixed": lambda e: UltralyticsFixedDetector(e.name, e.weights_path),
    "fake": lambda e: FakeDetector(e.name),
}


class UnknownModelError(KeyError):
    """TaskSpec.model named something the registry has never heard of."""


class ModelUnavailableError(RuntimeError):
    """The model exists in config but cannot run on this machine."""


@dataclass
class Entry:
    name: str
    type: str
    weights: str | None = None
    preload: bool = False
    requires_cuda: bool = False
    detector: Detector | None = None
    available: bool = True
    unavailable_reason: str | None = None
    error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def weights_path(self) -> Path:
        """Bare filenames resolve under WEIGHTS_DIR; explicit paths are respected."""
        if not self.weights:
            return Path()
        p = Path(self.weights)
        return p if p.parent != Path(".") else CFG.WEIGHTS_DIR / p

    @property
    def loaded(self) -> bool:
        return self.detector is not None and self.detector.is_loaded

    def build(self) -> None:
        """Construct the detector object. No I/O, no weights: this only makes
        the entry able to report what it is before anyone tries to load it."""
        if self.detector is None and self.available:
            self.detector = BUILDERS[self.type](self)

    def check_available(self, device: str) -> None:
        """Decide once, at boot, whether this entry can run here."""
        if self.type not in BUILDERS:
            self.available = False
            self.unavailable_reason = f"unknown type {self.type!r}"
        elif self.requires_cuda and device != "cuda":
            self.available = False
            self.unavailable_reason = f"needs CUDA, device is {device}"
        elif self.weights and self.weights_path.suffix == ".engine" and not self.weights_path.exists():
            # Engines are built on the GPU host and never committed, so a
            # missing one is expected everywhere else.
            self.available = False
            self.unavailable_reason = f"engine not built: {self.weights_path}"

    def describe(self) -> dict[str, Any]:
        d = {
            "name": self.name,
            "type": self.type,
            "open_vocab": self.detector.open_vocab if self.detector else None,
            "classes": self.detector.classes if self.detector else None,
            "loaded": self.loaded,
            "available": self.available,
        }
        if self.unavailable_reason:
            d["reason"] = self.unavailable_reason
        if self.error:
            d["error"] = self.error
        return d


class Registry:
    """Holds every Entry. Safe to call `get` from the loader worker while the
    API reads `manifest` on the event loop."""

    def __init__(self, entries: list[Entry]) -> None:
        self._entries = {e.name: e for e in entries}
        device = resolve_device()
        for e in self._entries.values():
            e.check_available(device)
            e.build()

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> Registry:
        path = Path(path or CFG.MODELS_CONFIG)
        raw = yaml.safe_load(path.read_text()) or {}
        return cls([Entry(**m) for m in raw.get("models", [])])

    # 1. Lookup --------------------------------------------------------
    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def entry(self, name: str) -> Entry:
        try:
            return self._entries[name]
        except KeyError:
            known = sorted(self._entries)
            raise UnknownModelError(f"no model {name!r}; registry has {known}") from None

    def is_loaded(self, name: str) -> bool:
        return name in self._entries and self._entries[name].loaded

    # 2. Loading -------------------------------------------------------
    def get(self, name: str) -> Detector:
        """Return a loaded detector, cold-loading it if needed.

        Worker thread only: this can take seconds. Raises rather than returning
        a half-built detector, so a failure upstream is a clean no-op.
        """
        e = self.entry(name)
        if not e.available:
            raise ModelUnavailableError(f"{name} unavailable: {e.unavailable_reason}")
        with e._lock:
            if e.loaded:
                return e.detector
            det = e.detector
            try:
                det.load()
                det.warmup(CFG.IMGSZ)
            except Exception as exc:
                # Leave the object in place so the entry can still say what it
                # is, but flag it so nothing downstream treats it as usable.
                e.error = f"{type(exc).__name__}: {exc}"
                det._loaded = False
                log.exception("failed to load %s", name)
                raise
            e.error = None
            log.info("loaded %s (%s)", name, e.type)
            return det

    def preload(self) -> list[str]:
        """Load everything marked preload. Returns the names that failed.

        Cold loading stays as a fallback path, so on the GPU host a model swap
        costs one frame instead of ten seconds.
        """
        failed = []
        for e in self._entries.values():
            if not (e.preload and e.available):
                continue
            try:
                self.get(e.name)
            except Exception:
                failed.append(e.name)
        return failed

    # 3. Reporting -----------------------------------------------------
    def manifest(self) -> list[dict[str, Any]]:
        """GET /models. The orchestrator builds the LLM's device manifest from
        this, so a fixed-vocab model must report its real class list."""
        return [e.describe() for e in self._entries.values()]

    def names(self) -> list[str]:
        return list(self._entries)
