"""The model zoo: every model the service can use, of every kind.

One registry, entries typed by role. Adding a model is an edit to models.yaml
plus, at most, a class in the folder its role belongs to; nothing in the frame
loop, the worker, the behaviours or the contracts changes.

Availability is decided at boot and never blocks it. A TensorRT engine on a
machine with no CUDA, a weights file that is not there, a Python package that
was never installed: each marks the entry `available: false` with a reason and
the process still comes up. The service reports what it can actually do rather
than refusing to start, which matters because the laptop and the GPU host can
do different things and both have to run.

Exactly one model per role is **active**. Switching the pose model is the same
operation as switching the detector.
"""

from __future__ import annotations

import importlib.util
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from config import CFG, resolve_device
from zoo.roles import ROLES, Kind, Role, kind_of

log = logging.getLogger("perception.zoo")


class UnknownModelError(KeyError):
    """Something named a model the registry has never heard of."""


class ModelUnavailableError(RuntimeError):
    """The model exists in config but cannot run on this machine."""


class NoModelForRole(RuntimeError):
    """Nothing available can fill a role something needs."""


@dataclass
class Entry:
    name: str
    type: str
    #: Defaults to detector so existing configs keep working unchanged.
    role: Role = "detector"
    weights: str | None = None
    preload: bool = False
    requires_cuda: bool = False
    #: Kind-specific settings (an OCR language, a pose confidence floor).
    options: dict[str, Any] = field(default_factory=dict)

    model: Any = None
    available: bool = True
    unavailable_reason: str | None = None
    error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def kind(self) -> Kind | None:
        return kind_of(self.type)

    @property
    def weights_path(self) -> Path:
        """Bare filenames resolve under WEIGHTS_DIR; explicit paths are kept."""
        if not self.weights:
            return Path()
        p = Path(self.weights)
        return p if p.parent != Path(".") else CFG.WEIGHTS_DIR / p

    @property
    def loaded(self) -> bool:
        return self.model is not None and getattr(self.model, "is_loaded", False)

    # 1. Availability ---------------------------------------------------
    def check_available(self, device: str) -> None:
        """Decide once, at boot, whether this entry can run here."""
        kind = self.kind
        if kind is None:
            self.available = False
            self.unavailable_reason = f"unknown type {self.type!r}"
        elif self.role != kind.role:
            # A typo in models.yaml, caught now rather than when a behaviour
            # asks for a pose model and gets a detector.
            self.available = False
            self.unavailable_reason = (
                f"type {self.type!r} fills role {kind.role!r}, not {self.role!r}")
        elif kind.requires and importlib.util.find_spec(kind.requires) is None:
            self.available = False
            self.unavailable_reason = f"{kind.requires} is not installed"
        elif self.requires_cuda and device != "cuda":
            self.available = False
            self.unavailable_reason = f"needs CUDA, device is {device}"
        elif (self.weights and self.weights_path.suffix == ".engine"
              and not self.weights_path.exists()):
            # Engines are built on the GPU host and never committed, so a
            # missing one is expected everywhere else.
            self.available = False
            self.unavailable_reason = f"engine not built: {self.weights_path}"

    def build(self) -> None:
        """Construct the object without loading weights, so the entry can
        report what it is before anyone tries to use it."""
        if self.model is None and self.available and self.kind is not None:
            try:
                self.model = self.kind.build(self)
            except Exception as exc:
                self.available = False
                self.unavailable_reason = f"{type(exc).__name__}: {exc}"

    # 2. Reporting ------------------------------------------------------
    def describe(self, active: bool = False) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "role": self.role,
            "type": self.type,
            "loaded": self.loaded,
            "available": self.available,
            "active": active,
        }
        # Detector-shaped fields, kept for the agent's device manifest.
        if self.role == "detector":
            d["open_vocab"] = getattr(self.model, "open_vocab", None)
            d["classes"] = getattr(self.model, "classes", None)
        if self.unavailable_reason:
            d["reason"] = self.unavailable_reason
        if self.error:
            d["error"] = self.error
        return d


class Registry:
    """Holds every Entry. Safe to call `get` from the worker while the API
    reads `manifest` on the event loop."""

    def __init__(self, entries: list[Entry]) -> None:
        self._entries = {e.name: e for e in entries}
        device = resolve_device()
        for e in self._entries.values():
            e.check_available(device)
            e.build()
        #: role -> name of the model currently filling it.
        self._active: dict[str, str] = {}
        for role in ROLES:
            first = self.first_available(role)
            if first:
                self._active[role] = first

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> Registry:
        path = Path(path or CFG.MODELS_CONFIG)
        raw = yaml.safe_load(path.read_text()) or {}
        return cls([Entry(**m) for m in raw.get("models", [])])

    # 1. Lookup ---------------------------------------------------------
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

    def names(self, role: Role | None = None) -> list[str]:
        return [n for n, e in self._entries.items() if role is None or e.role == role]

    def first_available(self, role: Role) -> str | None:
        """Pick something to fill a role.

        Prefers an open-vocabulary detector, because the agent invents class
        names and a fixed vocabulary would refuse most of them. Skips anything
        that has already failed to load: a model that just blew up is not a
        candidate to replace itself.
        """
        candidates = [e for e in self._entries.values()
                      if e.role == role and e.available and not e.error]
        if not candidates:
            return None
        candidates.sort(key=lambda e: (not getattr(e.model, "open_vocab", False),
                                       not e.preload))
        return candidates[0].name

    # 2. Active model per role -------------------------------------------
    def active(self, role: Role) -> str | None:
        return self._active.get(role)

    def has_role(self, role: Role) -> bool:
        """Whether anything can fill this role. Behaviours ask before running."""
        return self._active.get(role) is not None

    def set_active(self, role: Role, name: str) -> None:
        e = self.entry(name)
        if e.role != role:
            raise ModelUnavailableError(f"{name!r} is a {e.role}, not a {role}")
        if not e.available:
            raise ModelUnavailableError(f"{name} unavailable: {e.unavailable_reason}")
        self._active[role] = name

    def get_active(self, role: Role) -> Any:
        """The loaded model filling a role. Worker thread: this can load."""
        name = self._active.get(role)
        if name is None:
            have = [e.name for e in self._entries.values() if e.role == role]
            raise NoModelForRole(
                f"no {role} model available"
                + (f"; configured but unusable: {have}" if have else
                   f"; none configured. Add one to {CFG.MODELS_CONFIG.name}"))
        return self.get(name)

    # 3. Loading ---------------------------------------------------------
    def get(self, name: str) -> Any:
        """Return a loaded model, cold-loading it if needed.

        Worker thread only: this can take seconds. Raises rather than returning
        a half-built model, so a failure upstream is a clean no-op.
        """
        e = self.entry(name)
        if not e.available:
            raise ModelUnavailableError(f"{name} unavailable: {e.unavailable_reason}")
        with e._lock:
            if e.loaded:
                return e.model
            model = e.model
            try:
                model.load()
                warmup = getattr(model, "warmup", None)
                if warmup:
                    warmup(CFG.IMGSZ)
            except Exception as exc:
                # Leave the object in place so the entry can still say what it
                # is, but flag it so nothing downstream treats it as usable.
                e.error = f"{type(exc).__name__}: {exc}"
                setattr(model, "_loaded", False)
                log.exception("failed to load %s", name)
                raise
            e.error = None
            log.info("loaded %s (%s, %s)", name, e.role, e.type)
            return model

    def preload(self) -> list[str]:
        """Load everything marked preload. Returns the names that failed.

        Cold loading stays a fallback, so on the GPU host a swap costs one
        frame instead of ten seconds.
        """
        failed = []
        for e in self._entries.values():
            if not (e.preload and e.available):
                continue
            try:
                self.get(e.name)
            except Exception:
                failed.append(e.name)
                # A role whose only model just failed should not still claim
                # to be filled.
                if self._active.get(e.role) == e.name:
                    replacement = self.first_available(e.role)
                    if replacement and replacement != e.name:
                        self._active[e.role] = replacement
        return failed

    def try_get_active(self, role: Role) -> Any | None:
        """Like get_active, but None instead of raising.

        For roles the pipeline degrades without rather than refuses to start:
        with no embedder, attribute checks simply never match.
        """
        try:
            return self.get_active(role)
        except Exception as exc:
            log.warning("no %s model: %s", role, exc)
            return None

    # 4. Reporting -------------------------------------------------------
    def manifest(self) -> list[dict[str, Any]]:
        """GET /models. The agent builds its device manifest from this, so a
        fixed-vocabulary model must report its real class list."""
        return [e.describe(active=self._active.get(e.role) == e.name)
                for e in self._entries.values()]

    def roles(self) -> dict[str, Any]:
        """Which roles are filled, and by what. The agent reads this to know
        whether a gesture trigger is possible at all."""
        return {role: {"active": self._active.get(role),
                       "available": [e.name for e in self._entries.values()
                                     if e.role == role and e.available]}
                for role in ROLES}
