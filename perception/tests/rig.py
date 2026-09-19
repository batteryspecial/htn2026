"""A whole pipeline driven by hand, with no camera, no model and no CLIP.

Every test above this uses it, so the thing under test is the real loop, the
real builder and the real behaviour code, with only the outside world faked.
"""

from __future__ import annotations

import time

import numpy as np
import supervision as sv

from contracts import Phase
from detectors.base import empty_detections
from detectors.registry import Registry
from runtime.builder import Builder
from runtime.events import Bus
from runtime.loop import InferenceLoop
from runtime.state import Machine
from runtime.world import Shared
from stages.encoders import HashEncoder

MODELS_YAML = (
    "models:\n"
    "  - name: fake\n    type: fake\n    preload: true\n"
    "  - name: other\n    type: fake\n    preload: true\n"
    "  - name: gpu_only\n    type: ultralytics_fixed\n"
    "    weights: nope.engine\n    requires_cuda: true\n"
)

# Colours the HashEncoder can tell apart, so "red" really is not "blue".
BGR = {"red": (0, 0, 255), "blue": (255, 0, 0), "green": (0, 255, 0)}


class FakeCapture:
    """A camera we can freeze, kill and revive."""

    def __init__(self, shape=(480, 640, 3)):
        self.frame = np.zeros(shape, dtype=np.uint8)
        self.seq = 0
        self.ts = 0.0
        self.dead = False
        self.opened = True

    def tick(self, now):
        if not self.dead:
            self.seq += 1
            self.ts = now

    def read(self):
        return (None, 0.0, self.seq) if self.dead and self.seq == 0 else (self.frame, self.ts, self.seq)

    def age(self, now=None):
        return (now or time.time()) - self.ts if self.ts else float("inf")


def boxes(*specs, shape=(480, 640)):
    """Detections from (class_name, cx, cy, size) or (class, cx, cy, size, colour)."""
    if not specs:
        return empty_detections()
    xyxy, names, conf = [], [], []
    for spec in specs:
        name, cx, cy, size = spec[:4]
        xyxy.append([cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2])
        names.append(name)
        conf.append(0.9)
    return sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        confidence=np.array(conf, dtype=np.float32),
        class_id=np.zeros(len(specs), dtype=int),
        data={"class_name": np.array(names, dtype=object)},
    )


class Rig:
    def __init__(self, tmp_path, attributes: bool = True):
        p = tmp_path / "models.yaml"
        p.write_text(MODELS_YAML)
        self.registry = Registry.from_yaml(p)
        self.registry.preload()
        self.detector = self.registry.get("fake")

        self.events, self.states = [], []
        ev_bus, st_bus = Bus(history=50), Bus(latest_only=True)
        ev_bus.subscribe_callback(self.events.append)
        st_bus.subscribe_callback(self.states.append)
        self.event_bus = ev_bus

        self.encoder = HashEncoder() if attributes else None
        self.machine = Machine(emit=ev_bus.publish)
        self.machine.on_registry_ready()
        self.capture = FakeCapture()
        self.builder = Builder(self.registry, default_model="fake",
                               encoder=self.encoder).start()
        shared = Shared()
        shared.bank.encoder = self.encoder
        shared.bank.ttl = 0.0  # re-score every frame; tests move time by hand
        self.loop = InferenceLoop(self.capture, self.builder, self.machine,
                                  st_bus, ev_bus, shared)
        self.now = 1000.0

    def close(self):
        self.builder.stop()

    # 1. Driving -------------------------------------------------------
    def paint(self, *colours):
        """Colour the frame so the attribute encoder has something to read."""
        h, w = self.capture.frame.shape[:2]
        if not colours:
            return
        band = w // len(colours)
        for i, c in enumerate(colours):
            self.capture.frame[:, i * band:(i + 1) * band] = BGR[c]

    def frames(self, n, seen=None, dt=0.05):
        for _ in range(n):
            if seen is not None:
                self.detector.set_script([seen])
            self.now += dt
            self.capture.tick(self.now)
            self.loop.step(self.now)

    def settle(self, timeout=3.0):
        """Step frames until the builder has handed everything over."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.frames(1)
            if (self.builder._q.empty()
                    and self.machine.phase not in (Phase.SWITCHING, Phase.LOADING_MODEL)):
                return
            time.sleep(0.004)
        raise AssertionError(f"stuck in {self.machine.phase}")

    # 2. Behaviours ----------------------------------------------------
    def add(self, behavior_id, kind="highlight", detect=("thing",), **subject):
        from contracts import Behavior

        spec = Behavior(behavior_id=behavior_id, kind=kind,
                        subject={"detect": list(detect), **subject},
                        select=subject.pop("select", "largest"))
        self.builder.add_behavior(f"i-{behavior_id}", spec)
        return spec

    def remove(self, behavior_id):
        self.builder.remove_behavior(f"d-{behavior_id}", behavior_id)

    # 3. Reading -------------------------------------------------------
    @property
    def stages(self):
        return [getattr(e, "stage", None) for e in self.events if hasattr(e, "stage")]

    @property
    def fired(self):
        return [e for e in self.events if hasattr(e, "kind")]

    @property
    def last(self):
        return self.states[-1] if self.states else None

    def status(self, behavior_id):
        s = self.last
        if not s:
            return None
        return next((b for b in s.behaviors if b.behavior_id == behavior_id), None)
