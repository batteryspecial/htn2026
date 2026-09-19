"""A whole pipeline driven by hand, with no camera, no model and no CLIP.

Every test above this uses it, so what is under test is the real loop, the real
worker and the real behaviour code, with only the outside world faked.
"""

from __future__ import annotations

import time

import numpy as np
import supervision as sv

from actuator.virtual import VirtualMotor
from attributes.encoders import HashEncoder
from contracts import BehaviorSpec
from detectors.base import empty_detections
from detectors.registry import Registry
from runtime.events import Bus
from runtime.health import Health
from runtime.loop import InferenceLoop
from runtime.workers import Builder
from runtime.world import Shared

MODELS_YAML = (
    "models:\n"
    "  - name: fake\n    type: fake\n    preload: true\n"
    "  - name: other\n    type: fake\n    preload: true\n"
    "  - name: gpu_only\n    type: ultralytics_fixed\n"
    "    weights: nope.engine\n    requires_cuda: true\n"
)

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
        return (None, 0.0, self.seq) if self.dead and self.seq == 0 else (
            self.frame, self.ts, self.seq)

    def age(self, now=None):
        return (now or time.time()) - self.ts if self.ts else float("inf")


def boxes(*specs):
    """Detections from (class_name, cx, cy, size) tuples."""
    if not specs:
        return empty_detections()
    xyxy, names, conf = [], [], []
    for name, cx, cy, size in specs:
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
        # Start blind. Without this the detector invents boxes during settle()
        # and a track behaviour latches onto something no test asked for.
        self.detector.set_script([empty_detections()])

        self.events, self.states = [], []
        ev_bus, st_bus = Bus(history=50), Bus(latest_only=True)
        ev_bus.subscribe_callback(self.events.append)
        st_bus.subscribe_callback(self.states.append)
        self.event_bus, self.state_bus = ev_bus, st_bus

        self.encoder = HashEncoder() if attributes else None
        self.health = Health(emit=ev_bus.publish)
        self.capture = FakeCapture()
        self.builder = Builder(self.registry, default_model="fake",
                               encoder=self.encoder).start()
        shared = Shared()
        shared.bank.encoder = self.encoder
        shared.bank.ttl = 0.0  # re-score every frame; tests move time by hand
        self.motor = VirtualMotor()
        self.loop = InferenceLoop(self.capture, self.builder, self.health,
                                  st_bus, ev_bus, shared, actuator=self.motor)
        self.now = 1000.0

    def close(self):
        self.builder.stop()

    # 1. Driving -------------------------------------------------------
    def paint(self, *colours):
        """Colour the frame so the attribute encoder has something to read."""
        if not colours:
            return
        h, w = self.capture.frame.shape[:2]
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
        """Step frames until the worker's world is the one the loop is running.

        Never calls builder.poll(): that is the loop's queue, and draining it
        here would swallow the very outcome being waited for.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.frames(1)
            if self.builder.idle and self.loop.world is self.builder.world:
                return
            time.sleep(0.004)
        raise AssertionError("builder never settled")

    # 2. Behaviours ----------------------------------------------------
    def add(self, kind="highlight", detect=("thing",), **subject):
        params = subject.pop("params", {})
        render = subject.pop("render", {})
        spec = BehaviorSpec(kind=kind, subject={"detect": list(detect), **subject},
                            params=params, render=render)
        return self.builder.add_behavior(spec)

    def remove(self, behavior_id):
        self.builder.remove_behavior(behavior_id)

    # 3. Reading -------------------------------------------------------
    @property
    def fired(self):
        return self.events

    @property
    def last(self):
        return self.states[-1] if self.states else None

    def view(self, behavior_id):
        s = self.last
        if not s:
            return None
        return next((b for b in s.behaviors if b.id == behavior_id), None)

    def state_of(self, behavior_id):
        v = self.view(behavior_id)
        return v.state if v else None

    def types(self):
        return [e.type for e in self.events]

    def state_of_track_lost(self) -> bool:
        s = self.last
        return bool(s) and any(b.state in ("LOST", "SEARCHING") for b in s.behaviors)
