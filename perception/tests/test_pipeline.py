"""End-to-end tests of the running pipeline, with no model and no camera.

These drive the real loop, the real loader and the real state machine one frame
at a time, so the things being checked are the ones that actually fail on
stage: does a retask land, does a bad spec leave the previous target alone, and
does the car get told to stop when something breaks.
"""

import threading
import time

import numpy as np
import pytest
import supervision as sv

from contracts import Phase, TaskSpec
from detectors.base import empty_detections
from detectors.registry import Registry
from runtime.events import Bus
from runtime.loader import Loader
from runtime.loop import InferenceLoop
from runtime.state import Machine


# --- doubles -------------------------------------------------------------
class FakeCapture:
    """A camera we can freeze, kill and revive on demand."""

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
    """Build detections from (class_name, cx, cy, size) tuples."""
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
    """The whole pipeline, driven by hand."""

    def __init__(self, tmp_path):
        p = tmp_path / "models.yaml"
        p.write_text("models:\n  - name: fake\n    type: fake\n    preload: true\n"
                     "  - name: missing\n    type: ultralytics_fixed\n"
                     "    weights: nope.engine\n    requires_cuda: true\n")
        self.registry = Registry.from_yaml(p)
        self.registry.preload()
        self.detector = self.registry.get("fake")
        self.events, self.targets = [], []
        ev_bus, tg_bus = Bus(history=50), Bus(latest_only=True)
        ev_bus.subscribe_callback(self.events.append)
        tg_bus.subscribe_callback(self.targets.append)
        self.machine = Machine(emit=ev_bus.publish)
        self.machine.on_registry_ready()
        self.capture = FakeCapture()
        self.loader = Loader(self.registry).start()
        self.loop = InferenceLoop(self.capture, self.loader, self.machine, tg_bus, ev_bus)
        self.now = 1000.0

    def close(self):
        self.loader.stop()

    def spec(self, spec_id="s1", model="fake", **over):
        body = {"spec_id": spec_id, "model": model,
                "targets": [{"ref": "t1", "detect": ["pencil"]}]}
        body.update(over)
        return TaskSpec(**body)

    def send(self, instruction_id, spec):
        self.loader.submit(instruction_id, spec)

    def settle(self, timeout=3.0):
        """Advance frames until the loader has nothing left in flight."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.frames(1)
            if self.machine.phase not in (Phase.SWITCHING, Phase.LOADING_MODEL):
                return
        raise AssertionError(f"stuck in {self.machine.phase}")

    def frames(self, n, dt=0.05, seen=None):
        """Advance n frames. `seen` scripts what the detector reports."""
        for _ in range(n):
            if seen is not None:
                self.detector.set_script([seen])
            self.now += dt
            self.capture.tick(self.now)
            self.loop.step(self.now)

    @property
    def stages(self):
        return [e.stage for e in self.events]

    @property
    def last(self):
        return self.targets[-1] if self.targets else None


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.close()


PENCIL = ("pencil", 320, 240, 100)


# 1. The happy path -------------------------------------------------------
def test_a_spec_becomes_a_tracked_target(rig):
    rig.send("i1", rig.spec())
    rig.settle()
    rig.frames(5, seen=boxes(PENCIL))
    assert rig.machine.phase is Phase.TRACKING
    assert rig.last.visible and rig.last.label == "pencil"
    assert rig.stages[:3] == ["prepared", "applied", "active"]


def test_published_position_is_normalized_and_centered(rig):
    rig.send("i1", rig.spec())
    rig.settle()
    rig.frames(5, seen=boxes(PENCIL))
    s = rig.last
    assert abs(s.cx) < 0.01 and abs(s.cy) < 0.01
    assert 0 < s.area < 1


def test_offsets_have_the_sign_the_controller_expects(rig):
    rig.send("i1", rig.spec())
    rig.settle()
    rig.frames(5, seen=boxes(("pencil", 560, 400, 80)))
    # Right of centre and below it: both positive.
    assert rig.last.cx > 0 and rig.last.cy > 0


def test_every_frame_publishes_exactly_one_state(rig):
    rig.send("i1", rig.spec())
    rig.settle()
    before = len(rig.targets)
    rig.frames(10, seen=boxes(PENCIL))
    assert len(rig.targets) - before == 10


def test_every_published_state_carries_a_phase(rig):
    rig.send("i1", rig.spec())
    rig.settle()
    rig.frames(10, seen=boxes(PENCIL))
    assert all(s.phase is not None for s in rig.targets)


# 2. Retasking ------------------------------------------------------------
def test_retask_switches_the_target(rig):
    rig.send("i1", rig.spec(targets=[{"ref": "t1", "detect": ["pencil"]}]))
    rig.settle()
    rig.frames(5, seen=boxes(PENCIL, ("eraser", 100, 100, 60)))
    assert rig.last.label == "pencil"

    rig.send("i2", rig.spec(spec_id="s2", targets=[{"ref": "t1", "detect": ["eraser"]}]))
    rig.settle()
    rig.frames(5, seen=boxes(PENCIL, ("eraser", 100, 100, 60)))
    assert rig.last.label == "eraser"
    assert rig.last.spec_id == "s2"


def test_retask_resets_tracking_history(rig):
    """Make the tracker hand out several ids, then retask and watch it restart.

    Identity must not survive a retask: the operator asked for a different
    target, so carrying the old one's history forward is the wrong answer.
    """
    rig.send("i1", rig.spec())
    rig.settle()
    for _ in range(3):
        rig.frames(6, seen=boxes(PENCIL))
        rig.frames(40, seen=boxes())
    assert rig.last.track_id is None
    climbed = max(s.track_id for s in rig.targets if s.track_id)
    assert climbed > 1, "tracker never reissued an id, so the test proves nothing"

    rig.send("i2", rig.spec(spec_id="s2"))
    rig.settle()
    rig.frames(5, seen=boxes(PENCIL))
    assert rig.last.track_id == 1


def test_retask_forgets_a_locked_target(rig):
    """`locked` latches onto one instance. A new spec must drop the latch."""
    locked = {"ref": "t1", "detect": ["pencil"], "select": "locked"}
    rig.send("i1", rig.spec(targets=[locked]))
    rig.settle()
    # Cycle the target so the latch ends up on an id the fresh tracker cannot
    # reissue, which is what makes the next assertion mean something.
    for _ in range(3):
        rig.frames(6, seen=boxes(PENCIL))
        rig.frames(40, seen=boxes())
    assert rig.loop.active.select_states["t1"].locked_track_id > 1

    rig.send("i2", rig.spec(spec_id="s2", targets=[locked]))
    rig.settle()
    rig.frames(5, seen=boxes(PENCIL))
    assert rig.loop.active.select_states["t1"].locked_track_id == 1


def test_car_is_told_to_stop_during_a_retask(rig):
    rig.send("i1", rig.spec())
    rig.settle()
    rig.frames(5, seen=boxes(PENCIL))
    assert rig.machine.phase is Phase.TRACKING

    rig.send("i2", rig.spec(spec_id="s2"))
    rig.frames(1, seen=boxes(PENCIL))
    # Between accepting and acquiring, TRACKING must not be published.
    assert rig.machine.phase is not Phase.TRACKING


def test_an_impossible_spec_leaves_the_current_target_alone(rig):
    rig.send("i1", rig.spec())
    rig.settle()
    rig.frames(5, seen=boxes(PENCIL))

    rig.send("i2", rig.spec(spec_id="s2", model="missing"))
    rig.settle()
    rig.frames(5, seen=boxes(PENCIL))

    assert "failed" in rig.stages
    assert rig.machine.phase is Phase.TRACKING
    assert rig.last.spec_id == "s1"  # still the old spec


def test_only_the_last_of_a_burst_of_specs_is_applied(rig):
    for i in range(4):
        rig.send(f"i{i}", rig.spec(spec_id=f"s{i}"))
    rig.settle()
    rig.frames(5, seen=boxes(PENCIL))
    assert rig.last.spec_id == "s3"


def test_superseded_specs_are_reported_against_their_own_instruction(rig):
    for i in range(3):
        rig.send(f"i{i}", rig.spec(spec_id=f"s{i}"))
    rig.settle()
    failed = [e for e in rig.events if e.stage == "failed"]
    assert {e.instruction_id for e in failed} <= {"i0", "i1"}
    assert all(e.detail == "superseded" for e in failed)


def test_acquired_fires_once_per_spec_not_once_per_glimpse(rig):
    rig.send("i1", rig.spec())
    rig.settle()
    rig.frames(5, seen=boxes(PENCIL))
    rig.frames(10, seen=boxes())
    rig.frames(5, seen=boxes(PENCIL))
    assert rig.stages.count("active") == 1


# 3. Losing the target ----------------------------------------------------
def test_a_brief_dropout_does_not_stop_the_car(rig):
    rig.send("i1", rig.spec())
    rig.settle()
    rig.frames(5, seen=boxes(PENCIL))
    rig.frames(3, seen=boxes())
    assert rig.machine.phase is Phase.TRACKING


def test_a_sustained_dropout_does_stop_the_car(rig):
    rig.send("i1", rig.spec())
    rig.settle()
    rig.frames(5, seen=boxes(PENCIL))
    rig.frames(6, seen=boxes())
    assert rig.machine.phase is Phase.LOST
    assert rig.last.visible is False


def test_never_acquiring_reports_no_target(rig):
    rig.send("i1", rig.spec())
    rig.settle()
    rig.frames(80, seen=boxes())
    assert rig.machine.phase is Phase.NO_TARGET
    assert "no_target" in rig.stages


# 4. Broken hardware ------------------------------------------------------
def test_a_dead_camera_faults_and_keeps_talking(rig):
    rig.send("i1", rig.spec())
    rig.settle()
    rig.frames(5, seen=boxes(PENCIL))

    rig.capture.dead = True
    before = len(rig.targets)
    for _ in range(40):
        rig.now += 0.1
        rig.loop.step(rig.now)

    assert rig.machine.phase is Phase.FAULT
    assert len(rig.targets) > before, "a silent pipeline is indistinguishable from a dead one"
    assert all(not s.visible for s in rig.targets[before:])


def test_fault_heartbeat_is_slow_not_a_flood(rig):
    rig.send("i1", rig.spec())
    rig.settle()
    rig.frames(5, seen=boxes(PENCIL))
    rig.capture.dead = True
    before = len(rig.targets)
    for _ in range(100):  # 10 s of stepping at 100 Hz
        rig.now += 0.1
        rig.loop.step(rig.now)
    published = len(rig.targets) - before
    assert 10 <= published <= 80


def test_the_camera_coming_back_clears_the_fault(rig):
    rig.send("i1", rig.spec())
    rig.settle()
    rig.frames(5, seen=boxes(PENCIL))
    rig.capture.dead = True
    for _ in range(40):
        rig.now += 0.1
        rig.loop.step(rig.now)
    assert rig.machine.phase is Phase.FAULT

    rig.capture.dead = False
    rig.frames(5, seen=boxes(PENCIL))
    assert rig.machine.phase is Phase.TRACKING


def test_a_broken_model_faults_rather_than_crashing_the_loop(rig, monkeypatch):
    rig.send("i1", rig.spec())
    rig.settle()
    rig.frames(5, seen=boxes(PENCIL))

    def explode(self, frame):
        raise RuntimeError("CUDA out of memory")
    monkeypatch.setattr(type(rig.detector), "infer", explode)

    before = len(rig.targets)
    rig.frames(10)
    assert rig.machine.phase is Phase.FAULT
    assert len(rig.targets) > before


def test_inference_recovering_clears_the_fault(rig, monkeypatch):
    rig.send("i1", rig.spec())
    rig.settle()
    original = type(rig.detector).infer

    def explode(self, frame):
        raise RuntimeError("transient")
    monkeypatch.setattr(type(rig.detector), "infer", explode)
    rig.frames(5)
    assert rig.machine.phase is Phase.FAULT

    monkeypatch.setattr(type(rig.detector), "infer", original)
    rig.frames(5, seen=boxes(PENCIL))
    assert rig.machine.phase is Phase.TRACKING


# 5. Two targets ----------------------------------------------------------
def test_two_targets_alternate(rig):
    spec = rig.spec(
        targets=[{"ref": "a", "detect": ["pencil"]}, {"ref": "b", "detect": ["eraser"]}],
        arbitration={"alternate_s": 0.2},
    )
    rig.send("i1", spec)
    rig.settle()
    rig.frames(30, seen=boxes(PENCIL, ("eraser", 100, 100, 60)))
    refs = {s.target_ref for s in rig.targets[-25:]}
    assert refs == {"a", "b"}


def test_a_missing_target_yields_its_turn_early(rig):
    spec = rig.spec(
        targets=[{"ref": "a", "detect": ["pencil"]}, {"ref": "b", "detect": ["eraser"]}],
        arbitration={"alternate_s": 60.0},  # would never flip on its own
    )
    rig.send("i1", spec)
    rig.settle()
    rig.frames(40, seen=boxes(("eraser", 100, 100, 60)))
    assert rig.last.target_ref == "b"
    assert rig.last.visible
