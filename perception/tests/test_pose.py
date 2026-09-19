"""Gesture triggers, and the aux-model plumbing behind them.

The pose model is the first aux model, so these also pin the rules that every
later one inherits: it runs only when something wants it, its failure degrades
one behaviour rather than the frame, and a behaviour whose model is missing is
paused with a reason instead of sitting there looking healthy.
"""

import numpy as np
import pytest

from behaviors.kinds import build
from contracts import BehaviorSpec
from render.layers import Alert
from skills.pose import (
    L_SHOULDER,
    L_WRIST,
    R_SHOULDER,
    R_WRIST,
    detect_gesture,
    hand_raised,
    person_box,
)
from tests.rig import Rig, boxes

PERSON = ("person", 320, 240, 120)


def body(raised: str | None = None) -> np.ndarray:
    """One COCO skeleton. Wrists start below the shoulders (y grows down)."""
    kp = np.zeros((17, 3), dtype=np.float32)
    kp[:, 2] = 0.9
    kp[L_SHOULDER] = (280, 200, 0.9)
    kp[R_SHOULDER] = (360, 200, 0.9)
    kp[L_WRIST] = (270, 320, 0.9)
    kp[R_WRIST] = (370, 320, 0.9)
    if raised in ("left", "both"):
        kp[L_WRIST] = (270, 120, 0.9)
    if raised in ("right", "both"):
        kp[R_WRIST] = (370, 120, 0.9)
    return kp


# 1. The gesture rule ------------------------------------------------------
def test_a_hand_below_the_shoulder_is_not_raised():
    assert hand_raised(body()) is False


@pytest.mark.parametrize("side", ["left", "right", "both"])
def test_a_hand_above_the_shoulder_is_raised(side):
    assert hand_raised(body(side)) is True


def test_the_rule_compares_against_the_persons_own_shoulder():
    """So it works for someone sitting, standing, near or far, uncalibrated."""
    small = body("left")
    small[:, :2] *= 0.25          # same pose, quarter the size
    assert hand_raised(small) is True


def test_an_unconfident_keypoint_is_not_believed():
    kp = body("left")
    kp[L_WRIST][2] = 0.1
    assert hand_raised(kp) is False


def test_gesture_detection_returns_who():
    people = np.stack([body(), body("left"), body()])
    assert detect_gesture(people, "hand_raised") == [1]


def test_an_unknown_gesture_is_refused():
    with pytest.raises(ValueError):
        detect_gesture(np.stack([body()]), "backflip")


def test_a_box_is_drawn_round_the_confident_points():
    box = person_box(body())
    assert box is not None and box[2] > box[0] and box[3] > box[1]


def test_a_body_with_nothing_confident_has_no_box():
    kp = body()
    kp[:, 2] = 0.0
    assert person_box(kp) is None


# 2. The behaviour ---------------------------------------------------------
@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path, attributes=True)
    yield r
    r.close()


class FakePose:
    """Stands in for the pose model, so the behaviour is testable with no
    weights and no GPU."""

    role = "pose"
    is_loaded = True

    def __init__(self):
        self.people = np.zeros((0, 17, 3), dtype=np.float32)
        self.calls = 0

    def load(self):
        pass

    def keypoints(self, frame):
        self.calls += 1
        return self.people


def with_pose(rig):
    """Give the rig a pose model the way the registry would."""
    pose = FakePose()
    rig.registry._entries["pose"] = type(rig.registry.entry("fake"))(
        name="pose", role="pose", type="fake")
    rig.registry._entries["pose"].model = pose
    rig.registry._active["pose"] = "pose"
    rig.loop.registry = rig.registry
    return pose


def gesture(rig, **params):
    return rig.builder.add_behavior(BehaviorSpec(
        kind="pose_trigger", subject={"detect": ["person"]},
        params=params, render={"label": "hands"}))


def test_a_raised_hand_fires(rig):
    pose = with_pose(rig)
    b = gesture(rig, hold_frames=2)
    rig.settle()
    pose.people = np.stack([body("left")])
    rig.frames(4, seen=boxes(PERSON))
    assert "hand_raised" in rig.types()


def test_a_lowered_hand_does_not_fire(rig):
    pose = with_pose(rig)
    gesture(rig, hold_frames=2)
    rig.settle()
    pose.people = np.stack([body()])
    rig.frames(6, seen=boxes(PERSON))
    assert "hand_raised" not in rig.types()


def test_a_hand_passing_through_does_not_fire(rig):
    """A hand on its way somewhere else is not a raised hand."""
    pose = with_pose(rig)
    gesture(rig, hold_frames=4)
    rig.settle()
    pose.people = np.stack([body("left")])
    rig.frames(2, seen=boxes(PERSON))
    pose.people = np.stack([body()])
    rig.frames(4, seen=boxes(PERSON))
    assert "hand_raised" not in rig.types()


def test_one_raised_hand_is_one_alert(rig):
    pose = with_pose(rig)
    gesture(rig, hold_frames=2, cooldown_s=5.0)
    rig.settle()
    pose.people = np.stack([body("left")])
    rig.frames(40, seen=boxes(PERSON))
    assert rig.types().count("hand_raised") == 1


def test_the_alert_carries_a_picture(rig):
    pose = with_pose(rig)
    gesture(rig, hold_frames=2)
    rig.settle()
    pose.people = np.stack([body("left")])
    rig.frames(4, seen=boxes(PERSON))
    ev = next(e for e in rig.fired if e.type == "hand_raised")
    assert ev.snapshot_url and rig.loop.snapshots.get(ev.id, b"")[:2] == b"\xff\xd8"


def test_firing_flashes_the_frame(rig):
    pose = with_pose(rig)
    gesture(rig, hold_frames=2)
    rig.settle()
    pose.people = np.stack([body("left")])
    rig.frames(4, seen=boxes(PERSON))
    assert [l for l in rig.loop.view.layers if isinstance(l, Alert)]


def test_the_count_of_people_is_reported(rig):
    pose = with_pose(rig)
    b = gesture(rig)
    rig.settle()
    pose.people = np.stack([body(), body("left"), body()])
    rig.frames(3, seen=boxes(PERSON))
    assert rig.view(b).data["people"] == 3
    assert rig.view(b).data["gesturing"] == 1


# 3. Aux-model plumbing ----------------------------------------------------
def test_the_pose_model_runs_only_when_something_wants_it(rig):
    """A pipeline with no gesture behaviour pays nothing for one being
    possible."""
    pose = with_pose(rig)
    rig.add(detect=("person",))          # an ordinary highlight
    rig.settle()
    rig.frames(6, seen=boxes(PERSON))
    assert pose.calls == 0

    gesture(rig)
    rig.settle()
    rig.frames(3, seen=boxes(PERSON))
    assert pose.calls > 0


def test_a_broken_pose_model_does_not_lose_the_frame(rig, monkeypatch):
    pose = with_pose(rig)
    b = gesture(rig)
    rig.settle()
    monkeypatch.setattr(type(pose), "keypoints",
                        lambda self, f: (_ for _ in ()).throw(RuntimeError("cuda oom")))
    before = rig.loop.timings.frames
    rig.frames(5, seen=boxes(PERSON))
    assert rig.loop.timings.frames > before, "the frame loop stopped"
    assert rig.state_of(b) == "ACTIVE"


def test_a_gesture_behavior_with_no_pose_model_is_paused(rig):
    """Not left looking healthy while silently doing nothing."""
    b = gesture(rig)
    rig.settle()
    rig.frames(3, seen=boxes(PERSON))
    assert rig.state_of(b) == "PAUSED"
    assert "pose" in (rig.view(b).detail or "")


def test_it_resumes_when_a_pose_model_appears(rig):
    b = gesture(rig)
    rig.settle()
    assert rig.state_of(b) == "PAUSED"

    with_pose(rig)
    rig.builder.clear()                   # any op rebuilds and revalidates
    rig.settle()
    b2 = gesture(rig)
    rig.settle()
    rig.frames(3, seen=boxes(PERSON))
    assert rig.state_of(b2) == "ACTIVE"


@pytest.mark.parametrize("params", [{"gesture": "backflip"}, {"gesture": ""}])
def test_an_unknown_gesture_is_refused_before_it_starts(params):
    with pytest.raises(ValueError):
        build("x", BehaviorSpec(kind="pose_trigger",
                                subject={"detect": ["person"]}, params=params))
