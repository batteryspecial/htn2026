"""Finger gestures, and the two-aux-model wiring behind them.

None of this needs mediapipe installed: the rules are pure geometry over an
array, and the array is the contract with the model. What mediapipe produces
is only verifiable on a machine that has it and a camera.
"""

import numpy as np
import pytest

from contracts import BehaviorSpec
from skills import gestures
from skills.hands import (
    FINGERS,
    N_LANDMARKS,
    WRIST,
    fist,
    index_finger_raised,
    open_palm,
)
from tests.rig import Rig, boxes

PERSON = ("person", 320, 240, 120)


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.close()


def hand(extended=("index", "middle", "ring", "pinky"), flip=False) -> np.ndarray:
    """One hand, fingers either straight or curled.

    Laid out along +y from a wrist at the origin: the pip joint sits at 10 and
    an extended tip at 20, a curled one back at 5. `flip` points the whole
    hand the other way, which must change nothing — the rules measure distance
    from the wrist, not direction.
    """
    kp = np.zeros((N_LANDMARKS, 3), dtype=np.float32)
    kp[WRIST] = (0.0, 0.0, 1.0)
    sign = -1.0 if flip else 1.0
    for finger, (mcp, pip, tip) in FINGERS.items():
        kp[mcp] = (0.0, sign * 5.0, 1.0)
        kp[pip] = (0.0, sign * 10.0, 1.0)
        kp[tip] = (0.0, sign * (20.0 if finger in extended else 5.0), 1.0)
    return kp


# 1. The rules ------------------------------------------------------------
def test_an_index_finger_alone_is_the_gesture():
    assert index_finger_raised(hand(extended=("index",))) is True


def test_an_open_palm_is_not_a_raised_index_finger():
    """The reason the rule checks the other three fingers at all: an open palm
    has an extended index, so without them a wave fires the finger guard."""
    assert index_finger_raised(hand()) is False
    assert open_palm(hand()) is True


def test_a_fist_is_neither():
    closed = hand(extended=())
    assert fist(closed) is True
    assert index_finger_raised(closed) is False
    assert open_palm(closed) is False


@pytest.mark.parametrize("flip", [False, True])
def test_orientation_does_not_matter(flip):
    """A hand pointing down is still a hand. Distance from the wrist holds in
    any orientation; a tip-above-joint test would only hold in one."""
    assert index_finger_raised(hand(extended=("index",), flip=flip)) is True
    assert open_palm(hand(flip=flip)) is True


def test_the_thumb_is_never_consulted():
    """It folds sideways rather than curling toward the wrist, so the distance
    test does not describe it. Moving it must change no verdict."""
    palm = hand()
    before = (open_palm(palm), fist(palm), index_finger_raised(palm))
    palm[1:5] = (99.0, -99.0, 1.0)
    assert (open_palm(palm), fist(palm), index_finger_raised(palm)) == before


# 2. The vocabulary -------------------------------------------------------
def test_each_gesture_names_the_model_that_sees_it():
    assert gestures.ROLE_OF["hand_raised"] == "pose"
    assert gestures.ROLE_OF["index_finger_raised"] == "hands"


def test_available_narrows_to_what_is_loaded():
    """A rule whose model is missing is not a capability. This is what stops
    the agent installing a behaviour that can never fire."""
    assert gestures.available(lambda role: False) == []
    assert gestures.available(lambda role: True) == gestures.known()
    assert gestures.available(lambda role: role == "hands") == sorted(
        n for n, r in gestures.ROLE_OF.items() if r == "hands")


def test_an_unknown_gesture_says_what_is_known():
    with pytest.raises(ValueError, match="index_finger_raised"):
        gestures.detect(np.zeros((1, N_LANDMARKS, 3), dtype=np.float32), "snap")


def test_detect_returns_the_hands_doing_it():
    people = np.stack([hand(), hand(extended=("index",)), hand(extended=())])
    assert gestures.detect(people, "index_finger_raised") == [1]
    assert gestures.detect(people, "open_palm") == [0]
    assert gestures.detect(people, "fist") == [2]


# 3. Two aux models, end to end -------------------------------------------
class FakeHands:
    role = "hands"
    is_loaded = True

    def __init__(self):
        self.seen = np.zeros((0, N_LANDMARKS, 3), dtype=np.float32)

    def load(self):
        pass

    def keypoints(self, frame):
        return self.seen


def with_hands(rig):
    """Give the rig a hands model the way the registry would."""
    model = FakeHands()
    rig.registry._entries["hands"] = type(rig.registry.entry("fake"))(
        name="hands", role="hands", type="fake")
    rig.registry._entries["hands"].model = model
    rig.registry._active["hands"] = "hands"
    rig.loop.registry = rig.registry
    return model


def guard(rig, **params):
    return rig.builder.add_behavior(BehaviorSpec(
        kind="pose_trigger", subject={"detect": ["person"]},
        params=params, render={"label": "finger"}))


def test_a_raised_index_finger_fires(rig):
    model = with_hands(rig)
    guard(rig, gesture="index_finger_raised", hold_frames=2)
    rig.settle()
    model.seen = np.stack([hand(extended=("index",))])
    rig.frames(4, seen=boxes(PERSON))
    assert "index_finger_raised" in rig.types()


def test_a_finger_guard_without_a_hands_model_is_paused(rig):
    """The failure this whole model was added for.

    Before, `pose_trigger` declared `needs_roles = ("pose",)` for every
    gesture, so a finger guard armed as soon as a *body* model was loaded and
    then watched wrists forever. The role is now per gesture, so the guard is
    held with a reason instead of sitting there looking healthy.
    """
    b = guard(rig, gesture="index_finger_raised")
    rig.settle()
    rig.frames(3, seen=boxes(PERSON))
    assert rig.state_of(b) == "PAUSED"
    assert "hands" in (rig.view(b).detail or "")


def test_a_body_gesture_does_not_need_the_hands_model(rig):
    """The roles are independent in both directions: loading hands must not
    be what makes a raised hand work, nor the reverse."""
    with_hands(rig)
    b = guard(rig, gesture="hand_raised")
    rig.settle()
    rig.frames(3, seen=boxes(PERSON))
    assert rig.state_of(b) == "PAUSED"
    assert "pose" in (rig.view(b).detail or "")


def test_the_two_models_run_independently(rig):
    """Both loaded, a finger guard running: the loop must ask the hands model
    and not the body one, so neither gesture can shadow the other."""
    model = with_hands(rig)
    guard(rig, gesture="open_palm", hold_frames=2)
    rig.settle()
    model.seen = np.stack([hand()])
    rig.frames(4, seen=boxes(PERSON))
    assert "open_palm" in rig.types()
    assert "hand_raised" not in rig.types()
