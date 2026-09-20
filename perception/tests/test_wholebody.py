"""Whole-body keypoints: the slicing contract and the gestures it enables.

rtmlib is not needed for any of this. The 133-point array is the contract with
the model, and everything here is geometry over that array. Whether RTMPose
fills it at a usable frame rate is only answerable on a machine that has it.
"""

import numpy as np
import pytest

from contracts import BehaviorSpec
from skills import gestures
from skills.hands import N_LANDMARKS, index_finger_raised, open_palm
from skills.pose import L_SHOULDER, L_WRIST, R_SHOULDER, R_WRIST, hand_raised
from skills.wholebody import (
    FACE_INNER_LIP_BOTTOM,
    FACE_INNER_LIP_TOP,
    FACE_LEFT_EYE_OUTER,
    FACE_RIGHT_EYE_OUTER,
    N_KEYPOINTS,
    body,
    both_hands_raised,
    left_hand,
    mouth_open,
    right_hand,
)
# The same builder the hands tests use, on purpose: proving the slice works
# means feeding it the array those rules were written against, not a new one
# shaped to agree with them.
from tests.test_hands import hand as hand_shape
from tests.rig import Rig, boxes

PERSON = ("person", 320, 240, 120)


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.close()


def person(mouth_gap: float = 0.0, wrists: str = "down") -> np.ndarray:
    """One 133-point person, confident everywhere.

    Eyes 40px apart, so `mouth_gap` is read directly as a fraction of
    interocular distance by the rule under test.
    """
    kp = np.zeros((N_KEYPOINTS, 3), dtype=np.float32)
    kp[:, 2] = 1.0

    # Body block, COCO order: shoulders at y=100, wrists above or below them.
    kp[L_SHOULDER] = (100.0, 100.0, 1.0)
    kp[R_SHOULDER] = (140.0, 100.0, 1.0)
    above, below = 60.0, 160.0
    kp[L_WRIST] = (100.0, above if wrists in ("both", "left") else below, 1.0)
    kp[R_WRIST] = (140.0, above if wrists in ("both", "right") else below, 1.0)

    # Face block: eyes 40 apart, lips `mouth_gap * 40` apart.
    kp[FACE_RIGHT_EYE_OUTER] = (100.0, 50.0, 1.0)
    kp[FACE_LEFT_EYE_OUTER] = (140.0, 50.0, 1.0)
    kp[FACE_INNER_LIP_TOP] = (120.0, 70.0, 1.0)
    kp[FACE_INNER_LIP_BOTTOM] = (120.0, 70.0 + mouth_gap * 40.0, 1.0)
    return kp


# 1. The slicing contract -------------------------------------------------
def test_the_blocks_are_the_sizes_the_layout_promises():
    p = person()
    assert body(p).shape == (17, 3)
    assert left_hand(p).shape == (N_LANDMARKS, 3)
    assert right_hand(p).shape == (N_LANDMARKS, 3)


def test_the_body_block_is_coco_so_pose_rules_read_it_unchanged():
    """The reason skills/wholebody.py slices instead of reimplementing. If the
    first 17 points ever stop being COCO-ordered, this is what says so."""
    assert hand_raised(body(person(wrists="down"))) is False
    assert hand_raised(body(person(wrists="left"))) is True


def test_the_hand_blocks_are_21_point_so_hand_rules_read_them_unchanged():
    """Same claim for the hands: COCO-WholeBody uses the topology
    skills/hands.py already assumes, so nothing is rewritten."""
    p = person()
    p[91:112] = hand_shape(extended=("index",))
    p[112:133] = hand_shape()
    assert index_finger_raised(left_hand(p)) is True
    assert open_palm(left_hand(p)) is False
    assert open_palm(right_hand(p)) is True


# 2. The gestures the extra points enable ---------------------------------
def test_an_open_mouth_is_measured_against_the_eyes():
    assert mouth_open(person(mouth_gap=0.0)) is False
    assert mouth_open(person(mouth_gap=0.8)) is True


def test_mouth_open_does_not_depend_on_distance_from_the_camera():
    """The raw lip gap in pixels means nothing: a closed mouth up close
    measures wider than an open one across the room."""
    near, far = person(mouth_gap=0.8), person(mouth_gap=0.8)
    far[:, :2] *= 0.25                      # same person, four times further
    assert mouth_open(near) is True
    assert mouth_open(far) is True


def test_an_unconfident_face_is_not_an_open_mouth():
    p = person(mouth_gap=0.8)
    p[FACE_INNER_LIP_BOTTOM][2] = 0.0
    assert mouth_open(p) is False


def test_both_hands_means_both():
    assert both_hands_raised(person(wrists="both")) is True
    assert both_hands_raised(person(wrists="left")) is False
    assert both_hands_raised(person(wrists="down")) is False


# 3. Wiring ---------------------------------------------------------------
def test_the_new_gestures_name_the_wholebody_model():
    assert gestures.ROLE_OF["mouth_open"] == "wholebody"
    assert gestures.ROLE_OF["both_hands_raised"] == "wholebody"
    # Unchanged: loading this model does not yet satisfy the others.
    assert gestures.ROLE_OF["hand_raised"] == "pose"
    assert gestures.ROLE_OF["index_finger_raised"] == "hands"


class FakeWholeBody:
    role = "wholebody"
    is_loaded = True

    def __init__(self):
        self.seen = np.zeros((0, N_KEYPOINTS, 3), dtype=np.float32)

    def load(self):
        pass

    def keypoints(self, frame):
        return self.seen


def with_wholebody(rig):
    model = FakeWholeBody()
    rig.registry._entries["wholebody"] = type(rig.registry.entry("fake"))(
        name="wholebody", role="wholebody", type="fake")
    rig.registry._entries["wholebody"].model = model
    rig.registry._active["wholebody"] = "wholebody"
    rig.loop.registry = rig.registry
    return model


def test_an_open_mouth_fires_end_to_end(rig):
    model = with_wholebody(rig)
    rig.builder.add_behavior(BehaviorSpec(
        kind="pose_trigger", subject={"detect": ["person"]},
        params={"gesture": "mouth_open", "hold_frames": 2}))
    rig.settle()
    model.seen = np.stack([person(mouth_gap=0.8)])
    rig.frames(4, seen=boxes(PERSON))
    assert "mouth_open" in rig.types()


def test_a_wholebody_gesture_without_the_model_is_paused(rig):
    b = rig.builder.add_behavior(BehaviorSpec(
        kind="pose_trigger", subject={"detect": ["person"]},
        params={"gesture": "mouth_open"}))
    rig.settle()
    rig.frames(3, seen=boxes(PERSON))
    assert rig.state_of(b) == "PAUSED"
    assert "wholebody" in (rig.view(b).detail or "")
