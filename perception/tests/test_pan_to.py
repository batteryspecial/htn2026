"""Pan odometry and the pan_to behaviour.

The odometry is tested against synthetic pans where the true answer is known,
because "did it turn 45 degrees" has no ground truth on real footage without an
IMU. The behaviour is then tested on top of that.
"""

import cv2
import numpy as np
import pytest

from actuator.odometry import PanOdometry
from behaviors.base import Frame, Outcome
from behaviors.kinds import build
from contracts import BehaviorSpec
from render.layers import Alert, Arrow
from tests.rig import Rig, boxes


def scene(seed: int = 0, w: int = 640, h: int = 480) -> np.ndarray:
    """A textured frame. Phase correlation needs detail to lock onto; a flat
    wall genuinely has no measurable motion."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (h, w * 3, 3), dtype=np.uint8)
    return cv2.GaussianBlur(img, (9, 9), 0)


def pan(panorama: np.ndarray, x: int, w: int = 640) -> np.ndarray:
    """A 640px window onto a wide scene, at horizontal offset x."""
    return panorama[:, x:x + w]


# 1. The measurement ------------------------------------------------------
def test_a_still_camera_reports_no_turn():
    """The failure that matters most: integrating noise until a stationary
    camera claims to have turned."""
    odo = PanOdometry(hfov_deg=70)
    frame = pan(scene(), 300)
    for _ in range(60):
        odo.update(frame)
    assert abs(odo.yaw_deg) < 0.5


def test_panning_right_reads_positive():
    odo = PanOdometry(hfov_deg=70)
    for x in range(200, 440, 8):
        odo.update(pan(scene(), x))
    assert odo.yaw_deg > 5


def test_panning_left_reads_negative():
    odo = PanOdometry(hfov_deg=70)
    for x in range(440, 200, -8):
        odo.update(pan(scene(), x))
    assert odo.yaw_deg < -5


def test_the_measured_angle_matches_the_geometry():
    """Content sliding a full frame width is one whole field of view."""
    odo = PanOdometry(hfov_deg=70)
    panorama = scene()
    for x in range(0, 640 + 1, 8):
        odo.update(pan(panorama, x))
    # 640px of a 640px-wide frame == 70 degrees, within integration error.
    assert 60 < odo.yaw_deg < 80, f"got {odo.yaw_deg:.1f}"


def test_the_field_of_view_scales_the_answer():
    """hfov_deg is the calibration knob when the angle reads short or long."""
    panorama = scene()
    narrow, wide = PanOdometry(hfov_deg=35), PanOdometry(hfov_deg=70)
    for x in range(0, 320, 8):
        frame = pan(panorama, x)
        narrow.update(frame)
        wide.update(frame)
    assert wide.yaw_deg == pytest.approx(2 * narrow.yaw_deg, rel=0.05)


def test_a_scene_cut_is_ignored_rather_than_integrated():
    """A wrong reading is worse than a missing one."""
    odo = PanOdometry(hfov_deg=70)
    odo.update(pan(scene(0), 300))
    odo.update(pan(scene(0), 300))
    before = odo.yaw_deg
    for _ in range(3):
        odo.update(scene(99)[:, :640])   # a completely different scene
    assert abs(odo.yaw_deg - before) < 3.0
    assert odo.rejected > 0


def test_reset_starts_measuring_from_here():
    odo = PanOdometry(hfov_deg=70)
    panorama = scene()
    for x in range(200, 400, 8):
        odo.update(pan(panorama, x))
    assert odo.yaw_deg != 0
    odo.reset()
    assert odo.yaw_deg == 0 and odo.samples == 0


def test_a_featureless_wall_does_not_invent_motion():
    odo = PanOdometry(hfov_deg=70)
    flat = np.full((480, 640, 3), 120, np.uint8)
    for _ in range(30):
        odo.update(flat)
    assert abs(odo.yaw_deg) < 1.0


def test_odometry_is_cheap_enough_for_the_loop():
    import time

    odo = PanOdometry()
    panorama = scene()
    frames = [pan(panorama, x) for x in range(0, 240, 8)]
    for f in frames:
        odo.update(f)
    t = time.perf_counter()
    for f in frames:
        odo.update(f)
    per_frame = (time.perf_counter() - t) / len(frames)
    assert per_frame < 0.010, f"{per_frame * 1000:.1f}ms per frame"


# 2. The behaviour --------------------------------------------------------
@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.close()


def pan_to(rig, **params):
    return rig.builder.add_behavior(BehaviorSpec(
        kind="pan_to", subject={"detect": ["thing"]},
        params=params, render={"label": "turn"}))


def drive(rig, from_x, to_x, step=8):
    """Feed the rig a real pan by moving a window across a wide scene."""
    panorama = scene()
    rng = range(from_x, to_x, step if to_x > from_x else -step)
    for x in rng:
        rig.capture.frame = pan(panorama, x).copy()
        rig.frames(1, seen=boxes(("thing", 320, 240, 80)))


def layers(rig, cls):
    return [l for l in rig.loop.view.layers if isinstance(l, cls)]


def test_a_turn_starts_guiding(rig):
    b = pan_to(rig, deg=-45)
    rig.settle()
    rig.frames(2, seen=boxes())
    assert rig.state_of(b) == "GUIDING"


def test_guiding_asks_for_a_turn_the_right_way(rig):
    b = pan_to(rig, deg=-45)
    rig.settle()
    rig.frames(2, seen=boxes())
    assert rig.motor.last.rate_deg_s < 0
    assert "left" in rig.motor.last.reason


def test_a_rightward_target_asks_to_pan_right(rig):
    b = pan_to(rig, deg=45)
    rig.settle()
    rig.frames(2, seen=boxes())
    assert rig.motor.last.rate_deg_s > 0 and "right" in rig.motor.last.reason


def test_an_arrow_is_drawn_with_the_degrees_left(rig):
    pan_to(rig, deg=-45)
    rig.settle()
    rig.frames(2, seen=boxes())
    arrows = layers(rig, Arrow)
    assert arrows and arrows[0].direction == "left"
    assert "°" in arrows[0].label


def test_turning_far_enough_reaches_the_target(rig):
    """The whole item: the operator swings the camera and the pipeline knows
    when to stop them."""
    b = pan_to(rig, deg=-45, hfov_deg=70)
    rig.settle()
    drive(rig, 640, 200)
    assert rig.state_of(b) == "REACHED"
    assert "reached" in rig.types()


def test_turning_the_wrong_way_does_not_reach(rig):
    b = pan_to(rig, deg=-45, hfov_deg=70)
    rig.settle()
    drive(rig, 200, 640)
    assert rig.state_of(b) == "GUIDING"
    assert "reached" not in rig.types()


def test_the_remaining_angle_counts_down(rig):
    b = pan_to(rig, deg=-45, hfov_deg=70)
    rig.settle()
    rig.frames(2, seen=boxes())
    start = abs(rig.view(b).data["remaining_deg"])
    drive(rig, 640, 420)
    assert abs(rig.view(b).data["remaining_deg"]) < start


def test_reaching_stops_asking_for_a_turn(rig):
    """An arrow that keeps pointing after you have arrived is worse than none."""
    b = pan_to(rig, deg=-45, hfov_deg=70)
    rig.settle()
    drive(rig, 640, 200)
    assert rig.state_of(b) == "REACHED"
    assert not layers(rig, Arrow)
    assert layers(rig, Alert)


def test_the_count_is_visible_while_turning(rig):
    """Demo 4 is "turn and count", so the tally builds during the pan."""
    b = pan_to(rig, deg=-45)
    rig.settle()
    rig.frames(3, seen=boxes(("thing", 200, 200, 60), ("thing", 400, 300, 60)))
    assert rig.view(b).data["matches"] == 2


@pytest.mark.parametrize("params", [
    {}, {"deg": 0}, {"deg": "left"}, {"deg": 45, "hfov_deg": 0},
])
def test_a_malformed_turn_is_refused_before_it_starts(params):
    with pytest.raises(ValueError):
        build("x", BehaviorSpec(kind="pan_to", subject={"detect": ["thing"]},
                                params=params))
