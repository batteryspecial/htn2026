"""watch, count_line and privacy.

The three behaviours that turn one camera into a sentinel, a traffic counter
and a privacy camera. Between them they cover six items on the demo list, and
none of them needed new machinery — they are combinations of the selector, the
trigger and the render primitives.
"""

import numpy as np
import pytest

from behaviors.base import Frame, UnsupportedBehavior
from behaviors.kinds import build
from contracts import BehaviorSpec
from render.layers import Alert, Blur, Line, Zone
from tests.rig import Rig, boxes

DUCK = ("duck", 320, 240, 80)
PERSON_FAR = ("person", 600, 420, 70)
PERSON_NEAR = ("person", 380, 260, 70)


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.close()


def watch(rig, triggers, detect=("duck",), **params):
    spec = BehaviorSpec(kind="watch", subject={"detect": list(detect)},
                        params={"triggers": triggers, **params},
                        render={"label": "guard"})
    return rig.builder.add_behavior(spec)


def layers_of(rig, cls):
    return [l for l in rig.loop.view.layers if isinstance(l, cls)]


# 1. Arming ---------------------------------------------------------------
def test_a_guard_arms_only_once_the_subject_is_steady(rig):
    """It learns where the thing is before judging that it moved."""
    b = watch(rig, [{"type": "missing", "after_s": 1.0}])
    rig.settle()
    rig.frames(3, seen=boxes(DUCK), dt=0.1)
    assert rig.state_of(b) == "ARMING"
    rig.frames(12, seen=boxes(DUCK), dt=0.1)
    assert rig.state_of(b) == "ARMED"
    assert "armed" in rig.types()


def test_a_guard_will_not_arm_on_something_it_cannot_see(rig):
    b = watch(rig, [{"type": "missing", "after_s": 1.0}])
    rig.settle()
    rig.frames(20, seen=boxes(), dt=0.1)
    assert rig.state_of(b) == "ARMING"


def test_arming_records_where_the_subject_was(rig):
    b = watch(rig, [{"type": "moved", "min_shift": 0.1}])
    rig.settle()
    rig.frames(15, seen=boxes(DUCK), dt=0.1)
    assert rig.view(b).data["baseline"] == [0.0, 0.0]


# 2. Triggers -------------------------------------------------------------
def test_missing_fires_when_the_guarded_thing_goes(rig):
    """"Guard the table" is this, once per object."""
    b = watch(rig, [{"type": "missing", "after_s": 0.5}])
    rig.settle()
    rig.frames(15, seen=boxes(DUCK), dt=0.1)
    rig.frames(12, seen=boxes(), dt=0.1)
    assert "missing" in rig.types()
    assert rig.state_of(b) in ("FIRED", "COOLDOWN")


def test_moved_fires_only_past_the_threshold(rig):
    b = watch(rig, [{"type": "moved", "min_shift": 0.3}])
    rig.settle()
    rig.frames(15, seen=boxes(DUCK), dt=0.1)
    rig.frames(5, seen=boxes(("duck", 340, 240, 80)), dt=0.1)
    assert "moved" not in rig.types()
    rig.frames(5, seen=boxes(("duck", 560, 240, 80)), dt=0.1)
    assert "moved" in rig.types()


def test_near_fires_when_something_else_approaches(rig):
    """The sentinel: "tell me if anyone goes near the duck"."""
    b = watch(rig, [{"type": "near", "other": {"detect": ["person"]}, "margin": 0.05}])
    rig.settle()
    rig.frames(15, seen=boxes(DUCK, PERSON_FAR), dt=0.1)
    assert "near" not in rig.types()
    rig.frames(4, seen=boxes(DUCK, PERSON_NEAR), dt=0.1)
    assert "near" in rig.types()


def test_an_excluded_person_does_not_trip_the_guard(rig):
    """"Anyone in a red hoodie is authorised" is an exclude on the trigger's
    own selector, not a separate mechanism."""
    rig.paint("red")
    b = watch(rig, [{"type": "near",
                     "other": {"detect": ["person"], "exclude": ["red"]},
                     "margin": 0.05}])
    rig.settle()
    rig.frames(20, seen=boxes(DUCK, PERSON_NEAR), dt=0.1)
    assert "near" not in rig.types(), "an authorised person set off the alarm"


def test_appeared_fires_on_anything_showing_up(rig):
    b = watch(rig, [{"type": "appeared", "other": {"detect": ["person"]}}])
    rig.settle()
    rig.frames(15, seen=boxes(DUCK), dt=0.1)
    assert "appeared" not in rig.types()
    rig.frames(3, seen=boxes(DUCK, PERSON_FAR), dt=0.1)
    assert "appeared" in rig.types()


# 3. Rate limiting --------------------------------------------------------
def test_a_guard_does_not_fire_thirty_times_a_second(rig):
    """The agent is rate-limited: one alert that arrives beats a hundred
    that get dropped."""
    b = watch(rig, [{"type": "near", "other": {"detect": ["person"]}, "margin": 0.05}],
              cooldown_s=2.0)
    rig.settle()
    rig.frames(15, seen=boxes(DUCK), dt=0.1)
    rig.frames(60, seen=boxes(DUCK, PERSON_NEAR), dt=0.1)
    assert rig.types().count("near") <= 4


def test_a_fired_guard_returns_to_watching(rig):
    b = watch(rig, [{"type": "near", "other": {"detect": ["person"]}, "margin": 0.05}],
              cooldown_s=1.0)
    rig.settle()
    rig.frames(15, seen=boxes(DUCK), dt=0.1)
    rig.frames(4, seen=boxes(DUCK, PERSON_NEAR), dt=0.1)
    assert rig.state_of(b) == "FIRED"
    rig.frames(40, seen=boxes(DUCK), dt=0.1)
    assert rig.state_of(b) == "ARMED"


# 4. Evidence and drawing -------------------------------------------------
def test_an_alert_carries_a_picture(rig):
    """So the agent can look instead of asking a follow-up."""
    watch(rig, [{"type": "near", "other": {"detect": ["person"]}, "margin": 0.05}])
    rig.settle()
    rig.frames(15, seen=boxes(DUCK), dt=0.1)
    rig.frames(4, seen=boxes(DUCK, PERSON_NEAR), dt=0.1)
    near = next(e for e in rig.fired if e.type == "near")
    assert near.snapshot_url == f"/snapshots/{near.id}.jpg"
    assert rig.loop.snapshots.get(near.id, b"")[:2] == b"\xff\xd8"


def test_firing_flashes_the_frame(rig):
    watch(rig, [{"type": "near", "other": {"detect": ["person"]}, "margin": 0.05}])
    rig.settle()
    rig.frames(15, seen=boxes(DUCK), dt=0.1)
    rig.frames(4, seen=boxes(DUCK, PERSON_NEAR), dt=0.1)
    assert layers_of(rig, Alert)


def test_the_guarded_spot_is_drawn(rig):
    watch(rig, [{"type": "moved", "min_shift": 0.2}])
    rig.settle()
    rig.frames(15, seen=boxes(DUCK), dt=0.1)
    assert layers_of(rig, Zone)


# 5. Bad specs ------------------------------------------------------------
@pytest.mark.parametrize("params", [
    {"triggers": []},
    {"triggers": [{"type": "teleport"}]},
    {"triggers": [{"type": "near"}]},
    {"triggers": [{"type": "near", "other": {"detect": []}}]},
])
def test_a_malformed_guard_is_refused_before_it_starts(params):
    with pytest.raises((ValueError, UnsupportedBehavior)):
        build("x", BehaviorSpec(kind="watch", subject={"detect": ["duck"]},
                                params=params))


def test_a_trigger_class_gets_detected(rig):
    """A `near` trigger names a second thing; if the detector is not prepared
    for it the trigger can never fire."""
    watch(rig, [{"type": "near", "other": {"detect": ["person"]}}])
    rig.settle()
    assert set(rig.loop.world.prompt_union(rig.loop.behaviors.values())) == {"duck", "person"}


# 6. count_line -----------------------------------------------------------
def line(rig, **params):
    return rig.builder.add_behavior(BehaviorSpec(
        kind="count_line", subject={"detect": ["person"]},
        params=params, render={"label": "counter"}))


def test_crossing_the_line_counts_once(rig):
    b = line(rig, line=[[0.5, 0.0], [0.5, 1.0]])
    rig.settle()
    for cx in range(200, 460, 20):
        rig.frames(1, seen=boxes(("person", cx, 240, 80)))
    assert rig.view(b).data["in"] + rig.view(b).data["out"] == 1
    assert "crossed" in rig.types()


def test_standing_on_one_side_counts_nothing(rig):
    b = line(rig, line=[[0.5, 0.0], [0.5, 1.0]])
    rig.settle()
    rig.frames(20, seen=boxes(("person", 200, 240, 80)))
    assert rig.view(b).data["in"] == 0 and rig.view(b).data["out"] == 0


def test_crossing_back_counts_both_directions(rig):
    b = line(rig, line=[[0.5, 0.0], [0.5, 1.0]])
    rig.settle()
    for cx in list(range(200, 460, 20)) + list(range(440, 180, -20)):
        rig.frames(1, seen=boxes(("person", cx, 240, 80)))
    d = rig.view(b).data
    assert d["in"] == 1 and d["out"] == 1


def test_the_line_is_drawn_with_its_tally(rig):
    line(rig, line=[[0.5, 0.0], [0.5, 1.0]])
    rig.settle()
    rig.frames(3, seen=boxes(("person", 200, 240, 80)))
    drawn = layers_of(rig, Line)
    assert drawn and "in 0" in drawn[0].label


def test_a_line_needs_two_different_points():
    with pytest.raises(ValueError):
        build("x", BehaviorSpec(kind="count_line", subject={"detect": ["person"]},
                                params={"line": [[0.5, 0.5], [0.5, 0.5]]}))


# 7. privacy --------------------------------------------------------------
def test_privacy_blurs_every_match(rig):
    b = rig.builder.add_behavior(BehaviorSpec(
        kind="privacy", subject={"detect": ["person"]}, render={"label": "blur"}))
    rig.settle()
    rig.frames(3, seen=boxes(PERSON_FAR, PERSON_NEAR))
    assert rig.view(b).data["blurred"] == 2
    assert len(layers_of(rig, Blur)[0].boxes) == 2


def test_privacy_with_a_reference_blurs_everyone_else(rig):
    """"Blur everyone's face except mine" — and an unscored face is blurred,
    not shown. A privacy feature has to fail closed."""
    b = rig.builder.add_behavior(BehaviorSpec(
        kind="privacy", subject={"detect": ["person"], "ref_id": "r-nobody"},
        params={"keep_ref": "r-nobody"}, render={"label": "blur"}))
    rig.settle()
    rig.frames(4, seen=boxes(PERSON_FAR, PERSON_NEAR))
    assert rig.view(b).data["blurred"] == 2
    assert rig.view(b).data["kept"] == 0


def test_an_unknown_privacy_mode_is_refused():
    with pytest.raises(ValueError):
        build("x", BehaviorSpec(kind="privacy", subject={"detect": ["person"]},
                                params={"mode": "melt"}))


def test_blur_actually_changes_pixels():
    frame = np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    before = frame.copy()
    Blur(boxes=np.array([[10.0, 10.0, 90.0, 90.0]])).draw(frame)
    assert not np.array_equal(before, frame)
    assert np.array_equal(before[100:, 100:], frame[100:, 100:])


def test_inverted_blur_spares_the_boxes():
    frame = np.random.default_rng(1).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    before = frame.copy()
    Blur(boxes=np.array([[10.0, 10.0, 90.0, 90.0]]), invert=True).draw(frame)
    assert np.array_equal(before[20:80, 20:80], frame[20:80, 20:80])
    assert not np.array_equal(before[100:, 100:], frame[100:, 100:])
