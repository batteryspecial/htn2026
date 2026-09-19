"""Guidance arrows: the human is the motor.

The command is the thing that survives the hardware, so these test the command
and the arrow separately: the tracker decides what the view should do, the
virtual motor decides what that looks like, and later a stepper would decide
what it does. None of this changes when the stepper arrives.

The failure that matters is not a missing arrow, it is an arrow that goes away
when it should not.
"""

import numpy as np
import pytest

from actuator.virtual import VirtualMotor
from contracts import MotorCommand
from render.layers import Alert, Arrow
from tests.rig import Rig, boxes

CENTER = ("thing", 320, 240, 100)
RIGHT_EDGE = ("thing", 620, 240, 60)
LEFT_EDGE = ("thing", 20, 240, 60)


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.close()


def motor_of(rig):
    return rig.motor.last


def arrows(rig):
    return [l for l in rig.loop.view.layers if isinstance(l, Arrow)]


def alerts(rig):
    return [l for l in rig.loop.view.layers if isinstance(l, Alert)]


# 1. The command ----------------------------------------------------------
def test_a_centred_target_asks_for_nothing(rig):
    rig.add(kind="track", pick="ref")
    rig.settle()
    rig.frames(5, seen=boxes(CENTER))
    assert motor_of(rig) is None
    assert not arrows(rig)


def test_drifting_to_the_edge_asks_for_a_nudge(rig):
    rig.add(kind="track", pick="ref")
    rig.settle()
    rig.frames(5, seen=boxes(RIGHT_EDGE))
    cmd = motor_of(rig)
    assert cmd is not None and cmd.rate_deg_s > 0 and not cmd.urgent


def test_the_nudge_points_the_way_the_target_went(rig):
    rig.add(kind="track", pick="ref")
    rig.settle()
    rig.frames(5, seen=boxes(LEFT_EDGE))
    assert motor_of(rig).rate_deg_s < 0


def test_losing_the_target_is_urgent(rig):
    rig.add(kind="track", pick="ref")
    rig.settle()
    rig.frames(5, seen=boxes(RIGHT_EDGE))
    rig.frames(6, seen=boxes())
    cmd = motor_of(rig)
    assert cmd.urgent and cmd.rate_deg_s > 0, "should point at the edge it left by"


def test_guidance_can_be_switched_off(rig):
    rig.add(kind="track", pick="ref", params={"guidance": False})
    rig.settle()
    rig.frames(6, seen=boxes(RIGHT_EDGE))
    assert motor_of(rig) is None


def test_the_newest_tracker_takes_over_the_arrows(rig):
    """"now follow the dog" must steer, not the thing started ten minutes ago."""
    rig.add(kind="track", detect=("thing",), pick="ref")
    rig.settle()
    rig.frames(5, seen=boxes(LEFT_EDGE))
    assert motor_of(rig).rate_deg_s < 0

    rig.add(kind="track", detect=("other",), pick="ref")
    rig.settle()
    rig.frames(5, seen=boxes(LEFT_EDGE, ("other", 620, 240, 60)))
    assert motor_of(rig).rate_deg_s > 0, "the older behaviour is still steering"


# 2. The arrow ------------------------------------------------------------
def test_an_edge_warning_is_faint(rig):
    rig.add(kind="track", pick="ref")
    rig.settle()
    rig.frames(5, seen=boxes(RIGHT_EDGE))
    a = arrows(rig)
    assert len(a) == 1 and a[0].direction == "right" and a[0].strength < 0.5


def test_a_lost_target_gets_a_bold_held_arrow(rig):
    rig.add(kind="track", pick="ref")
    rig.settle()
    rig.frames(5, seen=boxes(LEFT_EDGE))
    rig.frames(6, seen=boxes())
    a = arrows(rig)
    assert len(a) == 1 and a[0].direction == "left" and a[0].strength == 1.0
    assert "pan left" in a[0].label


def test_the_arrow_is_held_while_the_target_is_gone(rig):
    rig.add(kind="track", pick="ref")
    rig.settle()
    rig.frames(5, seen=boxes(RIGHT_EDGE))
    for _ in range(12):
        rig.frames(1, seen=boxes())
        if rig.state_of_track_lost():
            assert arrows(rig), "the arrow blinked out while the target was gone"


def test_after_ten_seconds_it_becomes_a_banner(rig):
    """An arrow nobody is following stops meaning anything."""
    rig.add(kind="track", pick="ref")
    rig.settle()
    rig.frames(5, seen=boxes(RIGHT_EDGE))
    rig.frames(40, seen=boxes(), dt=0.5)
    assert not arrows(rig)
    assert any("searching" in a.text for a in alerts(rig))


# 3. The rule that protects the demo --------------------------------------
def test_a_passerby_does_not_cancel_the_arrow(rig):
    """The arrow goes away only when the *same* instance comes back. A random
    person clearing it is what makes the demo look broken."""
    rig.paint("red")
    rig.add(kind="track", pick="ref")
    rig.settle()
    rig.frames(6, seen=boxes(RIGHT_EDGE))
    rig.frames(6, seen=boxes())
    assert arrows(rig)

    rig.paint("blue")
    rig.frames(6, seen=boxes(("thing", 200, 240, 90)))
    assert arrows(rig), "a passerby cancelled the guidance arrow"


def test_the_same_instance_does_cancel_the_arrow(rig):
    rig.paint("red")
    rig.add(kind="track", pick="ref")
    rig.settle()
    rig.frames(6, seen=boxes(RIGHT_EDGE))
    rig.frames(6, seen=boxes())
    assert arrows(rig)
    rig.frames(6, seen=boxes(CENTER))
    assert not arrows(rig)


# 4. The motor on its own -------------------------------------------------
def test_the_virtual_motor_draws_nothing_for_no_command():
    assert VirtualMotor().command(None, (480, 640), 0.0) == []


def test_an_urgent_command_pulses():
    m = VirtualMotor()
    seen = {m.command(MotorCommand(rate_deg_s=-20, urgent=True), (480, 640), t)[0].pulse
            for t in (0.0, 0.2, 0.4, 0.6)}
    assert len(seen) > 1, "a held arrow that does not move is easy to ignore"


def test_the_banner_replaces_the_arrow_not_joins_it():
    m = VirtualMotor(search_after_s=1.0)
    cmd = MotorCommand(rate_deg_s=-20, urgent=True)
    m.command(cmd, (480, 640), 0.0)
    layers = m.command(cmd, (480, 640), 2.0)
    assert len(layers) == 1 and isinstance(layers[0], Alert)


def test_a_recovered_command_resets_the_search_timer():
    m = VirtualMotor(search_after_s=1.0)
    urgent = MotorCommand(rate_deg_s=-20, urgent=True)
    m.command(urgent, (480, 640), 0.0)
    m.command(None, (480, 640), 0.5)          # target came back
    layers = m.command(urgent, (480, 640), 1.6)  # and left again
    assert isinstance(layers[0], Arrow), "the banner should not carry over"


def test_arrows_draw_without_error():
    frame = np.zeros((240, 320, 3), np.uint8)
    for d in ("left", "right", "up", "down"):
        Arrow(direction=d, label=f"pan {d}").draw(frame)
    assert frame.any()
