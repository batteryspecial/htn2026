"""Behaviour engine: concurrency, selectors, per-kind state machines.

Independence is the thing that must not break. Several behaviours share one
detection pass, one tracker and one attribute cache, and adding or removing any
of them must leave the rest exactly as they were.
"""

import numpy as np
import pytest

from behaviors.base import UnsupportedBehavior
from behaviors.kinds import BUILT, KINDS, build
from contracts import BehaviorSpec
from tests.rig import Rig, boxes

THING = ("thing", 320, 240, 100)
OTHER = ("other", 100, 100, 60)


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.close()


# 1. highlight: ACTIVE / PAUSED -------------------------------------------
def test_highlight_is_active_from_the_start(rig):
    b = rig.add()
    rig.settle()
    rig.frames(3, seen=boxes(THING))
    assert rig.state_of(b) == "ACTIVE"


def test_highlight_counts_every_match(rig):
    b = rig.add()
    rig.settle()
    rig.frames(4, seen=boxes(("thing", 100, 100, 50), ("thing", 400, 300, 50),
                             ("thing", 500, 100, 50)))
    assert rig.view(b).matches == 3
    assert rig.view(b).data["count"] == 3


def test_highlight_reports_only_when_the_count_changes(rig):
    """Edge-triggered: the agent wants to hear that it changed, not that it is
    still four."""
    b = rig.add()
    rig.settle()
    rig.frames(5, seen=boxes(THING))          # 0 -> 1
    rig.frames(5, seen=boxes(THING))          # still 1, must stay quiet
    rig.frames(5, seen=boxes(THING, ("thing", 500, 100, 50)))  # 1 -> 2
    changes = [(e.data["previous"], e.data["count"])
               for e in rig.fired if e.type == "count_changed"]
    assert changes == [(0, 1), (1, 2)], "an unchanged count should say nothing"


def test_highlight_ignores_things_that_are_not_its_subject(rig):
    b = rig.add(detect=("thing",))
    rig.settle()
    rig.frames(4, seen=boxes(OTHER))
    assert rig.view(b).matches == 0


# 2. track: ACQUIRING -> TRACKING <-> EDGE -> LOST -> SEARCHING -----------
def test_track_starts_acquiring(rig):
    b = rig.add(kind="track", pick="ref")
    rig.settle()
    rig.frames(1, seen=boxes())
    assert rig.state_of(b) == "ACQUIRING"


def test_track_reaches_tracking_after_a_few_sightings(rig):
    b = rig.add(kind="track", pick="ref")
    rig.settle()
    rig.frames(1, seen=boxes(THING))
    assert rig.state_of(b) == "ACQUIRING"
    rig.frames(3, seen=boxes(THING))
    assert rig.state_of(b) == "TRACKING"


def test_track_enters_edge_near_the_frame_boundary(rig):
    """An early warning, not a failure: the subject is still visible."""
    b = rig.add(kind="track", pick="ref")
    rig.settle()
    rig.frames(5, seen=boxes(("thing", 620, 240, 60)))
    assert rig.state_of(b) == "EDGE"
    assert abs(rig.view(b).data["cx"]) > 0.7


def test_edge_returns_to_tracking_when_it_comes_back(rig):
    """Walked back in, rather than teleported: a jump across the frame is a
    different instance as far as the tracker is concerned."""
    b = rig.add(kind="track", pick="ref")
    rig.settle()
    rig.frames(5, seen=boxes(("thing", 620, 240, 60)))
    assert rig.state_of(b) == "EDGE"
    for cx in range(600, 300, -20):
        rig.frames(1, seen=boxes(("thing", cx, 240, 60)))
    assert rig.state_of(b) == "TRACKING"


def test_a_brief_dropout_does_not_lose_the_target(rig):
    b = rig.add(kind="track", pick="ref")
    rig.settle()
    rig.frames(5, seen=boxes(THING))
    rig.frames(4, seen=boxes())
    assert rig.state_of(b) == "TRACKING"
    rig.frames(1, seen=boxes())
    assert rig.state_of(b) == "LOST"


def test_lost_records_the_exit_side(rig):
    b = rig.add(kind="track", pick="ref")
    rig.settle()
    rig.frames(5, seen=boxes(("thing", 600, 240, 60)))
    rig.frames(6, seen=boxes())
    assert rig.state_of(b) == "LOST"
    assert "right" in rig.view(b).detail


def test_lost_becomes_searching_after_ten_seconds(rig):
    b = rig.add(kind="track", pick="ref")
    rig.settle()
    rig.frames(5, seen=boxes(THING))
    rig.frames(6, seen=boxes())
    assert rig.state_of(b) == "LOST"
    rig.frames(40, seen=boxes(), dt=0.5)
    assert rig.state_of(b) == "SEARCHING"


def test_track_emits_acquired_once_then_reacquired(rig):
    b = rig.add(kind="track", pick="ref")
    rig.settle()
    rig.frames(5, seen=boxes(THING))
    rig.frames(6, seen=boxes())
    rig.frames(5, seen=boxes(THING))
    types = [e.type for e in rig.fired if e.behavior_id == b]
    assert types.count("acquired") == 1
    assert "lost" in types


# 3. The rule that makes or breaks the demo -------------------------------
def test_a_passerby_does_not_become_the_target(rig):
    """Leaving LOST needs the *same* instance. A random person cancelling the
    guidance arrow is what makes the whole thing look broken."""
    rig.paint("red")
    b = rig.add(kind="track", pick="ref")
    rig.settle()
    rig.frames(6, seen=boxes(THING))
    locked = rig.view(b).data["track_id"]

    rig.frames(6, seen=boxes())
    assert rig.state_of(b) == "LOST"

    # A different instance of the same class, and it looks different.
    rig.paint("blue")
    rig.frames(6, seen=boxes(("thing", 200, 240, 90)))
    assert rig.state_of(b) == "LOST", "a passerby was adopted as the target"
    assert rig.view(b).data["track_id"] == locked


def test_the_same_instance_does_reacquire(rig):
    rig.paint("red")
    b = rig.add(kind="track", pick="ref")
    rig.settle()
    rig.frames(6, seen=boxes(THING))
    rig.frames(6, seen=boxes())
    assert rig.state_of(b) == "LOST"
    rig.frames(6, seen=boxes(("thing", 300, 240, 100)))
    assert rig.state_of(b) == "TRACKING"
    assert "reacquired" in [e.type for e in rig.fired if e.behavior_id == b]


# 4. Several at once ------------------------------------------------------
def test_two_behaviors_run_on_the_same_frame(rig):
    a = rig.add(detect=("thing",))
    b = rig.add(detect=("other",))
    rig.settle()
    rig.frames(4, seen=boxes(THING, OTHER))
    assert rig.view(a).matches == 1 and rig.view(b).matches == 1


def test_adding_a_behavior_does_not_disturb_the_running_one(rig):
    """Starting a privacy blur must not make the follow-cam lose its target."""
    a = rig.add(kind="track", detect=("thing",), pick="ref")
    rig.settle()
    rig.frames(8, seen=boxes(THING, OTHER))
    before = rig.view(a).data["track_id"]

    rig.add(detect=("other",))
    rig.settle()
    rig.frames(3, seen=boxes(THING, OTHER))
    assert rig.state_of(a) == "TRACKING"
    assert rig.view(a).data["track_id"] == before, "tracking reset by an unrelated change"


def test_removing_one_leaves_the_others(rig):
    a = rig.add(detect=("thing",))
    b = rig.add(detect=("other",))
    rig.settle()
    rig.frames(6, seen=boxes(THING, OTHER))
    rig.remove(b)
    rig.settle()
    rig.frames(3, seen=boxes(THING, OTHER))
    assert rig.view(b) is None
    assert rig.view(a).matches == 1


def test_behaviors_get_distinct_colors(rig):
    """Two targets must be visibly different on the projector."""
    ids = [rig.add(detect=("thing",)) for _ in range(4)]
    rig.settle()
    rig.frames(2, seen=boxes(THING))
    from render.layers import distinct

    colors = [rig.loop.behaviors[i].color for i in ids]
    assert all(distinct(a, b) for n, a in enumerate(colors) for b in colors[n + 1:])


def test_an_auto_color_avoids_one_already_in_use(rig):
    """Asking for a colour that matches the next palette slot must not leave
    two behaviours looking identical across a room."""
    from render.layers import PALETTE

    a = rig.add(detect=("thing",), render={"color": "#%02X%02X%02X" % (
        PALETTE[1][2], PALETTE[1][1], PALETTE[1][0])})
    b = rig.add(detect=("other",))
    rig.settle()
    rig.frames(2, seen=boxes(THING, OTHER))
    from render.layers import distinct

    assert distinct(rig.loop.behaviors[a].color, rig.loop.behaviors[b].color)


def test_a_named_color_is_honoured(rig):
    b = rig.add(detect=("thing",), render={"color": "#FFD400"})
    rig.settle()
    assert rig.loop.behaviors[b].color == (0, 212, 255)


# 5. PAUSED ---------------------------------------------------------------
def test_a_model_that_cannot_see_the_subject_pauses_it(rig, monkeypatch):
    b = rig.add(detect=("thing",))
    rig.settle()
    rig.frames(4, seen=boxes(THING))
    assert rig.state_of(b) == "ACTIVE"

    # `other` is a fixed-vocabulary model that has never heard of "thing".
    other = rig.registry.get("other")
    monkeypatch.setattr(type(other), "classes", property(lambda self: ["face"]))
    rig.builder.set_model("other")
    rig.settle()
    rig.frames(3, seen=boxes(THING))
    assert rig.state_of(b) == "PAUSED"
    assert "cannot detect" in rig.view(b).detail


def test_a_paused_behavior_resumes_when_a_capable_model_returns(rig, monkeypatch):
    b = rig.add(detect=("thing",))
    rig.settle()
    rig.frames(4, seen=boxes(THING))

    other = rig.registry.get("other")
    monkeypatch.setattr(type(other), "classes", property(lambda self: ["face"]))
    rig.builder.set_model("other")
    rig.settle()
    rig.frames(2, seen=boxes(THING))
    assert rig.state_of(b) == "PAUSED"

    monkeypatch.undo()
    rig.builder.set_model("fake")
    rig.settle()
    rig.frames(4, seen=boxes(THING))
    assert rig.state_of(b) == "ACTIVE"


def test_a_paused_behavior_does_nothing(rig, monkeypatch):
    b = rig.add(detect=("thing",))
    rig.settle()
    other = rig.registry.get("other")
    monkeypatch.setattr(type(other), "classes", property(lambda self: ["face"]))
    rig.builder.set_model("other")
    rig.settle()
    before = len(rig.fired)
    rig.frames(10, seen=boxes(THING))
    assert rig.view(b).matches == 0
    assert len(rig.fired) == before


def test_a_model_switch_is_announced(rig):
    rig.add(detect=("thing",))
    rig.settle()
    rig.builder.set_model("other")
    rig.settle()
    rig.frames(2, seen=boxes(THING))
    assert "model_switched" in rig.types()


# 6. Failure is a no-op ---------------------------------------------------
def test_an_unimplemented_kind_is_refused_before_anything_changes(rig):
    a = rig.add(detect=("thing",))
    rig.settle()
    rig.frames(4, seen=boxes(THING))

    rig.builder.add_behavior(BehaviorSpec(kind="pan_to", subject={"detect": ["face"]}))
    rig.settle()
    rig.frames(2, seen=boxes(THING))
    assert len(rig.loop.behaviors) == 1
    assert rig.state_of(a) == "ACTIVE"
    assert "behavior_failed" in rig.types()


def test_one_broken_behavior_does_not_stop_the_others(rig, monkeypatch):
    a = rig.add(detect=("thing",))
    b = rig.add(detect=("thing",))
    rig.settle()
    rig.frames(4, seen=boxes(THING))

    broken = rig.loop.behaviors[a]
    monkeypatch.setattr(type(broken), "on_frame",
                        lambda self, f, i, o: (_ for _ in ()).throw(RuntimeError("boom"))
                        if self.id == a else None)
    rig.frames(3, seen=boxes(THING))
    assert rig.state_of(a) == "FAILED"
    assert rig.state_of(b) == "ACTIVE"


# 7. The kind registry ----------------------------------------------------
def test_every_contract_kind_is_registered():
    from typing import get_args

    from linker.schemas import BehaviorKind

    assert set(get_args(BehaviorKind)) == set(KINDS)


def test_unbuilt_kinds_say_what_they_need():
    for kind in set(KINDS) - BUILT:
        with pytest.raises(UnsupportedBehavior) as e:
            build("x", BehaviorSpec(kind=kind, subject={"detect": ["a"]}))
        assert "not implemented" in str(e.value)
        assert sorted(BUILT)[0] in str(e.value)


def test_declared_states_cover_the_contract():
    """A kind whose state machine can reach a state the contract does not list
    would publish something nobody can parse."""
    from typing import get_args

    from linker.schemas import BehaviorState

    allowed = set(get_args(BehaviorState))
    for kind, cls in KINDS.items():
        assert set(cls.states) <= allowed, f"{kind} declares an unknown state"


# 8. A selector that finds nothing must say so ----------------------------
def test_a_selector_that_never_matches_says_so(rig):
    """Measured on real footage: YOLOE finds a rubber duck for "yellow duck"
    in every frame and for "duck" in none. Silence looks like success."""
    b = rig.add(detect=("nothing_like_this",))
    rig.settle()
    rig.frames(30, seen=boxes(THING), dt=0.2)
    assert "nothing matches" in (rig.view(b).detail or "")


def test_a_selector_that_does_match_stays_quiet(rig):
    b = rig.add(detect=("thing",))
    rig.settle()
    rig.frames(30, seen=boxes(THING), dt=0.2)
    assert rig.view(b).detail is None


def test_the_complaint_clears_once_something_shows_up(rig):
    b = rig.add(detect=("thing",))
    rig.settle()
    rig.frames(30, seen=boxes(), dt=0.2)
    assert "nothing matches" in (rig.view(b).detail or "")
    rig.frames(4, seen=boxes(THING))
    assert rig.view(b).detail is None
