"""Behaviour engine: concurrency, subjects, lifecycle.

The thing that must not break is independence. Several behaviours share one
detection pass, one tracker and one attribute cache, and adding or removing any
of them must leave the rest exactly as they were.
"""

import pytest

from behaviors.base import UnsupportedBehavior
from behaviors.kinds import BUILT, KINDS, build
from contracts import Behavior
from tests.rig import Rig, boxes

THING = ("thing", 320, 240, 100)
OTHER = ("other", 100, 100, 60)


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.close()


# 1. One behaviour --------------------------------------------------------
def test_a_behavior_becomes_active_when_it_sees_its_subject(rig):
    rig.add("b1")
    rig.settle()
    rig.frames(5, seen=boxes(THING))
    assert rig.status("b1").state == "active"
    assert rig.status("b1").matches == 1


def test_a_behavior_ignores_things_that_are_not_its_subject(rig):
    rig.add("b1", detect=("thing",))
    rig.settle()
    rig.frames(5, seen=boxes(OTHER))
    assert rig.status("b1").state == "arming"
    assert rig.status("b1").matches == 0


def test_a_behavior_needs_several_frames_before_it_believes_anything(rig):
    rig.add("b1")
    rig.settle()
    rig.frames(1, seen=boxes(THING))
    rig.frames(1, seen=boxes())
    rig.frames(2, seen=boxes(THING))
    assert rig.status("b1").state == "arming"
    rig.frames(1, seen=boxes(THING))
    assert rig.status("b1").state == "active"


def test_a_brief_dropout_does_not_end_a_behavior(rig):
    rig.add("b1")
    rig.settle()
    rig.frames(5, seen=boxes(THING))
    rig.frames(4, seen=boxes())
    assert rig.status("b1").state == "active"
    rig.frames(1, seen=boxes())
    assert rig.status("b1").state == "lost"


def test_a_lost_behavior_recovers(rig):
    rig.add("b1")
    rig.settle()
    rig.frames(5, seen=boxes(THING))
    rig.frames(6, seen=boxes())
    assert rig.status("b1").state == "lost"
    rig.frames(5, seen=boxes(THING))
    assert rig.status("b1").state == "active"


def test_highlight_counts_every_match(rig):
    rig.add("b1")
    rig.settle()
    rig.frames(5, seen=boxes(("thing", 100, 100, 50), ("thing", 400, 300, 50),
                             ("thing", 500, 100, 50)))
    assert rig.status("b1").matches == 3
    assert rig.status("b1").data["count"] == 3


# 2. Several at once ------------------------------------------------------
def test_two_behaviors_run_on_the_same_frame(rig):
    rig.add("a", detect=("thing",))
    rig.add("b", detect=("other",))
    rig.settle()
    rig.frames(5, seen=boxes(THING, OTHER))
    assert rig.status("a").state == "active" and rig.status("a").matches == 1
    assert rig.status("b").state == "active" and rig.status("b").matches == 1


def test_adding_a_behavior_does_not_disturb_the_running_one(rig):
    """The rule that matters most. Starting a privacy blur must not make the
    follow-cam lose its target."""
    rig.add("a", detect=("thing",))
    rig.settle()
    rig.frames(10, seen=boxes(THING, OTHER))
    before = rig.status("a").track_ids

    rig.add("b", detect=("other",))
    rig.settle()
    rig.frames(2, seen=boxes(THING, OTHER))
    assert rig.status("a").state == "active"
    assert rig.status("a").track_ids == before, "tracking was reset by an unrelated change"


def test_removing_a_behavior_leaves_the_others_running(rig):
    rig.add("a", detect=("thing",))
    rig.add("b", detect=("other",))
    rig.settle()
    rig.frames(10, seen=boxes(THING, OTHER))
    before = rig.status("a").track_ids

    rig.remove("b")
    rig.settle()
    rig.frames(2, seen=boxes(THING, OTHER))
    assert rig.status("b") is None
    assert rig.status("a").state == "active"
    assert rig.status("a").track_ids == before


def test_two_behaviors_can_share_one_subject(rig):
    """Detection runs once on the union, so overlapping subjects are free."""
    rig.add("a", kind="highlight", detect=("thing",))
    rig.add("b", kind="track", detect=("thing",))
    rig.settle()
    rig.frames(5, seen=boxes(THING))
    assert rig.status("a").matches == 1 and rig.status("b").matches == 1
    assert rig.loop.world.prompt_union(rig.loop.behaviors.values()) == ["thing"]


def test_clearing_stops_everything(rig):
    rig.add("a")
    rig.add("b", detect=("other",))
    rig.settle()
    rig.frames(5, seen=boxes(THING, OTHER))
    rig.builder.clear()
    rig.settle()
    rig.frames(2, seen=boxes(THING, OTHER))
    assert rig.last.behaviors == []


# 3. Failure is a no-op ---------------------------------------------------
def test_an_unimplemented_kind_is_refused_before_anything_changes(rig):
    rig.add("a")
    rig.settle()
    rig.frames(5, seen=boxes(THING))

    rig.builder.add_behavior("bad", Behavior(
        behavior_id="nope", kind="privacy", subject={"detect": ["face"]}))
    rig.settle()
    rig.frames(2, seen=boxes(THING))
    assert rig.status("nope") is None
    assert rig.status("a").state == "active"
    assert "failed" in rig.stages


def test_removing_something_that_is_not_there_is_refused(rig):
    rig.remove("ghost")
    rig.settle()
    assert "failed" in rig.stages


def test_one_broken_behavior_does_not_stop_the_others(rig, monkeypatch):
    rig.add("a", detect=("thing",))
    rig.add("b", detect=("thing",))
    rig.settle()
    rig.frames(5, seen=boxes(THING))

    broken = rig.loop.behaviors["a"]
    monkeypatch.setattr(type(broken), "on_frame",
                        lambda self, f, i, o: (_ for _ in ()).throw(RuntimeError("boom"))
                        if self.id == "a" else None)
    rig.frames(3, seen=boxes(THING))
    assert rig.status("a").state == "failed"
    assert rig.status("b").state == "active"


# 4. The kind registry ----------------------------------------------------
def test_every_contract_kind_is_registered():
    """A kind in the shared contract with no class would fail confusingly."""
    from typing import get_args
    from linker.schemas import BehaviorKind

    assert set(get_args(BehaviorKind)) == set(KINDS)


def test_unbuilt_kinds_say_what_they_need():
    for kind in set(KINDS) - BUILT:
        with pytest.raises(UnsupportedBehavior) as e:
            build(Behavior(behavior_id="x", kind=kind, subject={"detect": ["a"]}))
        assert "not implemented" in str(e.value)
        assert sorted(BUILT)[0] in str(e.value), "error should name what is available"
