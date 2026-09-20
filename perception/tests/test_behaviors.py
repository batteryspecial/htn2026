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

    rig.builder.add_behavior(BehaviorSpec(kind="keyboard", subject={"detect": ["keyboard"]}))
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


# 9. Low-confidence detections ---------------------------------------------
# The hole that let a tracker bug through for a day. Every fixture emitted
# confidence 0.9, comfortably above both the detector cutoff and ByteTrack's
# hidden `track_activation_threshold + 0.1` gate, so 354 tests all passed
# while real detections at 0.18-0.30 were found and silently never tracked.
def test_a_low_confidence_detection_still_becomes_a_track(rig):
    """Open-vocabulary matches routinely sit at 0.15-0.30. If the tracker
    refuses to create a track there, the behaviour layer sees nothing while
    the logs show inference succeeding."""
    b = rig.add(detect=("thing",))
    rig.settle()
    rig.frames(5, seen=boxes(THING, conf=0.20))
    assert rig.view(b).matches >= 1, "a real detection was found and never tracked"


def test_tracking_survives_just_above_the_detector_cutoff(rig):
    from config import CFG

    b = rig.add(kind="track", detect=("thing",), pick="ref")
    rig.settle()
    rig.frames(6, seen=boxes(THING, conf=CFG.CONF_THRESHOLD + 0.01))
    assert rig.state_of(b) == "TRACKING"


def test_the_tracker_gate_sits_below_the_detector_cutoff(rig):
    """ByteTrack adds 0.1 to its activation threshold before deciding whether
    to create a track. If that lands above CONF_THRESHOLD, everything between
    the two is detected and discarded."""
    from config import CFG

    gate = CFG.TRACK_ACTIVATION_THRESHOLD + 0.1
    assert gate <= CFG.CONF_THRESHOLD, (
        f"tracker creates no track below {gate:.2f} but the detector emits "
        f"down to {CFG.CONF_THRESHOLD:.2f}; the gap is silently discarded")


def test_every_tracker_gate_sits_below_what_it_actually_sees(rig):
    """All three gates, checked against the *boosted* cutoff, because that is
    the number ByteTrack is handed.

    Each one decides something different — whether a track can be born,
    confirmed, or continued — and each is fused with score, so each is really
    a confidence floor. Two of them are ours; the third is written inline in
    the dependency and is the reason the rescale exists.
    """
    from config import CFG
    from runtime.world import _boost

    seen = float(_boost(boxes(("f", 0, 0, 10),
                              conf=CFG.CONF_THRESHOLD)).confidence[0])
    gates = {
        "birth (det_thresh)": CFG.TRACK_ACTIVATION_THRESHOLD + 0.1,
        "continuation (minimum_matching_threshold)": 1.0 - CFG.MIN_MATCHING_THRESHOLD,
        "confirmation (hardcoded thresh=0.7)": 1.0 - 0.7,
    }
    for name, floor in gates.items():
        assert floor < seen, (
            f"{name} needs {floor:.2f} but a detection at the cutoff reaches "
            f"the tracker as {seen:.2f}; that band flickers every frame")


def test_a_detection_at_the_cutoff_keeps_its_track(rig):
    """The end-to-end version: born *and* kept, not born and abandoned."""
    from config import CFG

    b = rig.add(kind="track", detect=("thing",), pick="ref")
    rig.settle()
    rig.frames(8, seen=boxes(THING, conf=CFG.CONF_THRESHOLD + 0.005))
    assert rig.state_of(b) == "TRACKING"
    assert rig.view(b).matches == 1


# 10. The confidence rescale ----------------------------------------------
# ByteTrack's constants assume a COCO detector where a real object scores
# 0.8+. Open-vocabulary matching is a similarity on a different scale, where
# 0.15-0.30 is correct, so every gate landed in the wrong place. The worst of
# them is `thresh=0.7` written inline for confirming a new track, which cannot
# be configured — hence rescaling rather than patching a dependency.
def test_the_rescale_round_trips_exactly():
    """Behaviours gate on confidence, so the number they see must be the one
    the detector produced."""
    from runtime.world import _boost, _restore

    for c in (0.15, 0.2, 0.37, 0.6, 1.0):
        out = _restore(_boost(boxes(("f", 0, 0, 10), conf=c)))
        assert out.confidence[0] == pytest.approx(c, abs=1e-5)


def test_the_rescale_preserves_ordering():
    """Strictly increasing, so no matching decision changes — only the scale."""
    from runtime.world import _boost

    confs = [0.15, 0.18, 0.25, 0.4, 0.9]
    boosted = [_boost(boxes(("f", 0, 0, 10), conf=c)).confidence[0] for c in confs]
    assert boosted == sorted(boosted)
    assert all(b > a for a, b in zip(boosted, boosted[1:]))


def test_the_cutoff_lands_above_bytetracks_hardcoded_gate():
    """ByteTrack confirms a new track with `thresh=0.7`, fused with score,
    which demands 0.30. A detector emitting down to 0.15 must be mapped above
    that or nothing it finds can ever be confirmed."""
    from config import CFG
    from runtime.world import _boost

    at_cutoff = _boost(boxes(("f", 0, 0, 10), conf=CFG.CONF_THRESHOLD)).confidence[0]
    assert at_cutoff > 0.30, (
        f"a detection at the cutoff maps to {at_cutoff:.2f}, below ByteTrack's "
        f"hardcoded 0.30 confirmation gate; it will flicker every frame")


def test_a_cutoff_detection_survives_a_prior_empty_frame(rig):
    """The exact shape of the live bug. ByteTrack activates a new track
    immediately only on the very first frame it ever sees; after any empty
    frame a new track must re-match to confirm, and that is where low
    confidence used to die."""
    from config import CFG
    from detectors.base import empty_detections
    from runtime.world import Shared

    shared = Shared()
    shared.track(empty_detections())
    low = boxes(("f", 320, 240, 100), conf=CFG.CONF_THRESHOLD + 0.005)
    tracked = [len(shared.track(low)) for _ in range(6)]
    assert sum(tracked) >= 4, f"flickered: {tracked}"


def test_a_weak_match_no_longer_flickers(rig):
    """Asking for a red bottle and being shown a green one scores about 0.2:
    high enough to detect, too low to confirm. It used to produce a track per
    frame that never survived, which reads as a mask flashing."""
    b = rig.add(kind="track", detect=("thing",), pick="ref")
    rig.settle()
    rig.frames(2, seen=boxes())
    rig.frames(8, seen=boxes(THING, conf=0.20))
    assert rig.state_of(b) == "TRACKING"
    assert rig.view(b).matches == 1
