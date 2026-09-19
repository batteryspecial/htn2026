"""Attribute scoring: telling "a duck" from "the yellow one".

Split in two. The scoring maths is tested directly, because it is the part that
is subtly wrong rather than loudly wrong. Then the whole path is tested through
the running pipeline, because a correct score that never reaches a behaviour is
worth nothing.
"""

import numpy as np
import pytest
import supervision as sv

from attributes.clip_cache import AttributeBank, contrastive_scores, cosine, crops_from
from attributes.encoders import HashEncoder
from tests.rig import Rig, boxes


# 1. The maths ------------------------------------------------------------
def test_a_phrase_that_fits_beats_the_plain_class():
    """The whole reason scoring is contrastive: CLIP's raw similarities are
    too flat to threshold, so the phrase has to win a competition."""
    crop = np.array([[1.0, 0.0]], np.float32)
    phrase = np.array([[0.95, 0.31]], np.float32)
    baseline = np.array([[0.6, 0.8]], np.float32)
    assert contrastive_scores(crop, phrase, baseline)[0, 0] > 0.9


def test_a_phrase_that_does_not_fit_loses():
    crop = np.array([[0.0, 1.0]], np.float32)
    phrase = np.array([[0.95, 0.31]], np.float32)
    baseline = np.array([[0.6, 0.8]], np.float32)
    assert contrastive_scores(crop, phrase, baseline)[0, 0] < 0.1


def test_scores_are_probabilities():
    rng = np.random.default_rng(0)
    crops = rng.standard_normal((6, 8)).astype(np.float32)
    crops /= np.linalg.norm(crops, axis=1, keepdims=True)
    phrases = rng.standard_normal((3, 8)).astype(np.float32)
    phrases /= np.linalg.norm(phrases, axis=1, keepdims=True)
    out = contrastive_scores(crops, phrases, crops)
    assert out.shape == (6, 3)
    assert np.all((out >= 0) & (out <= 1))


def test_no_phrases_is_not_a_crash():
    crop = np.ones((2, 4), np.float32)
    assert contrastive_scores(crop, np.zeros((0, 4), np.float32), crop).shape == (2, 0)
    assert cosine(np.zeros((0, 4), np.float32), crop).shape == (0, 2)


def test_a_degenerate_box_still_yields_a_crop():
    """Trackers do produce zero-width boxes, and an empty array crashes CLIP."""
    frame = np.zeros((40, 40, 3), np.uint8)
    dets = sv.Detections(xyxy=np.array([[10.0, 10.0, 10.0, 10.0]], np.float32))
    crops = crops_from(frame, dets)
    assert len(crops) == 1 and crops[0].size > 0


def test_crops_are_clipped_to_the_frame():
    frame = np.zeros((40, 40, 3), np.uint8)
    dets = sv.Detections(xyxy=np.array([[-50.0, -50.0, 500.0, 500.0]], np.float32))
    assert crops_from(frame, dets)[0].shape[:2] == (40, 40)


# 2. The cache ------------------------------------------------------------
@pytest.fixture
def bank():
    enc = HashEncoder()
    b = AttributeBank(enc, ttl=1.0)
    phrases = ["red", "blue"]
    vecs = enc.encode_text(phrases)
    base = enc.encode_text(["a thing"])
    b.set_texts({p: vecs[i] for i, p in enumerate(phrases)}, {"thing": base[0]})
    return b


def frame_and_dets(colour, track_id=1):
    bgr = {"red": (0, 0, 255), "blue": (255, 0, 0)}[colour]
    frame = np.full((40, 40, 3), bgr, np.uint8)
    dets = sv.Detections(
        xyxy=np.array([[0.0, 0.0, 40.0, 40.0]], np.float32),
        tracker_id=np.array([track_id]),
        data={"class_name": np.array(["thing"], object)},
    )
    return frame, dets


def test_a_track_is_scored_against_every_phrase(bank):
    frame, dets = frame_and_dets("red")
    bank.update(frame, dets, now=0.0)
    scores = bank.scores_for(1)
    assert set(scores) == {"red", "blue"}
    assert scores["red"] > scores["blue"]


def test_include_and_exclude_read_the_same_scores(bank):
    frame, dets = frame_and_dets("red")
    bank.update(frame, dets, now=0.0)
    assert bank.matches(1, ["red"], [], 0.6) is True
    assert bank.matches(1, ["blue"], [], 0.6) is False
    assert bank.matches(1, [], ["red"], 0.6) is False
    assert bank.matches(1, [], ["blue"], 0.6) is True


def test_an_unscored_track_is_undecided_not_a_match(bank):
    """A behaviour must not act on a track whose appearance is unknown.
    That distinction is how a privacy blur avoids missing a face."""
    assert bank.matches(99, ["red"], [], 0.6) is None
    assert bank.matches(99, [], [], 0.6) is True  # nothing asked, nothing to wait for


def test_a_track_is_not_rescored_every_frame(bank):
    """The reason CLIP is affordable at all."""
    frame, dets = frame_and_dets("red")
    bank.update(frame, dets, now=0.0)
    for t in (0.1, 0.2, 0.5, 0.9):
        bank.update(frame, dets, now=t)
    assert bank.encodes == 1
    bank.update(frame, dets, now=2.0)
    assert bank.encodes == 2


def test_a_crowded_frame_cannot_stall_the_loop(bank):
    """Scoring is capped per frame, so fifty people do not become fifty
    forward passes in one tick."""
    bank.batch = 4
    n = 50
    frame = np.full((40, 40, 3), (0, 0, 255), np.uint8)
    dets = sv.Detections(
        xyxy=np.tile(np.array([[0.0, 0.0, 40.0, 40.0]], np.float32), (n, 1)),
        tracker_id=np.arange(1, n + 1),
        data={"class_name": np.array(["thing"] * n, object)},
    )
    bank.update(frame, dets, now=0.0)
    assert len(bank._cache) == 4


def test_the_cache_does_not_grow_without_bound(bank):
    """A live stream never ends, unlike the clips CLIP was benchmarked on."""
    frame = np.full((40, 40, 3), (0, 0, 255), np.uint8)
    bank.batch = 64
    for gen in range(8):
        ids = np.arange(gen * 60 + 1, gen * 60 + 61)
        dets = sv.Detections(
            xyxy=np.tile(np.array([[0.0, 0.0, 40.0, 40.0]], np.float32), (60, 1)),
            tracker_id=ids,
            data={"class_name": np.array(["thing"] * 60, object)},
        )
        bank.update(frame, dets, now=gen * 10.0)
    assert len(bank._cache) <= 320


def test_a_broken_encoder_does_not_take_down_the_pipeline(bank, monkeypatch):
    monkeypatch.setattr(type(bank.encoder), "encode_images",
                        lambda self, c: (_ for _ in ()).throw(RuntimeError("cuda oom")))
    frame, dets = frame_and_dets("red")
    bank.update(frame, dets, now=0.0)  # must not raise
    assert bank.matches(1, ["red"], [], 0.6) is None


# 3. Through the running pipeline -----------------------------------------
@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.close()


LEFT = ("thing", 160, 240, 200)   # left half of the frame
RIGHT = ("thing", 480, 240, 200)  # right half


def test_include_picks_only_the_matching_one(rig):
    """The headline capability: two identical objects, one colour word apart."""
    rig.paint("red", "blue")
    b = rig.add(detect=("thing",), include=["red"])
    rig.settle()
    rig.frames(6, seen=boxes(LEFT, RIGHT))
    assert rig.view(b).matches == 1
    assert rig.view(b).state == "ACTIVE"


def test_exclude_drops_only_the_matching_one(rig):
    """"ignore anybody wearing a black jacket" is this."""
    rig.paint("red", "blue")
    b = rig.add(detect=("thing",), exclude=["red"])
    rig.settle()
    rig.frames(6, seen=boxes(LEFT, RIGHT))
    assert rig.view(b).matches == 1


def test_no_attributes_means_everything_matches(rig):
    rig.paint("red", "blue")
    b = rig.add(detect=("thing",))
    rig.settle()
    rig.frames(6, seen=boxes(LEFT, RIGHT))
    assert rig.view(b).matches == 2


def test_an_impossible_attribute_matches_nothing(rig):
    rig.paint("red", "red")
    b = rig.add(detect=("thing",), include=["blue"])
    rig.settle()
    rig.frames(8, seen=boxes(LEFT, RIGHT))
    assert rig.view(b).matches == 0
    assert rig.view(b).state == "ACTIVE"


def test_two_behaviors_split_one_scene_by_attribute(rig):
    """Both read the same cached scores; the crops are encoded once."""
    rig.paint("red", "blue")
    reds = rig.add(detect=("thing",), include=["red"])
    blues = rig.add(detect=("thing",), include=["blue"])
    rig.settle()
    rig.frames(6, seen=boxes(LEFT, RIGHT))
    assert rig.view(reds).matches == 1
    assert rig.view(blues).matches == 1
    assert rig.view(reds).track_ids != rig.view(blues).track_ids


def test_attribute_scores_are_published_for_the_agent(rig):
    rig.paint("red", "blue")
    b = rig.add(detect=("thing",), include=["red"])
    rig.settle()
    rig.frames(6, seen=boxes(LEFT, RIGHT))
    scored = [t for t in rig.last.tracks if t.attributes]
    assert scored and "red" in scored[0].attributes


def test_a_pipeline_without_clip_still_runs(tmp_path):
    """No weights, no GPU, no CLIP: behaviours with no attributes still work."""
    r = Rig(tmp_path, attributes=False)
    try:
        b = r.add(detect=("thing",))
        r.settle()
        r.frames(6, seen=boxes(LEFT))
        assert r.view(b).state == "ACTIVE"
    finally:
        r.close()


def test_without_clip_an_attribute_behavior_never_falsely_matches(tmp_path):
    """Degrading to "matches nothing" is safe. Degrading to "matches
    everything" would blur the wrong faces."""
    r = Rig(tmp_path, attributes=False)
    try:
        b = r.add(detect=("thing",), include=["red"])
        r.settle()
        r.frames(8, seen=boxes(LEFT, RIGHT))
        assert r.view(b).matches == 0
    finally:
        r.close()


# 4. Competing phrases ----------------------------------------------------
def test_a_winning_include_beats_a_high_exclude(bank):
    """The measured case: a cream coat scores 0.78 for "dark jacket", over any
    sane bar, but 1.00 for "cream coat". Comparison gets it right where an
    absolute threshold cannot, because lighting moves every score together."""
    frame, dets = frame_and_dets("red")
    bank.update(frame, dets, now=0.0)
    scores = bank.scores_for(1)
    assert scores["red"] > scores["blue"]
    assert bank.matches(1, ["red"], ["blue"], 0.6) is True
    assert bank.matches(1, ["blue"], ["red"], 0.6) is False


def test_a_competing_include_still_has_to_clear_the_bar(bank):
    """Winning a comparison is not enough if nothing fits at all."""
    frame, dets = frame_and_dets("red")
    bank.update(frame, dets, now=0.0)
    assert bank.matches(1, ["blue"], ["red"], 0.99) is False


def test_exclude_alone_still_uses_an_absolute_bar(bank):
    frame, dets = frame_and_dets("red")
    bank.update(frame, dets, now=0.0)
    assert bank.matches(1, [], ["red"], 0.6) is False
    assert bank.matches(1, [], ["blue"], 0.6) is True


def test_competing_phrases_through_the_pipeline(rig):
    """Two objects, one word apart, decided by which phrase wins."""
    rig.paint("red", "blue")
    reds = rig.add(detect=("thing",), include=["red"], exclude=["blue"])
    rig.settle()
    rig.frames(6, seen=boxes(LEFT, RIGHT))
    assert rig.view(reds).matches == 1
