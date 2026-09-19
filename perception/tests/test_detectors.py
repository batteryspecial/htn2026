"""Detector-boundary tests, written against the contract in base.py rather than
against any implementation.

The invariants that matter are the ones the pipeline leans on:
prepare must be pure and must be the only thing that raises, apply must never
be reached with a bad spec, and infer must never hand back a shape the loop has
to branch on.
"""

import numpy as np
import pytest
import supervision as sv

from detectors.base import Detector, UnknownClassError, empty_detections, validate_vocab
from detectors.fake import FakeDetector


class FixedStub(Detector):
    """Minimal fixed-vocabulary detector. Stands in for COCO, a TensorRT build,
    or anything else with a frozen class list, without needing weights."""

    open_vocab = False

    def __init__(self, classes=("person", "dog", "shoe")):
        super().__init__("stub")
        self._classes = list(classes)
        self.active = None

    @property
    def classes(self):
        return list(self._classes)

    def load(self):
        self._loaded = True

    def warmup(self, imgsz=320):
        pass

    def prepare(self, prompts):
        validate_vocab(prompts, self._classes, self.name)
        return [self._classes.index(p) for p in dict.fromkeys(prompts)]

    def apply(self, prepared):
        self.active = prepared

    def infer(self, frame):
        return empty_detections()


@pytest.fixture(params=["fake", "fixed"])
def det(request):
    d = FakeDetector() if request.param == "fake" else FixedStub()
    d.load()
    return d


def vocab_of(d):
    return d.classes or ["person", "dog", "shoe"]


# 1. Vocabulary validation -----------------------------------------------
def test_open_vocab_reports_no_class_list():
    d = FakeDetector()
    assert d.open_vocab is True
    assert d.classes is None


def test_fixed_vocab_reports_its_real_list():
    d = FixedStub()
    assert d.open_vocab is False
    assert d.classes == ["person", "dog", "shoe"]


def test_fixed_vocab_rejects_a_class_it_cannot_produce():
    d = FixedStub()
    d.load()
    with pytest.raises(UnknownClassError) as e:
        d.prepare(["pencil"])
    # The orchestrator surfaces this to a human, so it has to name the culprit.
    assert "pencil" in str(e.value)


def test_fixed_vocab_rejects_a_partly_valid_spec():
    d = FixedStub()
    d.load()
    with pytest.raises(UnknownClassError):
        d.prepare(["person", "unicorn"])


def test_open_vocab_accepts_a_word_no_dataset_has():
    d = FakeDetector()
    d.load()
    assert d.prepare(["a slightly bent paperclip"])


def test_empty_prompt_list_is_rejected_everywhere(det):
    with pytest.raises(UnknownClassError):
        det.prepare([])


def test_vocab_check_is_case_sensitive():
    # COCO says "person". "Person" is a different string and must not slip past,
    # or the filter silently matches nothing at inference time.
    with pytest.raises(UnknownClassError):
        validate_vocab(["Person"], ["person"], "stub")


# 2. prepare is pure ------------------------------------------------------
def test_failed_prepare_leaves_the_detector_untouched():
    d = FixedStub()
    d.load()
    d.apply(d.prepare(["person"]))
    before = d.active
    with pytest.raises(UnknownClassError):
        d.prepare(["unicorn"])
    assert d.active == before


def test_prepare_alone_changes_nothing(det):
    baseline = getattr(det, "active_classes", None) or getattr(det, "active", None)
    det.prepare(vocab_of(det)[:1])
    after = getattr(det, "active_classes", None) or getattr(det, "active", None)
    assert after == baseline


def test_prepare_is_repeatable(det):
    a = det.prepare(vocab_of(det)[:2])
    b = det.prepare(vocab_of(det)[:2])
    assert a == b


def test_prepare_dedups_repeated_prompts(det):
    once = det.prepare(vocab_of(det)[:1])
    twice = det.prepare(vocab_of(det)[:1] * 3)
    assert len(twice) == len(once) == 1


# 3. infer output shape ---------------------------------------------------
def test_infer_always_carries_class_name(det, frame):
    det.apply(det.prepare(vocab_of(det)[:1]))
    assert "class_name" in det.infer(frame).data


def test_infer_before_any_spec_returns_empty_not_garbage(frame):
    d = FakeDetector()
    d.load()
    out = d.infer(frame)
    assert len(out) == 0 and "class_name" in out.data


def test_infer_arrays_stay_aligned(frame):
    d = FakeDetector()
    d.load()
    d.apply(d.prepare(["a", "b", "c"]))
    out = d.infer(frame)
    assert len(out.xyxy) == len(out.confidence) == len(out.data["class_name"]) == 3


def test_fake_boxes_stay_inside_the_frame(frame):
    d = FakeDetector()
    d.load()
    d.apply(d.prepare(["a", "b"]))
    h, w = frame.shape[:2]
    for _ in range(200):
        for x1, y1, x2, y2 in d.infer(frame).xyxy:
            assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h


def test_fake_actually_moves(frame):
    d = FakeDetector()
    d.load()
    d.apply(d.prepare(["a"]))
    first = d.infer(frame).xyxy.copy()
    for _ in range(10):
        later = d.infer(frame).xyxy
    assert not np.allclose(first, later)


def test_scripted_detector_repeats_its_last_frame(frame):
    d = FakeDetector()
    d.load()
    d.set_script([empty_detections()])
    assert all(len(d.infer(frame)) == 0 for _ in range(5))


# 4. Swapping -------------------------------------------------------------
def test_apply_replaces_rather_than_accumulates(frame):
    d = FakeDetector()
    d.load()
    d.apply(d.prepare(["a", "b"]))
    d.apply(d.prepare(["c"]))
    assert d.active_classes == ["c"]
    assert set(d.infer(frame).data["class_name"]) == {"c"}


def test_describe_is_honest_before_loading():
    d = FixedStub()
    assert d.describe()["loaded"] is False
    d.load()
    assert d.describe() == {
        "name": "stub",
        "open_vocab": False,
        "classes": ["person", "dog", "shoe"],
        "loaded": True,
    }
