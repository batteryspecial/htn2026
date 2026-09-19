"""Tests that load real weights. Skipped by default.

    PERCEPTION_TEST_WEIGHTS=1 DEVICE=cpu IMGSZ=320 pytest tests/test_detectors_real.py

This is the file to run first on the GPU host after a deploy: it checks the one
assumption the whole architecture rests on, that YOLOE's vocabulary can be
swapped at runtime and that swapping is cheap on the loop thread.
"""

import time

import numpy as np
import pytest

from detectors.base import UnknownClassError
from zoo.registry import Registry

pytestmark = pytest.mark.weights


@pytest.fixture(scope="module")
def registry():
    return Registry.from_yaml()


@pytest.fixture(scope="module")
def yoloe(registry):
    return registry.get("yoloe")


@pytest.fixture(scope="module")
def coco(registry):
    return registry.get("coco")


def test_yoloe_accepts_a_vocabulary_no_dataset_ships(yoloe):
    prepared = yoloe.prepare(["pencil", "eraser"])
    assert prepared.names == ["pencil", "eraser"]


def test_yoloe_vocabulary_actually_swaps(yoloe, frame):
    yoloe.apply(yoloe.prepare(["pencil", "dog"]))
    assert yoloe.active_classes == ["pencil", "dog"]
    yoloe.apply(yoloe.prepare(["eraser"]))
    assert yoloe.active_classes == ["eraser"]
    yoloe.infer(frame)


def test_applying_a_vocabulary_is_cheap_enough_for_one_frame(yoloe):
    """The whole two-thread design exists because prepare is slow and apply is
    not. If that stops being true, the swap belongs somewhere else."""
    prepared = yoloe.prepare(["pencil"])
    t = time.perf_counter()
    yoloe.apply(prepared)
    assert time.perf_counter() - t < 0.02


def test_yoloe_output_is_box_shaped_and_named(yoloe, frame):
    yoloe.apply(yoloe.prepare(["person"]))
    out = yoloe.infer(frame)
    assert "class_name" in out.data
    assert out.mask is None  # masks dropped; the Cutie stretch would keep them
    assert out.xyxy.shape[1] == 4


def test_coco_knows_its_own_vocabulary(coco):
    assert coco.classes and "person" in coco.classes
    assert len(coco.classes) == 80


def test_coco_refuses_a_prompt_outside_its_vocabulary(coco):
    with pytest.raises(UnknownClassError):
        coco.prepare(["pencil"])


def test_coco_filters_to_the_requested_classes(coco, frame):
    coco.apply(coco.prepare(["person"]))
    out = coco.infer(frame)
    names = set(out.data["class_name"])
    assert names <= {"person"}


def test_both_detectors_agree_on_output_shape(yoloe, coco, frame):
    """The loop must not be able to tell which model produced a frame."""
    yoloe.apply(yoloe.prepare(["person"]))
    coco.apply(coco.prepare(["person"]))
    a, b = yoloe.infer(frame), coco.infer(frame)
    assert type(a) is type(b)
    assert set(a.data) >= {"class_name"} and set(b.data) >= {"class_name"}


def test_registry_boots_with_every_configured_model(registry):
    failed = registry.preload()
    assert failed == [], f"preload failed for {failed}"


def test_manifest_lists_a_usable_open_vocab_model(registry):
    manifest = registry.manifest()
    assert any(m["open_vocab"] and m["loaded"] for m in manifest)
