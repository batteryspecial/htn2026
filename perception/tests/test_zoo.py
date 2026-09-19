"""The model zoo: roles, availability, and switching.

The point of the zoo is that adding a model is a config edit. These tests are
mostly about what happens when a model *cannot* run, because that is the case
the laptop and the GPU host disagree on and both have to boot.
"""

import textwrap

import pytest

from zoo.registry import (
    Entry,
    ModelUnavailableError,
    NoModelForRole,
    Registry,
    UnknownModelError,
)
from zoo.roles import KINDS, ROLES, types_for


def write(tmp_path, body):
    p = tmp_path / "models.yaml"
    p.write_text(textwrap.dedent(body))
    return p


@pytest.fixture
def reg(tmp_path):
    return Registry.from_yaml(write(tmp_path, """
        models:
          - name: fake
            role: detector
            type: fake
            preload: true
          - name: coco_like
            role: detector
            type: fake
          - name: hasher
            role: embedder
            type: hash_encoder
            preload: true
          - name: gpu_only
            role: detector
            type: ultralytics_fixed
            weights: nope.engine
            requires_cuda: true
          - name: wrong_role
            role: pose
            type: fake
          - name: nonsense
            role: detector
            type: not_a_real_type
        """))


# 1. Roles ----------------------------------------------------------------
def test_every_registered_type_declares_a_known_role():
    assert all(k.role in ROLES for k in KINDS.values())


def test_each_role_can_be_listed():
    assert "yoloe" in types_for("detector")
    assert "open_clip" in types_for("embedder")
    assert "ultralytics_pose" in types_for("pose")


def test_a_role_defaults_to_detector():
    """So configs written before roles existed keep working."""
    assert Entry(name="x", type="fake").role == "detector"


def test_a_type_in_the_wrong_role_is_caught_at_boot(reg):
    """A typo, found now rather than when a behaviour asks for a pose model
    and is handed a detector."""
    e = reg.entry("wrong_role")
    assert e.available is False
    assert "fills role 'detector'" in e.unavailable_reason


def test_a_missing_python_package_is_a_reason_not_a_crash(tmp_path):
    r = Registry.from_yaml(write(tmp_path, """
        models:
          - name: ocr
            role: ocr
            type: definitely_not_installed
        """))
    assert r.entry("ocr").available is False
    assert "unknown type" in r.entry("ocr").unavailable_reason


# 2. Booting with broken entries ------------------------------------------
def test_boot_survives_models_it_cannot_run(reg):
    assert reg.preload() == []
    assert reg.is_loaded("fake") and reg.is_loaded("hasher")


def test_unrunnable_entries_are_marked_not_hidden(reg):
    by_name = {m["name"]: m for m in reg.manifest()}
    assert by_name["gpu_only"]["available"] is False
    assert "reason" in by_name["gpu_only"]
    assert by_name["nonsense"]["available"] is False


def test_every_role_reports_what_fills_it(reg):
    roles = reg.roles()
    assert roles["detector"]["active"] in ("fake", "coco_like")
    assert roles["embedder"]["active"] == "hasher"
    assert roles["pose"]["active"] is None
    assert roles["ocr"]["available"] == []


def test_an_unfilled_role_is_answerable_without_raising(reg):
    """Behaviours ask this before running, every frame."""
    assert reg.has_role("detector") and reg.has_role("embedder")
    assert not reg.has_role("pose") and not reg.has_role("ocr")


# 3. Getting a model -------------------------------------------------------
def test_getting_an_unfilled_role_says_what_is_missing(reg):
    with pytest.raises(NoModelForRole) as e:
        reg.get_active("ocr")
    assert "ocr" in str(e.value)


def test_a_role_configured_but_unusable_says_so(reg):
    with pytest.raises(NoModelForRole) as e:
        reg.get_active("pose")
    assert "wrong_role" in str(e.value), "should name the model that could not run"


def test_try_get_active_degrades_instead_of_raising(reg):
    """The pipeline runs without an embedder; attributes simply never match."""
    assert reg.try_get_active("ocr") is None
    assert reg.try_get_active("embedder") is not None


def test_an_unavailable_model_is_refused(reg):
    with pytest.raises(ModelUnavailableError):
        reg.get("gpu_only")


def test_an_unknown_model_names_the_alternatives(reg):
    with pytest.raises(UnknownModelError) as e:
        reg.get("rfdetr")
    assert "fake" in str(e.value)


# 4. Switching -------------------------------------------------------------
def test_switching_within_a_role(reg):
    reg.set_active("detector", "coco_like")
    assert reg.active("detector") == "coco_like"


def test_switching_a_model_into_the_wrong_role_is_refused(reg):
    with pytest.raises(ModelUnavailableError) as e:
        reg.set_active("pose", "fake")
    assert "is a detector" in str(e.value)


def test_switching_to_an_unavailable_model_is_refused(reg):
    with pytest.raises(ModelUnavailableError):
        reg.set_active("detector", "gpu_only")


def test_roles_are_independent(reg):
    """Swapping the pose model must not disturb the detector."""
    before = reg.active("embedder")
    reg.set_active("detector", "coco_like")
    assert reg.active("embedder") == before


def test_a_failed_preload_hands_the_role_to_something_else(tmp_path, monkeypatch):
    """A role whose chosen model just failed should not still claim to be
    filled, or every behaviour needing it breaks silently."""
    r = Registry.from_yaml(write(tmp_path, """
        models:
          - name: broken
            role: detector
            type: fake
            preload: true
          - name: spare
            role: detector
            type: fake
            preload: true
        """))
    r.set_active("detector", "broken")

    def explode():
        raise OSError("no weights")

    # Patch the instance, not the class: both entries are the same type.
    r.entry("broken").model.load = explode
    assert r.preload() == ["broken"]
    assert r.active("detector") == "spare"


# 5. The manifest the agent reads ------------------------------------------
def test_the_manifest_says_which_model_is_active(reg):
    active = [m for m in reg.manifest() if m["active"]]
    assert {m["role"] for m in active} == {"detector", "embedder"}


def test_only_detectors_report_a_vocabulary(reg):
    by_name = {m["name"]: m for m in reg.manifest()}
    assert "open_vocab" in by_name["fake"]
    assert "open_vocab" not in by_name["hasher"]


def test_adding_a_model_is_only_a_config_edit(tmp_path):
    """The claim the zoo exists to make. A second pose model is one entry."""
    r = Registry.from_yaml(write(tmp_path, """
        models:
          - name: pose_a
            role: pose
            type: ultralytics_pose
            weights: yolo11n-pose.pt
          - name: pose_b
            role: pose
            type: ultralytics_pose
            weights: yolo11s-pose.pt
        """))
    assert sorted(r.names("pose")) == ["pose_a", "pose_b"]
    assert r.has_role("pose")
    r.set_active("pose", "pose_b")
    assert r.active("pose") == "pose_b"
