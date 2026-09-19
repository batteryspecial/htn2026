"""Registry tests.

Two things matter here and they pull in opposite directions: the service must
never refuse to boot because a model is missing, and it must never quietly hand
the pipeline a model that cannot run.
"""

import textwrap

import pytest

from detectors.registry import (
    Entry,
    ModelUnavailableError,
    Registry,
    UnknownModelError,
)


def write_yaml(tmp_path, body):
    p = tmp_path / "models.yaml"
    p.write_text(textwrap.dedent(body))
    return p


@pytest.fixture
def reg(tmp_path):
    return Registry.from_yaml(
        write_yaml(
            tmp_path,
            """
            models:
              - name: fake
                type: fake
                preload: true
              - name: gpu_only
                type: ultralytics_fixed
                weights: whatever.engine
                requires_cuda: true
                preload: true
              - name: nonsense
                type: not_a_real_type
                preload: true
            """,
        )
    )


# 1. Lookup ---------------------------------------------------------------
def test_unknown_model_names_what_is_available(reg):
    with pytest.raises(UnknownModelError) as e:
        reg.entry("rfdetr")
    assert "rfdetr" in str(e.value) and "fake" in str(e.value)


def test_membership_does_not_raise(reg):
    assert "fake" in reg and "rfdetr" not in reg


# 2. Boot never fails -----------------------------------------------------
def test_boot_survives_a_model_it_cannot_run(reg):
    # gpu_only and nonsense are both unusable; the registry still came up.
    assert reg.preload() == []
    assert reg.is_loaded("fake")


def test_unrunnable_entries_are_marked_not_hidden(reg):
    names = {m["name"]: m for m in reg.manifest()}
    assert names["gpu_only"]["available"] is False
    assert names["nonsense"]["available"] is False
    assert "reason" in names["gpu_only"]


def test_unknown_type_is_a_config_error_not_a_crash(reg):
    assert "unknown type" in reg.entry("nonsense").unavailable_reason


# 3. Unavailable models are refused, not fudged ---------------------------
def test_getting_an_unavailable_model_raises(reg):
    with pytest.raises(ModelUnavailableError):
        reg.get("gpu_only")


def test_a_failed_load_is_reported_and_stays_unloaded(tmp_path, monkeypatch):
    r = Registry.from_yaml(
        write_yaml(tmp_path, "models:\n  - name: fake\n    type: fake\n")
    )
    def boom(self):
        raise OSError("weights corrupt")
    monkeypatch.setattr(type(r.entry("fake").detector), "load", boom)

    with pytest.raises(OSError):
        r.get("fake")
    assert r.is_loaded("fake") is False
    assert "weights corrupt" in r.entry("fake").describe()["error"]


def test_preload_reports_failures_without_raising(tmp_path, monkeypatch):
    r = Registry.from_yaml(
        write_yaml(tmp_path, "models:\n  - name: fake\n    type: fake\n    preload: true\n")
    )
    monkeypatch.setattr(
        type(r.entry("fake").detector), "load", lambda self: (_ for _ in ()).throw(OSError("nope"))
    )
    assert r.preload() == ["fake"]


def test_a_second_get_after_success_does_not_reload(reg):
    first = reg.get("fake")
    calls = []
    type(first).load = lambda self: calls.append(1)
    assert reg.get("fake") is first
    assert calls == []


# 4. Manifest is the orchestrator's source of truth -----------------------
def test_manifest_reports_capability_even_when_unloaded(reg):
    fake = next(m for m in reg.manifest() if m["name"] == "fake")
    assert fake["open_vocab"] is True


def test_manifest_covers_every_configured_model(reg):
    assert {m["name"] for m in reg.manifest()} == set(reg.names())


# 5. Weights paths --------------------------------------------------------
def test_bare_filename_resolves_under_the_weights_dir():
    from config import CFG

    assert Entry(name="x", type="fake", weights="a.pt").weights_path == CFG.WEIGHTS_DIR / "a.pt"


def test_explicit_path_is_left_alone(tmp_path):
    e = Entry(name="x", type="fake", weights=str(tmp_path / "a.pt"))
    assert e.weights_path == tmp_path / "a.pt"


def test_missing_engine_is_unavailable_rather_than_a_late_crash(tmp_path):
    e = Entry(name="trt", type="ultralytics_fixed", weights=str(tmp_path / "gone.engine"))
    e.check_available("cuda")
    assert e.available is False and "not built" in e.unavailable_reason
