"""The state machine is what stops the car. Every row of the transition table
in CLAUDE.md gets a test, plus the debouncing that surrounds it."""

import pytest

from contracts import Phase
from runtime.state import Machine


@pytest.fixture
def m():
    """A machine that has booted, with every event captured."""
    events = []
    mach = Machine(emit=events.append)
    mach.events = events
    mach.on_registry_ready()
    return mach


def stages(m):
    return [e.stage for e in m.events]


def apply_spec(m, t=0.0, iid="i1", sid="s1"):
    """Shortcut through the accept -> prepare -> swap path."""
    m.on_spec(iid, sid)
    m.on_prepared()
    m.on_applied(t)


def see(m, n, visible=True, t0=0.0, dt=0.05):
    for i in range(n):
        m.on_frame(visible, t0 + i * dt)


# 1. Boot
def test_registry_ready_goes_idle(m):
    assert m.phase is Phase.IDLE


def test_boot_failure_goes_fault():
    m = Machine()
    m.on_boot_failed("yoloe missing")
    assert m.phase is Phase.FAULT


# 2. Spec lifecycle
def test_spec_to_tracking_emits_ordered_events(m):
    apply_spec(m)
    see(m, 3)
    assert m.phase is Phase.TRACKING
    assert stages(m) == ["prepared", "applied", "active"]


def test_cold_model_inserts_loading_phase(m):
    m.on_spec("i1", "s1")
    m.on_model_loading("coco_trt")
    assert m.phase is Phase.LOADING_MODEL
    m.on_model_loaded("coco_trt")
    assert m.phase is Phase.SWITCHING
    assert stages(m) == ["model_loading", "model_loaded"]


def test_prepare_failure_reverts_and_keeps_old_task(m):
    apply_spec(m)
    see(m, 3)
    assert m.phase is Phase.TRACKING

    m.on_spec("i2", "s2")
    m.on_prepare_failed("unknown class: unicorn")
    # Reverts to the phase it was in, so the old spec keeps driving.
    assert m.phase is Phase.TRACKING
    assert stages(m)[-1] == "failed"
    assert m.events[-1].detail == "unknown class: unicorn"


def test_load_failure_from_loading_model_also_reverts(m):
    apply_spec(m)
    see(m, 3)
    m.on_spec("i2", "s2")
    m.on_model_loading("rfdetr")
    m.on_prepare_failed("weights not found")
    assert m.phase is Phase.TRACKING


def test_superseded_spec_is_reported_against_the_loser(m):
    m.on_spec("i1", "s1")
    m.on_superseded("i1", "s1")
    m.on_spec("i2", "s2")
    failed = [e for e in m.events if e.stage == "failed"]
    assert len(failed) == 1
    assert failed[0].instruction_id == "i1"
    assert failed[0].detail == "superseded"


def test_delete_spec_goes_idle(m):
    apply_spec(m)
    see(m, 3)
    m.on_clear_spec()
    assert m.phase is Phase.IDLE
    assert m.spec_id is None


# 3. Acquire / track / lose
def test_acquire_needs_consecutive_hits(m):
    apply_spec(m)
    m.on_frame(True, 0.0)
    m.on_frame(False, 0.05)  # breaks the streak
    m.on_frame(True, 0.10)
    m.on_frame(True, 0.15)
    assert m.phase is Phase.ACQUIRING
    m.on_frame(True, 0.20)
    assert m.phase is Phase.TRACKING


def test_acquire_times_out_to_no_target(m):
    apply_spec(m, t=0.0)
    m.on_frame(False, 2.9)
    assert m.phase is Phase.ACQUIRING
    m.on_frame(False, 3.0)
    assert m.phase is Phase.NO_TARGET
    assert stages(m)[-1] == "no_target"


def test_no_target_still_reacquires(m):
    apply_spec(m, t=0.0)
    m.on_frame(False, 3.0)
    assert m.phase is Phase.NO_TARGET
    see(m, 3, t0=4.0)
    assert m.phase is Phase.TRACKING


def test_tracking_survives_a_short_dropout(m):
    apply_spec(m)
    see(m, 3)
    see(m, 4, visible=False, t0=1.0)
    assert m.phase is Phase.TRACKING
    m.on_frame(False, 1.5)
    assert m.phase is Phase.LOST


def test_lost_reacquires_without_a_second_active_event(m):
    apply_spec(m)
    see(m, 3)
    see(m, 5, visible=False, t0=1.0)
    assert m.phase is Phase.LOST
    see(m, 3, t0=2.0)
    assert m.phase is Phase.TRACKING
    # `active` is the retask-time metric: once per spec, not once per reacquire.
    assert stages(m).count("active") == 1


def test_new_spec_rearms_the_active_event(m):
    apply_spec(m)
    see(m, 3)
    apply_spec(m, t=5.0, iid="i2", sid="s2")
    see(m, 3, t0=5.0)
    assert stages(m).count("active") == 2


def test_frames_are_ignored_while_switching(m):
    m.on_spec("i1", "s1")
    see(m, 10)
    assert m.phase is Phase.SWITCHING


# 4. Faults
def test_fault_after_consecutive_inference_errors(m):
    apply_spec(m)
    see(m, 3)
    m.on_infer_error("cuda oom", 1.0)
    m.on_infer_error("cuda oom", 1.1)
    assert m.phase is Phase.TRACKING
    m.on_infer_error("cuda oom", 1.2)
    assert m.phase is Phase.FAULT


def test_one_good_frame_clears_the_error_streak(m):
    apply_spec(m)
    see(m, 3)
    m.on_infer_error("blip", 1.0)
    m.on_infer_error("blip", 1.1)
    m.on_infer_ok()
    m.on_infer_error("blip", 1.2)
    assert m.phase is Phase.TRACKING


def test_recovery_reacquires_when_a_spec_is_loaded(m):
    apply_spec(m)
    see(m, 3)
    m.on_fault("camera dead", 1.0)
    assert m.phase is Phase.FAULT
    m.on_recovered(2.0)
    assert m.phase is Phase.ACQUIRING


def test_recovery_without_a_spec_goes_idle(m):
    m.on_fault("camera dead", 1.0)
    m.on_recovered(2.0)
    assert m.phase is Phase.IDLE


def test_spec_during_fault_is_held_not_applied(m):
    m.on_fault("camera dead", 1.0)
    m.on_spec("i1", "s1")
    assert m.phase is Phase.FAULT
    m.on_recovered(2.0)
    assert m.phase is Phase.ACQUIRING
    assert m.spec_id == "s1"


def test_frames_during_fault_never_reach_tracking(m):
    apply_spec(m)
    m.on_fault("camera dead", 1.0)
    see(m, 20, t0=2.0)
    assert m.phase is Phase.FAULT
