"""Concurrency tests for the background pieces.

These are the parts that pass in a single-threaded test and then misbehave on
stage, so each is pushed until it would break: a consumer slower than the
producer, instructions faster than the worker, a video source never there.
"""

import asyncio
import contextlib
import threading
import time

import cv2
import numpy as np
import pytest

from contracts import BehaviorSpec
from zoo.registry import Registry
from runtime.workers import Builder
from runtime.capture import Capture
from runtime.events import Bus
from runtime.ops import Accepted, Applied, Rejected
from tests.rig import MODELS_YAML


# --- capture -------------------------------------------------------------
@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for i in range(20):
        w.write(np.full((48, 64, 3), (i * 10) % 256, np.uint8))
    w.release()
    return str(path)


def test_capture_reads_a_real_file(clip):
    c = Capture(clip).start()
    try:
        assert c.wait_for_frame(3.0)
        frame, ts, seq = c.read()
        assert frame.shape == (48, 64, 3) and seq >= 1 and ts > 0
    finally:
        c.stop()


def test_capture_loops_a_clip_forever(clip):
    c = Capture(clip).start()
    try:
        c.wait_for_frame(3.0)
        time.sleep(1.2)  # 20 frames at 30fps is well under a second
        assert c.read()[2] > 20
    finally:
        c.stop()


def test_a_slow_consumer_gets_the_newest_frame_not_a_backlog(clip):
    """Why capture keeps one slot. Acting on a stale frame is worse than
    acting on fewer of them."""
    c = Capture(clip).start()
    try:
        c.wait_for_frame(3.0)
        seqs = []
        for _ in range(4):
            time.sleep(0.25)
            seqs.append(c.read()[2])
        gaps = [b - a for a, b in zip(seqs, seqs[1:])]
        assert all(g > 1 for g in gaps), f"nothing was dropped: {seqs}"
    finally:
        c.stop()


def test_a_missing_source_does_not_kill_the_thread():
    c = Capture("/nope/not/a/file.mp4").start()
    try:
        time.sleep(0.3)
        assert c.read()[0] is None and c.age() == float("inf")
        assert c._thread.is_alive(), "capture thread died instead of retrying"
    finally:
        c.stop()


def test_stop_is_idempotent(clip):
    c = Capture(clip).start()
    c.wait_for_frame(3.0)
    c.stop()
    c.stop()
    assert not c._thread.is_alive()


# --- builder -------------------------------------------------------------
@pytest.fixture
def builder(tmp_path):
    p = tmp_path / "models.yaml"
    p.write_text(MODELS_YAML)
    reg = Registry.from_yaml(p)
    reg.preload()
    b = Builder(reg, default_model="fake").start()
    yield b
    b.stop()


def behavior(kind="highlight", detect=("thing",)):
    return BehaviorSpec(kind=kind, subject={"detect": list(detect)})


def drain(builder, quiet=0.25, timeout=5.0):
    """Collect outcomes until the worker has been silent for `quiet` seconds."""
    got, deadline, last = [], time.time() + timeout, time.time()
    while time.time() < deadline:
        batch = builder.poll()
        if batch:
            got.extend(batch)
            last = time.time()
        elif time.time() - last > quiet:
            break
        time.sleep(0.005)
    return got


def test_one_behavior_produces_one_applied_world(builder):
    builder.add_behavior(behavior())
    out = drain(builder)
    assert isinstance(out[0], Accepted)
    assert sum(isinstance(o, Applied) for o in out) == 1


def test_operations_accumulate_rather_than_replace(builder):
    """The core difference from the single-spec model: adding a behaviour
    keeps the ones already there."""
    ids = [builder.add_behavior(behavior()) for _ in range(5)]
    drain(builder)
    assert set(builder.world.behaviors) == set(ids)
    assert len(set(ids)) == 5, "ids must be unique"


def test_a_burst_of_instructions_all_land(builder):
    for i in range(20):
        builder.add_behavior(behavior())
    out = drain(builder)
    applied = [o for o in out if isinstance(o, Applied)]
    assert len(applied) == 20
    assert len(builder.world.behaviors) == 20


def test_concurrent_submits_never_lose_an_instruction(builder):
    """The agent is async; two tool calls can land at the same moment, and two
    threads must not be handed the same id."""
    ids, lock = [], threading.Lock()

    def send():
        got = builder.add_behavior(behavior())
        with lock:
            ids.append(got)

    threads = [threading.Thread(target=send) for _ in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    out = drain(builder)
    assert len(set(ids)) == 30, "two behaviours were given the same id"
    accepted = {o.op.seq for o in out if isinstance(o, Accepted)}
    answered = {o.op.seq for o in out if isinstance(o, (Applied, Rejected))}
    assert len(accepted) == 30
    assert accepted == answered, f"no answer for {accepted - answered}"
    assert set(builder.world.behaviors) == set(ids)


def test_a_rejected_behavior_leaves_the_world_alone(builder):
    good = builder.add_behavior(behavior())
    drain(builder)
    builder.add_behavior(behavior(kind="keyboard"))
    out = drain(builder)
    assert any(isinstance(o, Rejected) for o in out)
    assert set(builder.world.behaviors) == {good}


def test_a_failing_build_does_not_kill_the_worker(builder, monkeypatch):  # noqa: D103
    det = builder.registry.get("fake")
    monkeypatch.setattr(type(det), "prepare",
                        lambda self, p: (_ for _ in ()).throw(ValueError("unknown class")))
    builder.add_behavior(behavior())
    assert any(isinstance(o, Rejected) for o in drain(builder))

    monkeypatch.undo()
    builder.add_behavior(behavior())
    assert any(isinstance(o, Applied) for o in drain(builder)), "worker died"


def test_a_model_swap_asks_for_a_tracking_reset(builder):
    """Tracker ids from a different detector mean nothing."""
    builder.add_behavior(behavior())
    drain(builder)
    builder.set_model("other")
    applied = [o for o in drain(builder) if isinstance(o, Applied)]
    assert applied and applied[-1].reset_tracking is True


def test_adding_a_behavior_does_not_ask_for_a_reset(builder):
    builder.add_behavior(behavior())
    drain(builder)
    builder.add_behavior(behavior())
    applied = [o for o in drain(builder) if isinstance(o, Applied)]
    assert applied and applied[-1].reset_tracking is False


def test_swapping_to_the_model_already_in_use_is_refused(builder):
    builder.add_behavior(behavior())
    drain(builder)
    builder.set_model("fake")
    assert any(isinstance(o, Rejected) and "already" in o.reason for o in drain(builder))


def test_poll_never_blocks_when_idle(builder):
    t = time.perf_counter()
    assert builder.poll() == []
    assert time.perf_counter() - t < 0.05


# --- event bus -----------------------------------------------------------
def test_publish_without_a_consumer_is_safe():
    b = Bus(history=5)
    for i in range(10):
        b.publish(i)
    assert b.history() == [5, 6, 7, 8, 9]


def test_a_late_subscriber_gets_the_backlog():
    async def go():
        b = Bus(history=10)
        b.attach(asyncio.get_running_loop())
        for i in range(3):
            b.publish(i)
        stream = b.stream()
        return [await anext(stream) for _ in range(3)]

    assert asyncio.run(go()) == [0, 1, 2]


def test_a_slow_state_consumer_gets_the_newest_not_the_oldest():
    """State is level-triggered: a stale frame is worse than a gap."""
    async def go():
        b = Bus(latest_only=True)
        b.attach(asyncio.get_running_loop())
        stream = b.stream()
        pending = asyncio.ensure_future(anext(stream))
        await asyncio.sleep(0.02)
        b.publish(0)
        first = await pending
        for i in range(1, 101):
            b.publish(i)
        await asyncio.sleep(0.05)
        return first, await anext(stream), b.dropped

    first, latest, dropped = asyncio.run(go())
    assert first == 0 and latest == 100 and dropped == 99


def test_publishing_from_another_thread_reaches_the_loop():
    async def go():
        b = Bus()
        b.attach(asyncio.get_running_loop())
        stream = b.stream()
        got = asyncio.ensure_future(anext(stream))
        await asyncio.sleep(0.02)
        threading.Thread(target=lambda: b.publish("from a thread")).start()
        return await asyncio.wait_for(got, 2.0)

    assert asyncio.run(go()) == "from a thread"


def test_a_disconnected_client_is_forgotten():
    async def go():
        b = Bus()
        b.attach(asyncio.get_running_loop())
        stream = b.stream()
        pending = asyncio.ensure_future(anext(stream))
        await asyncio.sleep(0.02)
        assert b.subscribers == 1
        pending.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await pending
        await stream.aclose()
        return b.subscribers

    assert asyncio.run(go()) == 0, "a closed websocket left its queue behind"
