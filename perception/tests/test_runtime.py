"""Concurrency tests for the three background pieces.

These are the parts that work perfectly in a single-threaded test and then
misbehave on stage, so each one is pushed until it would break: a consumer
slower than the producer, a burst of instructions faster than the worker, a
video source that was never there.
"""

import asyncio
import contextlib
import threading
import time

import cv2
import numpy as np
import pytest

from contracts import TaskSpec
from detectors.registry import Registry
from runtime.capture import Capture
from runtime.events import Bus
from runtime.loader import Accepted, Failed, Loader, Prepared


# --- capture -------------------------------------------------------------
@pytest.fixture
def clip(tmp_path):
    """A 20-frame clip whose pixel value encodes its frame number."""
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
        time.sleep(1.2)  # the clip is 20 frames at 30fps, well under a second
        assert c.read()[2] > 20
    finally:
        c.stop()


def test_a_slow_consumer_gets_the_newest_frame_not_a_backlog(clip):
    """The whole reason capture keeps one slot. A consumer that falls behind
    must skip ahead, because steering from a stale frame is worse than
    steering from fewer of them."""
    c = Capture(clip).start()
    try:
        c.wait_for_frame(3.0)
        seqs = []
        for _ in range(4):
            time.sleep(0.25)
            seqs.append(c.read()[2])
        gaps = [b - a for a, b in zip(seqs, seqs[1:])]
        assert all(g > 1 for g in gaps), f"consumer saw every frame, so nothing was dropped: {seqs}"
    finally:
        c.stop()


def test_a_missing_source_does_not_kill_the_thread():
    c = Capture("/nope/not/a/file.mp4").start()
    try:
        time.sleep(0.3)
        assert c.read()[0] is None
        assert c.age() == float("inf")
        assert c._thread.is_alive(), "capture thread died instead of retrying"
    finally:
        c.stop()


def test_stop_is_idempotent_and_quick(clip):
    c = Capture(clip).start()
    c.wait_for_frame(3.0)
    c.stop()
    c.stop()
    assert not c._thread.is_alive()


# --- loader --------------------------------------------------------------
@pytest.fixture
def loader(tmp_path):
    p = tmp_path / "models.yaml"
    p.write_text("models:\n  - name: fake\n    type: fake\n    preload: true\n")
    reg = Registry.from_yaml(p)
    reg.preload()
    ld = Loader(reg).start()
    yield ld
    ld.stop()


def spec(spec_id="s1", detect=("pencil",)):
    return TaskSpec(spec_id=spec_id, model="fake",
                    targets=[{"ref": "t1", "detect": list(detect)}])


def drain(loader, quiet=0.25, timeout=5.0):
    """Collect outcomes until the worker has been silent for `quiet` seconds."""
    got, deadline, last = [], time.time() + timeout, time.time()
    while time.time() < deadline:
        batch = loader.poll()
        if batch:
            got.extend(batch)
            last = time.time()
        elif time.time() - last > quiet:
            break
        time.sleep(0.005)
    return got


def test_a_single_spec_produces_one_prepared_task(loader):
    loader.submit("i1", spec())
    out = drain(loader)
    assert isinstance(out[0], Accepted)
    assert sum(isinstance(o, Prepared) for o in out) == 1


def test_a_burst_of_specs_applies_only_the_last(loader):
    """An operator retyping an instruction must not queue up four retasks."""
    for i in range(20):
        loader.submit(f"i{i}", spec(f"s{i}"))
    out = drain(loader)

    prepared = [o for o in out if isinstance(o, Prepared)]
    assert len(prepared) == 1
    assert prepared[0].job.spec.spec_id == "s19"

    superseded = [o for o in out if isinstance(o, Failed) and o.reason == "superseded"]
    accepted = [o for o in out if isinstance(o, Accepted)]
    assert len(accepted) == 20
    # Every instruction gets exactly one terminal answer: prepared or superseded.
    assert len(superseded) + len(prepared) == 20


def test_concurrent_submits_never_lose_an_instruction(loader):
    """Danny's orchestrator is async; two instructions can land at once."""
    def send(i):
        loader.submit(f"i{i}", spec(f"s{i}"))

    threads = [threading.Thread(target=send, args=(i,)) for i in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    out = drain(loader)
    accepted = {o.job.instruction_id for o in out if isinstance(o, Accepted)}
    answered = {o.job.instruction_id for o in out if isinstance(o, (Prepared, Failed))}
    assert len(accepted) == 30
    assert accepted == answered, f"no answer for {accepted - answered}"


def test_a_failing_prepare_does_not_kill_the_worker(loader, monkeypatch):
    det = loader.registry.get("fake")

    def explode(self, prompts):
        raise ValueError("unknown class: unicorn")
    monkeypatch.setattr(type(det), "prepare", explode)
    loader.submit("bad", spec())
    out = drain(loader)
    assert any(isinstance(o, Failed) and "unicorn" in o.reason for o in out)

    monkeypatch.undo()
    loader.submit("good", spec("s2"))
    out = drain(loader)
    assert any(isinstance(o, Prepared) for o in out), "worker died on the first failure"


def test_failure_reason_reaches_the_operator_intact(loader, monkeypatch):
    det = loader.registry.get("fake")
    monkeypatch.setattr(
        type(det), "prepare",
        lambda self, p: (_ for _ in ()).throw(ValueError("coco does not know 'pencil'")),
    )
    loader.submit("i1", spec())
    failed = [o for o in drain(loader) if isinstance(o, Failed)]
    assert failed and "pencil" in failed[0].reason


def test_poll_never_blocks_when_idle(loader):
    t = time.perf_counter()
    assert loader.poll() == []
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


def test_a_slow_target_consumer_gets_the_newest_not_the_oldest():
    """Target states are level-triggered: a stale one is worse than a gap."""
    async def go():
        b = Bus(latest_only=True)
        b.attach(asyncio.get_running_loop())
        stream = b.stream()
        pending = asyncio.ensure_future(anext(stream))
        await asyncio.sleep(0.02)  # generator registered and waiting

        b.publish(0)
        first = await pending
        # Now nobody is reading, and the producer keeps going.
        for i in range(1, 101):
            b.publish(i)
        await asyncio.sleep(0.05)
        return first, await anext(stream), b.dropped

    first, latest, dropped = asyncio.run(go())
    assert first == 0
    assert latest == 100, "a slow consumer was handed a stale target position"
    assert dropped == 99


def test_publishing_from_another_thread_reaches_the_loop():
    async def go():
        b = Bus(history=0)
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
