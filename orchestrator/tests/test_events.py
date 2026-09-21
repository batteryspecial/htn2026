"""Event replays and camera recovery must not produce stale spoken alerts."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.runner import AgentRunner
from events.watcher import EventWatcher


def event(kind: str, *, ts: float | None = None, **fields) -> str:
    return json.dumps({"id": uuid.uuid4().hex, "type": kind,
                       "ts": time.time() if ts is None else ts,
                       "notify": True, **fields})


@pytest.fixture
async def watcher(perception, monkeypatch):
    monkeypatch.setattr("events.watcher.CAMERA_LOST_GRACE_S", 0)
    watcher = EventWatcher(SimpleNamespace(event_turn=AsyncMock()), perception)
    yield watcher
    await watcher.stop()


async def drain(watcher):
    await asyncio.gather(*list(watcher._pending.values()), return_exceptions=True)


async def test_camera_recovery_during_startup_prevents_a_loss_turn(watcher, monkeypatch, trace):
    monkeypatch.setattr("events.watcher.CAMERA_LOST_GRACE_S", 60)

    await watcher._handle(event("camera_lost", detail="no frame for infs"))
    await asyncio.sleep(0)  # The pending loss is now in its grace period.
    await watcher._handle(event("camera_ok", detail="camera back"))

    watcher.runner.event_turn.assert_not_awaited()
    assert not watcher._pending
    assert [entry.label for entry in trace] == ["camera_lost", "camera_ok"]


async def test_health_recheck_suppresses_loss_when_recovery_event_was_missed(watcher):
    await watcher._handle(event("camera_lost"))
    await drain(watcher)

    watcher.runner.event_turn.assert_not_awaited()


async def test_a_camera_that_stays_down_still_wakes_the_agent(watcher, pipeline):
    pipeline.health_payload.update(camera_ok=False, status="no_camera")

    await watcher._handle(event("camera_lost", detail="no recent frame"))
    await drain(watcher)

    watcher.runner.event_turn.assert_awaited_once()
    assert "no recent frame" in watcher.runner.event_turn.call_args.args[0]


async def test_inference_fault_is_not_hidden_by_a_working_camera(watcher, pipeline):
    # Perception also uses camera_lost for inference faults.
    pipeline.health_payload.update(camera_ok=True, status="fault")

    await watcher._handle(event("camera_lost", detail="inference failing"))
    await drain(watcher)

    watcher.runner.event_turn.assert_awaited_once()


async def test_recovery_cancels_an_alert_already_waiting_on_the_runner_lock(
        perception, memory, pipeline, monkeypatch):
    monkeypatch.setattr("events.watcher.CAMERA_LOST_GRACE_S", 0)
    pipeline.health_payload.update(camera_ok=False, status="no_camera")
    runner = AgentRunner(perception, memory)
    invoked = AsyncMock()
    monkeypatch.setattr(runner.graph, "ainvoke", invoked)
    watcher = EventWatcher(runner, perception)
    queued = asyncio.Event()
    original_event_turn = runner.event_turn

    async def observe_queued(*args, **kwargs):
        queued.set()
        return await original_event_turn(*args, **kwargs)

    monkeypatch.setattr(runner, "event_turn", observe_queued)
    try:
        async with runner._lock:
            await watcher._handle(event("camera_lost"))
            await asyncio.wait_for(queued.wait(), 1)
            await watcher._handle(event("camera_ok"))

        invoked.assert_not_awaited()
        assert not runner._tasks
        assert not watcher._pending
    finally:
        await watcher.stop()


async def test_recovery_cancels_a_running_alert(watcher, pipeline):
    pipeline.health_payload.update(camera_ok=False, status="no_camera")
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def pending_model(*_args):
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    watcher.runner.event_turn.side_effect = pending_model
    await watcher._handle(event("camera_lost"))
    await asyncio.wait_for(started.wait(), 1)
    await watcher._handle(event("camera_ok"))

    assert cancelled.is_set()
    assert not watcher._pending


async def test_an_older_replayed_recovery_does_not_cancel_a_new_loss(watcher, pipeline):
    pipeline.health_payload.update(camera_ok=False, status="no_camera")
    now = time.time()

    await watcher._handle(event("camera_lost", ts=now))
    await watcher._handle(event("camera_ok", ts=now - 10))
    await drain(watcher)

    watcher.runner.event_turn.assert_awaited_once()


async def test_old_events_are_logged_but_do_not_wake_and_replays_are_deduplicated(
        watcher, trace):
    old = event("near", ts=time.time() - 60, behavior_id="b1")

    await watcher._handle(old)
    await watcher._handle(old)

    watcher.runner.event_turn.assert_not_awaited()
    assert len(trace) == 1


async def test_a_new_loss_after_recovery_is_not_dropped_by_the_previous_cooldown(
        watcher, pipeline):
    pipeline.health_payload.update(camera_ok=False, status="no_camera")

    await watcher._handle(event("camera_lost"))
    await drain(watcher)
    await watcher._handle(event("camera_ok"))
    await watcher._handle(event("camera_lost"))
    await drain(watcher)

    assert watcher.runner.event_turn.await_count == 2
