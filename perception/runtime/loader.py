"""The background worker that builds a new task while the old one keeps running.

Everything expensive about a retask happens here: loading a model that is not
resident, encoding the prompt vocabulary, building a fresh tracker. The
inference loop never waits for any of it.

Two rules give the whole hot-swap its safety:

1. **The newest spec wins.** There is one pending slot, not a queue. A spec
   that arrives mid-prepare replaces whatever was queued, and the displaced one
   is reported as `superseded` so the UI stops waiting on it. Under a human
   typing instructions quickly, the only thing that matters is the last one.
2. **A failure is a no-op.** Nothing is handed over until it is complete, so a
   bad model name or an unknown class costs a `failed` event and nothing else.
   The car keeps following whatever it was already following.

The worker never touches the phase or the active task. It posts outcomes to a
queue that the inference loop drains at the top of a frame, which keeps the
loop the only writer of pipeline state.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

from contracts import TaskSpec
from detectors.registry import Registry
from runtime.task import PreparedTask

log = logging.getLogger("perception.loader")


@dataclass(frozen=True)
class Job:
    instruction_id: str
    spec: TaskSpec
    gen: int
    submitted_ts: float


# Outcomes, drained by the loop. Each maps onto one state machine trigger.
@dataclass(frozen=True)
class Accepted:
    """A spec passed validation and is now queued. Posted synchronously by
    submit() so the phase changes on the loop thread, not the API thread."""

    job: Job


@dataclass(frozen=True)
class Cleared:
    """DELETE /spec. Routed through the same queue so stopping the car is a
    loop-thread decision like every other one."""


@dataclass(frozen=True)
class ModelLoading:
    job: Job
    model: str


@dataclass(frozen=True)
class ModelLoaded:
    job: Job
    model: str
    seconds: float


@dataclass(frozen=True)
class Prepared:
    job: Job
    task: PreparedTask
    seconds: float


@dataclass(frozen=True)
class Failed:
    job: Job
    reason: str


Outcome = Accepted | Cleared | ModelLoading | ModelLoaded | Prepared | Failed


class Loader:
    def __init__(self, registry: Registry) -> None:
        self.registry = registry
        self._pending: Job | None = None
        self._gen = 0
        self._cv = threading.Condition()
        self._out: queue.Queue[Outcome] = queue.Queue()
        self._stop = False
        self._thread: threading.Thread | None = None
        self.prepared_count = 0

    # 1. Producing work -------------------------------------------------
    def submit(self, instruction_id: str, spec: TaskSpec) -> None:
        """Called from the API thread. Returns immediately."""
        with self._cv:
            self._gen += 1
            job = Job(instruction_id=instruction_id, spec=spec, gen=self._gen,
                      submitted_ts=time.time())
            displaced, self._pending = self._pending, job
            if displaced is not None:
                # Never even started. Tell its owner now rather than leaving
                # the status board waiting on something that will never come.
                self._out.put(Failed(displaced, "superseded"))
            # Queued inside the lock so the worker cannot post a result for
            # this job before the loop has seen that it was accepted.
            self._out.put(Accepted(job))
            self._cv.notify()

    def clear(self) -> None:
        """Drop the current spec. Cancels anything pending on the way."""
        with self._cv:
            self._gen += 1
            displaced, self._pending = self._pending, None
            if displaced is not None:
                self._out.put(Failed(displaced, "superseded"))
            self._out.put(Cleared())

    # 2. Consuming results ----------------------------------------------
    def poll(self) -> list[Outcome]:
        """Called from the inference loop. Never blocks."""
        out = []
        while True:
            try:
                out.append(self._out.get_nowait())
            except queue.Empty:
                return out

    # 3. Worker ----------------------------------------------------------
    def start(self) -> Loader:
        self._thread = threading.Thread(target=self._run, name="loader", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()
        if self._thread:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._pending is None and not self._stop:
                    self._cv.wait()
                if self._stop:
                    return
                job, self._pending = self._pending, None
            self._process(job)

    def _process(self, job: Job) -> None:
        started = time.perf_counter()
        try:
            model = job.spec.model
            if not self.registry.is_loaded(model):
                # Cold path. Only a fallback: everything available is preloaded
                # at boot, so this should not happen during a demo.
                self._out.put(ModelLoading(job, model))
                t0 = time.perf_counter()
                detector = self.registry.get(model)
                self._out.put(ModelLoaded(job, model, time.perf_counter() - t0))
            else:
                detector = self.registry.get(model)

            prepared = detector.prepare(job.spec.prompt_union())
            task = PreparedTask.build(job.instruction_id, job.spec, model, detector, prepared)
        except Exception as exc:
            log.warning("prepare failed for %s: %s", job.instruction_id, exc)
            self._out.put(Failed(job, f"{type(exc).__name__}: {exc}"))
            return

        if self._is_stale(job):
            # A newer spec landed while this one was building. Drop it; the
            # work is wasted but nothing has been mutated.
            self._out.put(Failed(job, "superseded"))
            return

        self.prepared_count += 1
        self._out.put(Prepared(job, task, time.perf_counter() - started))

    def _is_stale(self, job: Job) -> bool:
        with self._cv:
            return job.gen != self._gen
