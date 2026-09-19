"""The background worker that builds a new world.

Everything expensive about a change happens here: loading a model, encoding
prompt vocabularies, encoding CLIP phrases. The inference loop never waits for
any of it, and never sees a half-built world.

The safety rules are unchanged from the single-spec version and are the reason
retasking live is safe:

1. Nothing is handed over until it is complete, so a failure is a no-op. The
   behaviours already running keep running.
2. The loop is the only thread that installs anything. This worker posts
   outcomes to a queue the loop drains at the top of a frame.

What is new is that operations accumulate rather than replace: adding a
behaviour keeps the others, so each operation is applied to the world as it
stood when the operation was picked up.
"""

from __future__ import annotations

import logging
import queue
import threading
import time

from behaviors.base import RuntimeBehavior
from behaviors.kinds import build as build_behavior
from detectors.registry import Registry
from runtime.ops import (
    Accepted,
    Applied,
    ModelLoaded,
    ModelLoading,
    Op,
    Outcome,
    Rejected,
)
from runtime.world import World

log = logging.getLogger("perception.builder")


class Builder:
    def __init__(self, registry: Registry, default_model: str = "yoloe",
                 encoder=None) -> None:
        self.registry = registry
        self.default_model = default_model
        #: CLIP, or None to run with attributes disabled. Text encoding happens
        #: on this thread so the loop never pays for it.
        self.encoder = encoder
        #: The worker's own copy of the world. The loop has the installed one;
        #: this is what the next operation is applied to.
        self.world: World | None = None
        self._q: queue.Queue[Op] = queue.Queue()
        self._out: queue.Queue[Outcome] = queue.Queue()
        self._seq = 0
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.applied_count = 0

    # 1. Submitting ------------------------------------------------------
    def submit(self, op: Op) -> Op:
        """Called from the API thread. Returns immediately."""
        with self._lock:
            self._seq += 1
            op = Op(kind=op.kind, instruction_id=op.instruction_id, seq=self._seq,
                    behavior=op.behavior, behavior_id=op.behavior_id, model=op.model)
            self._out.put(Accepted(op))
            self._q.put(op)
        return op

    def add_behavior(self, instruction_id: str, behavior) -> Op:
        return self.submit(Op("add_behavior", instruction_id, behavior=behavior))

    def remove_behavior(self, instruction_id: str, behavior_id: str) -> Op:
        return self.submit(Op("remove_behavior", instruction_id, behavior_id=behavior_id))

    def clear(self, instruction_id: str = "clear") -> Op:
        return self.submit(Op("clear", instruction_id))

    def set_model(self, instruction_id: str, model: str) -> Op:
        return self.submit(Op("set_model", instruction_id, model=model))

    # 2. Draining --------------------------------------------------------
    def poll(self) -> list[Outcome]:
        """Called from the inference loop. Never blocks."""
        out = []
        while True:
            try:
                out.append(self._out.get_nowait())
            except queue.Empty:
                return out

    # 3. Worker ----------------------------------------------------------
    def start(self) -> Builder:
        self._thread = threading.Thread(target=self._run, name="builder", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._q.put(Op("clear", "__stop__"))
        if self._thread:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            op = self._q.get()
            if op.instruction_id == "__stop__":
                return
            try:
                self._process(op)
            except Exception as exc:  # a worker that dies takes the demo with it
                log.exception("builder failed on %s", op.describe())
                self._out.put(Rejected(op, f"{type(exc).__name__}: {exc}"))

    def _process(self, op: Op) -> None:
        started = time.perf_counter()
        current = self.world
        behaviors: dict[str, RuntimeBehavior] = dict(current.behaviors) if current else {}
        model = current.model_name if current else self.default_model
        reset = False

        # 3a. Work out what the new world should contain.
        if op.kind == "add_behavior":
            try:
                behaviors[op.behavior.behavior_id] = build_behavior(op.behavior)
            except Exception as exc:
                self._out.put(Rejected(op, f"{type(exc).__name__}: {exc}"))
                return
        elif op.kind == "remove_behavior":
            if op.behavior_id not in behaviors:
                self._out.put(Rejected(op, f"no behaviour {op.behavior_id!r}"))
                return
            behaviors.pop(op.behavior_id)
        elif op.kind == "clear":
            behaviors = {}
        elif op.kind == "set_model":
            if op.model == model:
                self._out.put(Rejected(op, f"already using {op.model!r}"))
                return
            model, reset = op.model, True

        # 3b. Build it. Anything that raises leaves the running world alone.
        try:
            world = self._assemble(op, model, behaviors)
        except Exception as exc:
            log.warning("%s failed: %s", op.describe(), exc)
            self._out.put(Rejected(op, f"{type(exc).__name__}: {exc}"))
            return

        self.world = world
        self.applied_count += 1
        self._out.put(Applied(op, world, time.perf_counter() - started, reset_tracking=reset))

    def _assemble(self, op: Op, model: str, behaviors: dict[str, RuntimeBehavior]) -> World:
        """Load, prepare and encode everything the new world needs."""
        if not self.registry.is_loaded(model):
            # Cold path. Everything available is preloaded at boot, so this is
            # a fallback rather than something that happens during a demo.
            self._out.put(ModelLoading(op, model))
            t0 = time.perf_counter()
            detector = self.registry.get(model)
            self._out.put(ModelLoaded(op, model, time.perf_counter() - t0))
        else:
            detector = self.registry.get(model)

        active = list(behaviors.values())
        prompts = World.prompt_union(active)
        prepared = detector.prepare(prompts) if prompts else None

        texts, baselines = self._encode(active)
        return World(model_name=model, detector=detector, prepared=prepared,
                     behaviors=behaviors, text_vectors=texts, baseline_vectors=baselines)

    def _encode(self, behaviors) -> tuple[dict, dict]:
        """Encode every CLIP phrase once, here, off the loop.

        Text encoding is the slow half of an attribute check; doing it on the
        worker is why a behaviour with attributes still swaps in one frame.
        """
        encoder = self.encoder
        phrases = World.attribute_texts(behaviors)
        if encoder is None or not phrases:
            return {}, {}
        classes = World.prompt_union(behaviors)
        vecs = encoder.encode_text(phrases)
        base = encoder.encode_text([f"a {c}" for c in classes])
        return (
            {p: vecs[i] for i, p in enumerate(phrases)},
            {c: base[i] for i, c in enumerate(classes)},
        )
