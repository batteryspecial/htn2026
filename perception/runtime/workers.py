"""The background worker that builds a new world.

Everything expensive about a change happens here: loading a model, encoding
prompt vocabularies, encoding CLIP phrases. The frame loop never waits for any
of it and never sees a half-built world.

Two rules make hot-swapping safe:

1. Nothing is handed over until it is complete, so a failure is a no-op. The
   behaviours already running keep running.
2. The loop is the only thread that installs anything. This worker posts
   outcomes to a queue the loop drains at the top of a frame.

Ids are assigned here rather than by the caller, so two agents cannot collide
on a name.
"""

from __future__ import annotations

import itertools
import logging
import queue
import threading
import time
from dataclasses import replace

from behaviors.base import Behavior
from behaviors.kinds import build as build_behavior
from contracts import BehaviorSpec
from zoo.registry import Registry
from runtime.ops import (
    Accepted,
    Applied,
    ModelLoaded,
    ModelLoading,
    Op,
    Outcome,
    Rejected,
)
from attributes.references import ReferenceStore
from runtime.world import World

log = logging.getLogger("perception.workers")


class Builder:
    def __init__(self, registry: Registry, default_model: str = "yoloe",
                 encoder=None) -> None:
        self.registry = registry
        self.default_model = default_model
        #: CLIP, or None to run with attributes disabled. Text encoding happens
        #: on this thread so the loop never pays for it.
        self.encoder = encoder
        self.references_encoder_set = False
        #: The worker's own copy. The loop has the installed one; this is what
        #: the next operation is applied to.
        self.world: World | None = None
        #: Registered reference images. Written here, read by the API for
        #: thumbnails and by the world for matching.
        self.references = ReferenceStore()
        self._q: queue.Queue[Op] = queue.Queue()
        self._out: queue.Queue[Outcome] = queue.Queue()
        self._seq = itertools.count(1)
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._reference_done = threading.Condition()
        self._reference_results: dict[str, Applied | Rejected] = {}
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._inflight = 0
        self.applied_count = 0

    # 1. Submitting ------------------------------------------------------
    def _submit(self, op: Op) -> Op:
        with self._lock:
            op = replace(op, seq=next(self._seq))
            self._inflight += 1
            self._publish(Accepted(op))
            self._q.put(op)
        return op

    def add_behavior(self, spec: BehaviorSpec) -> str:
        """Returns the assigned id immediately; the behaviour starts later."""
        with self._lock:
            behavior_id = f"b{next(self._ids)}"
        self._submit(Op("add_behavior", behavior_id=behavior_id, spec=spec))
        return behavior_id

    def remove_behavior(self, behavior_id: str) -> Op:
        return self._submit(Op("remove_behavior", behavior_id=behavior_id))

    def clear(self) -> Op:
        return self._submit(Op("clear"))

    def set_model(self, model: str) -> Op:
        return self._submit(Op("set_model", model=model))

    def add_reference(self, image=None, vector=None, label: str | None = None) -> str:
        """Queue a reference; the API waits on ``wait_reference`` for encoding."""
        import uuid

        ref_id = f"r{uuid.uuid4().hex[:8]}"
        self._submit(Op("add_reference", ref_id=ref_id, image=image,
                        vector=vector, label=label))
        return ref_id

    def wait_reference(self, ref_id: str,
                       timeout: float = 10.0) -> Applied | Rejected | None:
        """Wait until a submitted reference was encoded or rejected.

        Only the reference HTTP route uses this acknowledgement. Behaviour
        and model changes remain asynchronous so the frame loop never waits.
        """
        deadline = time.monotonic() + timeout
        with self._reference_done:
            while ref_id not in self._reference_results:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._reference_done.wait(remaining)
            return self._reference_results.pop(ref_id)

    def _publish(self, outcome: Outcome) -> None:
        self._out.put(outcome)
        if (isinstance(outcome, (Applied, Rejected))
                and outcome.op.kind == "add_reference"
                and outcome.op.ref_id):
            with self._reference_done:
                self._reference_results[outcome.op.ref_id] = outcome
                if len(self._reference_results) > 200:
                    self._reference_results.pop(next(iter(self._reference_results)))
                self._reference_done.notify_all()

    @property
    def idle(self) -> bool:
        """Nothing queued and nothing being built.

        Distinct from "the queue is empty": an op that has been picked up but
        not finished leaves the queue empty while the world is still the old
        one, and anything waiting on the queue alone would run ahead of it.
        """
        with self._lock:
            return self._inflight == 0 and self._q.empty()

    # 2. Draining --------------------------------------------------------
    def poll(self) -> list[Outcome]:
        """Called from the frame loop. Never blocks."""
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
        self._q.put(Op("clear", seq=-1))
        if self._thread:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            op = self._q.get()
            if op.seq == -1:
                return
            try:
                self._process(op)
            except Exception as exc:  # a dead worker takes the demo with it
                log.exception("builder failed on %s", op.describe())
                self._publish(Rejected(op, f"{type(exc).__name__}: {exc}"))
            finally:
                with self._lock:
                    self._inflight -= 1

    def _process(self, op: Op) -> None:
        started = time.perf_counter()
        current = self.world
        behaviors: dict[str, Behavior] = dict(current.behaviors) if current else {}
        model = current.model_name if current else self.default_model
        reset = False

        if op.kind == "add_behavior":
            try:
                behaviors[op.behavior_id] = build_behavior(
                    op.behavior_id, op.spec, index=len(behaviors),
                    taken={b.color for b in behaviors.values()})
            except Exception as exc:
                self._publish(Rejected(op, f"{type(exc).__name__}: {exc}"))
                return
        elif op.kind == "remove_behavior":
            if op.behavior_id not in behaviors:
                self._publish(Rejected(op, f"no behaviour {op.behavior_id!r}"))
                return
            behaviors.pop(op.behavior_id)
        elif op.kind == "clear":
            behaviors = {}
        elif op.kind == "set_model":
            if op.model == model:
                self._publish(Rejected(op, f"already using {op.model!r}"))
                return
            model, reset = op.model, True
        elif op.kind == "add_reference":
            try:
                self._register(op)
            except Exception as exc:
                self._publish(Rejected(op, f"{type(exc).__name__}: {exc}"))
                return

        try:
            world = self._assemble(op, model, behaviors)
        except Exception as exc:
            log.warning("%s failed: %s", op.describe(), exc)
            if op.kind == "add_reference" and op.ref_id:
                self.references.items.pop(op.ref_id, None)
            self._publish(Rejected(op, f"{type(exc).__name__}: {exc}"))
            return

        self.world = world
        self.applied_count += 1
        self._publish(Applied(
            op, world, time.perf_counter() - started, reset_tracking=reset))

    def _register(self, op: Op) -> None:
        """Encode an uploaded photo, or store an embedding taken from a frame."""
        self.references.encoder = self.encoder
        if op.vector is not None:
            ref = self.references.add_vector(op.vector, op.label)
        else:
            if self.encoder is None:
                raise RuntimeError("references need CLIP loaded")
            ref = self.references.add_image(op.image, op.label)
        # The id was handed to the caller already, so keep it.
        self.references.items.pop(ref.ref_id, None)
        ref.ref_id = op.ref_id
        self.references.items[op.ref_id] = ref

    def _assemble(self, op: Op, model: str, behaviors: dict[str, Behavior]) -> World:
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
        # A fixed-vocabulary model is asked only for what it knows; behaviours
        # wanting the rest are paused by revalidate() rather than refused.
        if detector.classes is not None:
            prompts = [p for p in prompts if p in set(detector.classes)]
        prepared = detector.prepare(prompts) if prompts else None

        texts, baselines = self._encode(active)
        world = World(model_name=model, detector=detector, prepared=prepared,
                      behaviors=behaviors, text_vectors=texts,
                      baseline_vectors=baselines,
                      ref_vectors=self.references.vectors())
        world.revalidate(self.registry)
        return world

    def _encode(self, behaviors) -> tuple[dict, dict]:
        """Encode every CLIP phrase once, here, off the loop.

        Text encoding is the slow half of an attribute check, so doing it on
        the worker is why a behaviour with attributes still swaps in one frame.
        """
        phrases = World.attribute_texts(behaviors)
        if self.encoder is None or not phrases:
            return {}, {}
        classes = World.prompt_union(behaviors)
        vecs = self.encoder.encode_text(phrases)
        base = self.encoder.encode_text([f"a {c}" for c in classes])
        return ({p: vecs[i] for i, p in enumerate(phrases)},
                {c: base[i] for i, c in enumerate(classes)})
