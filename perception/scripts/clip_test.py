"""Run the real pipeline against a real clip and report what it saw.

    python scripts/clip_test.py scenarios/follow_red.json
    python scripts/clip_test.py scenarios/*.json --record out/

No server, no orchestrator, no network: the same loop, builder, detector and
attribute engine the service runs, driven against a recorded file. That makes
a run reproducible, which is the only way to tell "the model missed it" apart
from "the pipeline dropped it" when something fails on stage.

A scenario is a clip, a model, a timeline of behaviours to start and stop, and
what should have happened by the end. It exits non-zero when reality disagrees,
so a set of them is a pre-demo checklist rather than something to read.

With --record it also writes the annotated video, which is both evidence and
the backup footage for a demo that will not cooperate live.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CFG, setup_logging  # noqa: E402
from contracts import Behavior  # noqa: E402
from detectors.registry import Registry  # noqa: E402
from runtime.builder import Builder  # noqa: E402
from runtime.capture import Capture  # noqa: E402
from runtime.events import make_buses  # noqa: E402
from runtime.loop import InferenceLoop  # noqa: E402
from runtime.state import Machine  # noqa: E402
from runtime.world import Shared  # noqa: E402
from server.stream import Streamer  # noqa: E402
from scripts.dump import Dumper  # noqa: E402
from stages.encoders import ClipEncoder  # noqa: E402

PASS, FAIL = "  PASS", "  FAIL"


class Run:
    """One scenario, executed."""

    def __init__(self, scenario: dict, record: Path | None = None,
                 dump: Path | None = None):
        self.s = scenario
        self.name = scenario.get("name", "unnamed")
        self.record = record
        self.dumper = Dumper(dump, self.name) if dump else None
        self.events, self.fired = [], []
        self.seen: dict[str, dict] = {}   # behaviour id -> what it ever reached
        self.applied_at: dict[str, float] = {}
        self.asked_at: dict[str, float] = {}
        self.frames = 0

    # 1. Setup ---------------------------------------------------------
    def build(self) -> None:
        registry = Registry.from_yaml()
        failed = registry.preload()
        if failed:
            print(f"  (models that would not load: {failed})")

        encoder = None
        if self.needs_attributes:
            encoder = ClipEncoder()
            try:
                encoder.load()
            except Exception as exc:
                print(f"  (CLIP unavailable, attribute checks will not match: {exc})")
                encoder = None

        ev, st = make_buses()
        ev.subscribe_callback(self._on_event)
        self.machine = Machine(emit=ev.publish)
        self.machine.on_registry_ready()

        clip = self.s["clip"]
        if not Path(clip).exists():
            raise FileNotFoundError(f"clip not found: {clip}")
        self.capture = Capture(clip)
        self.builder = Builder(registry, default_model=self.s.get("model", "yoloe"),
                               encoder=encoder).start()
        shared = Shared()
        shared.bank.encoder = encoder
        self.loop = InferenceLoop(self.capture, self.builder, self.machine, st, ev, shared)
        self.streamer = Streamer(self.loop) if (self.record or self.dumper) else None

    @property
    def needs_attributes(self) -> bool:
        return any(step.get("add", {}).get("subject", {}).get("include")
                   or step.get("add", {}).get("subject", {}).get("exclude")
                   for step in self.s.get("steps", []))

    def _on_event(self, ev) -> None:
        self.events.append(ev)
        if hasattr(ev, "kind"):
            self.fired.append(ev)

    # 2. Execution ------------------------------------------------------
    def play(self) -> None:
        """Run the clip once, applying each step at its scheduled time."""
        duration = float(self.s.get("duration_s", 15.0))
        steps = sorted(self.s.get("steps", []), key=lambda s: s.get("at", 0.0))
        writer = None

        self.capture.start()
        if not self.capture.wait_for_frame(10.0):
            raise RuntimeError("no frame from the clip in 10s")

        start = time.time()
        try:
            while True:
                elapsed = time.time() - start
                if elapsed > duration:
                    break
                while steps and steps[0].get("at", 0.0) <= elapsed:
                    self._apply(steps.pop(0), elapsed)

                if self.loop.step() is not None:
                    self.frames += 1
                    self._observe()
                    writer = self._maybe_write(writer)
                    if self.dumper:
                        self.dumper.maybe(self.loop, self.streamer.jpeg())
                time.sleep(0.001)
        finally:
            if writer:
                writer.release()
            self.builder.stop()
            self.capture.stop()

    def _apply(self, step: dict, elapsed: float) -> None:
        if "add" in step:
            b = Behavior(**step["add"])
            self.asked_at[b.behavior_id] = time.time()
            self.builder.add_behavior(f"clip-{b.behavior_id}", b)
            print(f"  [{elapsed:5.1f}s] start {b.behavior_id} "
                  f"({b.kind}: {'+'.join(b.subject.detect)}"
                  f"{' include=' + str(b.subject.include) if b.subject.include else ''}"
                  f"{' exclude=' + str(b.subject.exclude) if b.subject.exclude else ''})")
        elif "remove" in step:
            self.builder.remove_behavior("clip", step["remove"])
            print(f"  [{elapsed:5.1f}s] stop  {step['remove']}")
        elif "model" in step:
            self.builder.set_model("clip", step["model"])
            print(f"  [{elapsed:5.1f}s] model -> {step['model']}")

    def _observe(self) -> None:
        """Record the best each behaviour ever managed, not just the last frame."""
        summary = self.loop.last_summary
        if not summary:
            return
        for b in summary.behaviors:
            row = self.seen.setdefault(b.behavior_id, {
                "kind": b.kind, "states": set(), "max_matches": 0, "frames": 0})
            row["states"].add(b.state)
            row["max_matches"] = max(row["max_matches"], b.matches)
            row["frames"] += 1
            if b.state == "active" and b.behavior_id not in self.applied_at:
                self.applied_at[b.behavior_id] = time.time()

    def _maybe_write(self, writer):
        if not self.streamer or not self.record:
            return writer
        data = self.streamer.jpeg()
        if not data:
            return writer
        import numpy as np

        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if writer is None:
            self.record.mkdir(parents=True, exist_ok=True)
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.name)
            path = self.record / f"{safe}.mp4"
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                     CFG.STREAM_FPS, (frame.shape[1], frame.shape[0]))
            print(f"  recording to {path}")
        writer.write(frame)
        return writer

    # 3. Verdict --------------------------------------------------------
    def report(self) -> int:
        print(f"\n  frames={self.frames} fps={self.loop.timings.fps:.1f} "
              f"timings={ {k: round(v, 1) for k, v in self.loop.timings.ewma.items()} }")
        for bid, row in self.seen.items():
            latency = (f"{1000 * (self.applied_at[bid] - self.asked_at[bid]):.0f} ms"
                       if bid in self.applied_at and bid in self.asked_at else "never active")
            print(f"  {bid:12} {row['kind']:10} states={sorted(row['states'])} "
                  f"max_matches={row['max_matches']} acquired_in={latency}")
        if self.fired:
            kinds = {}
            for e in self.fired:
                kinds[e.kind] = kinds.get(e.kind, 0) + 1
            print(f"  events: {kinds}")

        if self.dumper:
            where = self.dumper.finish({
                "frames": self.frames,
                "fps": round(self.loop.timings.fps, 1),
                "timings_ms": {k: round(v, 1) for k, v in self.loop.timings.ewma.items()},
                "behaviors": {b: {"states": sorted(r["states"]),
                                  "max_matches": r["max_matches"]}
                              for b, r in self.seen.items()},
                "events": [e.kind for e in self.fired],
            })
            print(f"  dumped for review: {where}/index.html")

        expect = self.s.get("expect", {})
        if not expect:
            print("\n  (no expectations declared, nothing to check)")
            return 0

        print()
        failures = 0
        for bid, want in expect.items():
            row = self.seen.get(bid)
            for check, result, detail in self._checks(bid, row, want):
                print(f"{PASS if result else FAIL}  {bid}: {check}{detail}")
                failures += not result
        return failures

    def _checks(self, bid, row, want):
        if row is None:
            yield "behaviour ever existed", False, " (never appeared)"
            return
        if "reaches" in want:
            got = want["reaches"] in row["states"]
            yield (f"reaches {want['reaches']!r}", got,
                   f" (saw {sorted(row['states'])})")
        if "min_matches" in want:
            got = row["max_matches"] >= want["min_matches"]
            yield (f"at least {want['min_matches']} match(es)", got,
                   f" (peaked at {row['max_matches']})")
        if "max_matches" in want:
            got = row["max_matches"] <= want["max_matches"]
            yield (f"at most {want['max_matches']} match(es)", got,
                   f" (peaked at {row['max_matches']})")
        if "acquire_under_ms" in want:
            have = bid in self.applied_at and bid in self.asked_at
            ms = 1000 * (self.applied_at[bid] - self.asked_at[bid]) if have else None
            got = have and ms <= want["acquire_under_ms"]
            yield (f"acquired within {want['acquire_under_ms']} ms", got,
                   f" ({ms:.0f} ms)" if have else " (never acquired)")
        if "events" in want:
            fired = {e.kind for e in self.fired if e.behavior_id == bid}
            for kind in want["events"]:
                yield f"fired {kind!r}", kind in fired, f" (fired {sorted(fired)})"


def parse(argv: list[str]) -> tuple[list[str], Path | None, Path | None]:
    """Scenario paths plus optional output directories. A flag consumes its
    value, so `--record out/` does not leave `out/` looking like a scenario."""
    paths, record, dump, i = [], None, None, 0
    while i < len(argv):
        if argv[i] == "--record":
            record = Path(argv[i + 1])
            i += 2
        elif argv[i] == "--dump":
            dump = Path(argv[i + 1])
            i += 2
        elif argv[i].startswith("--"):
            i += 1
        else:
            paths.append(argv[i])
            i += 1
    return paths, record, dump


def main() -> int:
    setup_logging()
    args, record, dump = parse(sys.argv[1:])
    if not args:
        print(__doc__)
        return 2

    total = 0
    for path in args:
        scenario = json.loads(Path(path).read_text())
        print(f"\n{'=' * 70}\n{scenario.get('name', path)}\n  clip: {scenario['clip']}"
              f"  model: {scenario.get('model', 'yoloe')}\n{'=' * 70}")
        run = Run(scenario, record, dump)
        try:
            run.build()
            run.play()
            total += run.report()
        except Exception as exc:
            print(f"{FAIL}  scenario blew up: {type(exc).__name__}: {exc}")
            total += 1

    print(f"\n{'=' * 70}")
    print("ALL SCENARIOS PASSED" if total == 0 else f"{total} CHECK(S) FAILED")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
