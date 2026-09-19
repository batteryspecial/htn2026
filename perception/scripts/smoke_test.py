"""Run the whole pipeline against a generated clip, with no GPU and no weights.

    python scripts/smoke_test.py

Prints one line per phase change plus the retask latency, which is the number
the demo is judged on. Uses the `fake` detector, so this proves the plumbing,
not the perception.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CFG, setup_logging  # noqa: E402
from contracts import Phase, TaskSpec  # noqa: E402
from detectors.registry import Registry  # noqa: E402
from runtime.capture import Capture  # noqa: E402
from runtime.events import make_buses  # noqa: E402
from runtime.loader import Loader  # noqa: E402
from runtime.loop import InferenceLoop  # noqa: E402
from runtime.state import Machine  # noqa: E402


def make_clip(path: Path, n: int = 60) -> Path:
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
    for i in range(n):
        f = np.full((480, 640, 3), 30, np.uint8)
        cv2.circle(f, (100 + i * 8, 240), 40, (0, 140, 255), -1)
        w.write(f)
    w.release()
    return path


def spec(spec_id: str, what: str) -> TaskSpec:
    return TaskSpec(spec_id=spec_id, model="fake",
                    targets=[{"ref": "t1", "detect": [what]}])


def main() -> int:
    setup_logging()
    clip = make_clip(Path(CFG.WEIGHTS_DIR).parent / "clips" / "smoke.mp4")

    registry = Registry.from_yaml()
    registry.preload()
    events, targets = make_buses()
    seen: list = []
    events.subscribe_callback(lambda e: seen.append(e))

    machine = Machine(emit=events.publish)
    machine.on_registry_ready()
    capture = Capture(str(clip)).start()
    loader = Loader(registry).start()
    loop = InferenceLoop(capture, loader, machine, targets, events)

    capture.wait_for_frame(5.0)
    state = {"phase": None, "pending": None, "sent": 0.0}

    def pump(seconds: float) -> None:
        """Run the loop for a while, narrating phase changes."""
        deadline = time.time() + seconds
        while time.time() < deadline:
            loop.step()
            if machine.phase is not state["phase"]:
                print(f"  phase -> {machine.phase.value}")
                state["phase"] = machine.phase
                if machine.phase is Phase.TRACKING and state["pending"]:
                    print(f"  RETASK LATENCY {state['pending']}: "
                          f"{1000 * (time.time() - state['sent']):.0f} ms")
                    state["pending"] = None
            time.sleep(0.002)

    try:
        pump(0.5)
        for spec_id, what in (("s1", "pencil"), ("s2", "eraser")):
            print(f"\ninstruction: follow the {what}")
            state["pending"], state["sent"] = spec_id, time.time()
            loader.submit(f"i-{spec_id}", spec(spec_id, what))
            pump(2.0)
    finally:
        loader.stop()
        capture.stop()

    s = loop.last_state
    print(f"\nfinal: phase={s.phase.value} visible={s.visible} label={s.label} "
          f"cx={s.cx:+.2f} cy={s.cy:+.2f} area={s.area:.3f}")
    print(f"timings: {loop.timings.line()}")
    print(f"events: {[e.stage for e in seen]}")
    return 0 if s.phase is Phase.TRACKING else 1


if __name__ == "__main__":
    raise SystemExit(main())
