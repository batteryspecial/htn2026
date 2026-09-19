"""Entry point. Wires the threads together and hands them to the server.

    cd perception
    uvicorn main:app --host 0.0.0.0 --port 8001

Everything configurable comes from the environment; see config.py. Bind to
0.0.0.0 and read service URLs from env vars, because on the day the car, the
orchestrator and this process are rarely on the same machine.
"""

from __future__ import annotations

import logging

from config import CFG, resolve_device, setup_logging
from detectors.registry import Registry
from runtime.builder import Builder
from runtime.capture import Capture
from runtime.events import make_buses
from runtime.loop import InferenceLoop
from runtime.state import Machine
from runtime.world import Shared
from server.api import Service, create_app
from stages.encoders import ClipEncoder

log = logging.getLogger("perception.main")


def build() -> Service:
    """Construct everything. Model loading happens here; threads start later."""
    setup_logging()
    log.info("device=%s imgsz=%s source=%s", resolve_device(), CFG.IMGSZ, CFG.VIDEO_SOURCE)

    registry = Registry.from_yaml()
    failed = registry.preload()
    if failed:
        log.warning("models failed to preload: %s", failed)

    events, states = make_buses()
    machine = Machine(emit=events.publish)

    # CLIP is optional: without it, behaviours with no attributes still work
    # and ones that need attributes simply never match. Better than refusing
    # to start.
    encoder = ClipEncoder()
    try:
        encoder.load()
    except Exception as exc:
        log.warning("CLIP unavailable, attributes disabled: %s", exc)
        encoder = None

    # The service comes up even with no usable detector: it reports FAULT
    # rather than refusing to start, so the operator can see what is wrong.
    if any(e["loaded"] for e in registry.manifest()):
        machine.on_registry_ready()
    else:
        machine.on_boot_failed(f"no model loaded; failed: {failed}")

    capture = Capture()
    builder = Builder(registry, default_model=_default_model(registry), encoder=encoder)
    shared = Shared()
    shared.bank.encoder = encoder
    loop = InferenceLoop(capture, builder, machine, states, events, shared)
    return Service(registry=registry, capture=capture, builder=builder, loop=loop,
                   machine=machine, events=events, states=states)


def _default_model(registry: Registry) -> str:
    """Prefer an open-vocabulary model: the agent invents class names."""
    loaded = [m for m in registry.manifest() if m["loaded"]]
    for m in loaded:
        if m["open_vocab"]:
            return m["name"]
    return loaded[0]["name"] if loaded else "yoloe"


app = create_app(build())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=CFG.HOST, port=CFG.PORT, log_level=CFG.LOG_LEVEL.lower())
