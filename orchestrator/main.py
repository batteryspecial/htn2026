"""Entry point. `python main.py` from inside orchestrator/."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Run from anywhere: the package imports itself by module name.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn  # noqa: E402

from app.api import create_app  # noqa: E402
from config import CFG  # noqa: E402

logging.basicConfig(
    level=CFG.log_level,
    format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

app = create_app()


def main() -> None:
    log = logging.getLogger("orchestrator")
    log.info("agent layer on :%d — model %s (%s), perception at %s",
             CFG.port, CFG.model, CFG.provider, CFG.perception_base)
    if not CFG.api_key:
        log.warning("no API key: copy .env.example to .env. The service will "
                    "boot and every turn will say so.")

    uvicorn.run(app, host="0.0.0.0", port=CFG.port, log_level=CFG.log_level.lower())


if __name__ == "__main__":
    main()
