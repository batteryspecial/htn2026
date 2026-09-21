"""Settings, read once from the environment.

Same shape as `perception/config.py` on purpose: one frozen object, every
value overridable by an env var, no settings scattered through the code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent

load_dotenv(ROOT / ".env")


def _s(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


def _f(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _i(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Config:
    # 1. Model -----------------------------------------------------------
    provider: str = _s("ORCH_PROVIDER", "openai")
    model: str = _s("ORCH_MODEL", "gpt-6-astra")
    #: Must accept images. Blank reuses `model`.
    vision_model: str = _s("ORCH_VISION_MODEL", "") or _s("ORCH_MODEL", "gpt-6-astra")
    #: Blank disables vector search; the memory falls back to full-text.
    embed_model: str = _s("ORCH_EMBED_MODEL", "text-embedding-3-small")

    # 2. Services --------------------------------------------------------
    perception_base: str = _s("PERCEPTION_BASE", "http://localhost:8001").rstrip("/")
    port: int = _i("PORT", 8000)
    log_level: str = _s("LOG_LEVEL", "INFO")

    # 3. Agent loop ------------------------------------------------------
    #: Hard cap on model steps in one turn; each may request several tools.
    max_steps: int = _i("ORCH_MAX_STEPS", 4)
    #: Wake the agent at most once per behaviour per this many seconds. A
    #: person standing near the duck would otherwise fire thirty times a
    #: second, and one alert that arrives beats a hundred that get dropped.
    event_cooldown_s: float = _f("ORCH_EVENT_COOLDOWN_S", 5.0)
    #: A tool call that takes longer than this is a demo that has stalled.
    tool_timeout_s: float = _f("ORCH_TOOL_TIMEOUT_S", 12.0)
    #: Frontier models stall occasionally; the frontend learned this the hard way.
    llm_timeout_s: float = _f("ORCH_LLM_TIMEOUT_S", 20.0)
    turn_timeout_s: float = _f("ORCH_TURN_TIMEOUT_S", 35.0)

    # 4. Paths -----------------------------------------------------------
    documents: Path = ROOT / "documents"
    memory_dir: Path = Path(_s("ORCH_MEMORY_DIR", str(ROOT / ".lancedb")))
    #: Built frontend, served at / when it exists. Vite's dev server is the
    #: normal path; this is for a single-process demo machine.
    static_dir: Path = REPO_ROOT / "frontend" / "dist"

    @property
    def api_key(self) -> str:
        env = "ANTHROPIC_API_KEY" if self.provider == "anthropic" else "OPENAI_API_KEY"
        return os.environ.get(env, "")


CFG = Config()
