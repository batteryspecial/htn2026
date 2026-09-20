"""Fixtures, and the environment every test runs in.

The environment is set *before* `config` is imported, because `CFG` is frozen
at import time. Three of these matter:

* `OPENAI_API_KEY` is blanked. A developer's real key sitting in `.env` would
  otherwise turn a unit test into a billed network call — the phrase memory
  embeds on write, and `load_dotenv` would happily supply it.
* `ORCH_EMBED_MODEL` is blanked, so the memory runs full-text only. That is
  the offline mode, and it is the one CI can actually run.
* `PERCEPTION_BASE` points at a host that does not exist, so anything that
  escapes the mock transport fails loudly instead of hitting a real pipeline
  someone has running on :8001.
"""

from __future__ import annotations

import os
import tempfile

_TMP = tempfile.mkdtemp(prefix="orchestrator-tests-")

os.environ["OPENAI_API_KEY"] = ""
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["ORCH_EMBED_MODEL"] = ""
os.environ["ORCH_MEMORY_DIR"] = os.path.join(_TMP, "lancedb")
os.environ["PERCEPTION_BASE"] = "http://perception.test"
os.environ["ORCH_MAX_STEPS"] = "4"
os.environ["ORCH_EVENT_COOLDOWN_S"] = "5"
os.environ["LOG_LEVEL"] = "WARNING"

import pytest  # noqa: E402

from app.trace import BUS, TraceEntry  # noqa: E402
from memory.store import PhraseMemory  # noqa: E402
from tests.fake_pipeline import FakePipeline  # noqa: E402
from tools.perception import Perception  # noqa: E402


@pytest.fixture
def pipeline() -> FakePipeline:
    """Perception, in process."""
    return FakePipeline()


@pytest.fixture
async def perception(pipeline: FakePipeline):
    """A real `Perception` client whose socket is the fake pipeline.

    Everything under test is the production path: URL building, status
    handling, the `{"ok": ...}` contract. Only the wire is fake.
    """
    client = Perception(transport=pipeline.transport())
    try:
        yield client
    finally:
        await client.aclose()


@pytest.fixture
def memory(tmp_path) -> PhraseMemory:
    """A phrase memory of its own per test, seeded and full-text only."""
    return PhraseMemory(uri=str(tmp_path / "lancedb"), embed_model="")


@pytest.fixture(autouse=True)
def clean_bus():
    """The trace bus is a module global; a leftover backlog breaks replay."""
    BUS._recent.clear()
    BUS._subscribers.clear()
    yield
    BUS._recent.clear()
    BUS._subscribers.clear()


@pytest.fixture
def trace() -> list[TraceEntry]:
    """Everything published while a test runs, in order."""
    entries: list[TraceEntry] = []
    original = BUS.publish

    def capture(entry: TraceEntry) -> None:
        entries.append(entry)
        original(entry)

    BUS.publish = capture  # type: ignore[method-assign]
    try:
        yield entries
    finally:
        BUS.publish = original  # type: ignore[method-assign]


def kinds_of(entries: list[TraceEntry]) -> list[str]:
    return [e.kind for e in entries]


def labels_of(entries: list[TraceEntry], kind: str | None = None) -> list[str]:
    return [e.label for e in entries if kind is None or e.kind == kind]
