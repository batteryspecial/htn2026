"""Phrase memory: which wordings actually found things, and where.

This is the retrieval corpus that earns its place. The selector rules
themselves are ~5k tokens and live in the system prompt, where they always
apply and cannot be half-retrieved — Rules 4 and 5 are coupled, and fetching
one without the other produces worse specs than no retrieval at all.

What cannot fit in a prompt, and grows with use, is evidence: this phrase
scored 0.22 here, that one scored 0.00, and the bare noun never works. The
agent reads it as a **prior** for what to try first. `/probe` is what settles
it, because phrasing does not transfer between scenes (SELECTORS.md Rule 3) —
so a remembered phrase is a good first guess and nothing more.

Lookups are **hybrid**: vector for meaning, full-text for exact words, fused
with reciprocal rank. Vector search alone misfires on a corpus this short and
this repetitive — "a person wearing a dark jacket" and "a person wearing a
cream coat" are neighbours in embedding space while meaning opposite things,
and a query about a duck pulls back coats. Full-text alone cannot match
"duck" to "rubber duck toy".

With no embedding model configured it runs full-text only, which needs no
network — the right mode for an offline venue.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass
from typing import Any

from config import CFG

log = logging.getLogger("orchestrator.memory")

TABLE = "phrases"


@dataclass
class PhraseRecord:
    """One measurement of one wording."""

    #: The wording put into `detect`.
    phrase: str
    #: What the operator called it — the query side of the lookup.
    subject: str
    #: Highest confidence seen, 0 when nothing was found.
    max_conf: float = 0.0
    #: How many frames or probes found it.
    found: int = 0
    #: "probe" | "acquired" | "never_acquired" | "seed"
    outcome: str = "probe"
    #: Free text: which room, which detector, what the clip was.
    scene: str = ""
    note: str = ""
    ts: float = 0.0

    def text(self) -> str:
        """What gets embedded and indexed."""
        return f"{self.subject} :: {self.phrase} :: {self.note}".strip(" :")


class PhraseMemory:
    """LanceDB-backed, and degrades to in-process when LanceDB is missing.

    Losing the memory costs the agent a prior, not its ability to work — it
    still probes. So an unavailable store is a warning, never a failure.
    """

    def __init__(self, uri: str | None = None, embed_model: str | None = None) -> None:
        self.uri = uri or str(CFG.memory_dir)
        self.embed_model = embed_model if embed_model is not None else CFG.embed_model
        self._table = None
        self._fallback: list[PhraseRecord] = []
        self._embedder = None
        #: Whether a usable full-text index exists, and whether rows have been
        #: written since it was last built.
        self._fts = False
        self._dirty = False
        self._open()

    # 1. Setup -----------------------------------------------------------

    def _open(self) -> None:
        try:
            import lancedb
        except ImportError:
            log.warning("lancedb is not installed — phrase memory is in-process only")
            return

        try:
            db = lancedb.connect(self.uri)
            self._embedder = self._make_embedder()
            if TABLE in db.table_names():
                self._table = db.open_table(TABLE)
            else:
                self._table = db.create_table(TABLE, data=self._seed_rows(db))
                self._ensure_fts()
            if self._table is not None and not self._fts:
                self._ensure_fts()
            log.info("phrase memory at %s (%s)", self.uri, self.mode)
        except Exception as exc:                       # noqa: BLE001
            log.warning("phrase memory unavailable (%s) — falling back in-process", exc)
            self._table = None

    def _make_embedder(self):
        """None means full-text search, which needs no network."""
        if not self.embed_model or not CFG.api_key:
            return None
        try:
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model=self.embed_model, api_key=CFG.api_key)
        except Exception as exc:                        # noqa: BLE001
            log.warning("no embedder (%s) — using full-text search", exc)
            return None

    def _ensure_fts(self) -> None:
        """(Re)build the full-text index.

        Rows added after the index was built are not in it, so this runs again
        whenever the table has been written to. At this scale — tens to low
        hundreds of rows — rebuilding costs microseconds, and the alternative
        is a memory that silently stops returning its newest measurements.
        """
        if self._table is None:
            return
        try:
            self._table.create_fts_index("text", replace=True)
            self._fts = True
        except Exception as exc:                        # noqa: BLE001
            log.debug("no full-text index: %s", exc)
            self._fts = False
        finally:
            self._dirty = False

    @property
    def mode(self) -> str:
        """How lookups are answered. Shown in the log and by GET /memory."""
        if self._table is None:
            return "in-process"
        if self._embedder and self._fts:
            return "hybrid (vector + full-text, RRF)"
        if self._embedder:
            return "vector"
        return "full-text"

    def _seed_rows(self, _db: Any) -> list[dict[str, Any]]:
        from memory.seed import SEED

        rows = [self._row(record) for record in SEED]
        log.info("seeded phrase memory with %d measurements", len(rows))
        return rows

    def _row(self, record: PhraseRecord) -> dict[str, Any]:
        row = asdict(record)
        row["ts"] = record.ts or time.time()
        row["text"] = record.text()
        row["vector"] = self._embed(row["text"])
        return row

    def _embed(self, text: str) -> list[float]:
        if self._embedder is None:
            # LanceDB wants a consistent width; a zero vector keeps the schema
            # stable and search falls through to full-text.
            return [0.0] * 8
        try:
            return self._embedder.embed_query(text)
        except Exception as exc:                        # noqa: BLE001
            log.warning("embedding failed (%s) — storing without a vector", exc)
            return [0.0] * 8

    # 2. Writing ---------------------------------------------------------

    def remember(self, record: PhraseRecord) -> None:
        """Record one measurement. Called after every probe and every run."""
        record.ts = record.ts or time.time()
        if self._table is None:
            self._fallback.append(record)
            return
        try:
            self._table.add([self._row(record)])
            self._dirty = True
        except Exception as exc:                        # noqa: BLE001
            log.warning("could not write to phrase memory: %s", exc)
            self._fallback.append(record)

    def remember_probe(self, subject: str, results: list[dict[str, Any]],
                       scene: str = "") -> None:
        """Everything a `/probe` measured, including the wordings that failed.

        The zeros matter as much as the hits: "duck scored 0.00 here" is what
        stops the agent trying it again.
        """
        for result in results:
            self.remember(PhraseRecord(
                phrase=result.get("phrase", ""),
                subject=subject,
                max_conf=float(result.get("max_conf", 0.0)),
                found=int(result.get("found", 0)),
                outcome="probe",
                scene=scene,
            ))

    def remember_outcome(self, subject: str, phrases: list[str],
                         acquired: bool, note: str = "") -> None:
        """Whether the spec that used these phrasings actually worked."""
        for phrase in phrases:
            self.remember(PhraseRecord(
                phrase=phrase,
                subject=subject,
                outcome="acquired" if acquired else "never_acquired",
                note=note,
            ))

    # 3. Reading ---------------------------------------------------------

    def suggest(self, subject: str, limit: int = 6) -> list[PhraseRecord]:
        """Wordings worth trying for this subject.

        Relevance first, evidence second. The search returns rows nearest to
        the query; ranking those purely by confidence puts a well-measured
        entry about coats at the top of a question about ducks, which is
        worse than useless — the agent would try it.
        """
        rows = self._search(subject, limit * 4)
        if not rows:
            return []

        # Collapse to one entry per phrase, keeping the strongest evidence.
        # `rank` is the search's own ordering, which is the relevance signal.
        best: dict[str, PhraseRecord] = {}
        ranks: dict[str, int] = {}
        for position, row in enumerate(rows):
            record = PhraseRecord(
                phrase=row.get("phrase", ""),
                subject=row.get("subject", ""),
                max_conf=float(row.get("max_conf", 0.0)),
                found=int(row.get("found", 0)),
                outcome=row.get("outcome", ""),
                scene=row.get("scene", ""),
                note=row.get("note", ""),
                ts=float(row.get("ts", 0.0)),
            )
            if not record.phrase:
                continue
            seen = best.get(record.phrase)
            if seen is None or _weight(record) > _weight(seen):
                best[record.phrase] = record
                ranks[record.phrase] = min(ranks.get(record.phrase, position), position)

        total = max(len(rows), 1)

        def score(record: PhraseRecord) -> float:
            # Relevance dominates; evidence orders the entries that are
            # comparably relevant.
            relevance = 1.0 - ranks.get(record.phrase, total) / total
            return relevance * 2.0 + _weight(record)

        return sorted(best.values(), key=score, reverse=True)[:limit]

    def _search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Hybrid where possible: vector for meaning, full-text for exact words.

        Vector search alone misfires on this corpus. The entries are short and
        share vocabulary — "a person wearing a dark jacket" and "a person
        wearing a cream coat" are neighbours in embedding space while being
        opposite in meaning, and a query about a duck can pull back coats.
        Full-text alone has the opposite failure: it cannot match "duck" to
        "rubber duck toy" unless the words line up.

        Running both and fusing with reciprocal rank keeps an entry only when
        at least one retriever is confident, which is what this corpus needs.
        """
        if self._table is None:
            return [asdict(r) for r in self._fallback
                    if query.lower() in r.text().lower()][:limit]

        if self._dirty:
            self._ensure_fts()

        vector = self._embed(query) if self._embedder is not None else None
        has_vector = vector is not None and any(vector)

        try:
            if has_vector and self._fts:
                return (self._table.search(query_type="hybrid")
                        .vector(vector).text(query)
                        .limit(limit).to_list())
            if has_vector:
                return self._table.search(vector).limit(limit).to_list()
            if self._fts:
                return self._table.search(query, query_type="fts").limit(limit).to_list()
        except Exception as exc:                        # noqa: BLE001
            log.warning("phrase memory search failed (%s) — trying full-text", exc)
            try:
                return self._table.search(query, query_type="fts").limit(limit).to_list()
            except Exception as inner:                  # noqa: BLE001
                log.warning("full-text search failed too: %s", inner)
                return []

        return []

    def brief(self, subject: str, limit: int = 6) -> str:
        """The prior, as a few lines for the prompt. Empty when nothing is known."""
        records = self.suggest(subject, limit)
        if not records:
            return ""

        lines = []
        for r in records:
            if r.outcome == "never_acquired":
                verdict = "never acquired"
            elif r.outcome == "acquired":
                verdict = "acquired successfully"
            elif r.found:
                verdict = f"found, best score {r.max_conf:.2f}"
            else:
                verdict = "found nothing"
            where = f" [{r.scene}]" if r.scene else ""
            lines.append(f'- "{r.phrase}" — {verdict}{where}')

        return ("Measured before, for similar subjects. A prior, not a fact: "
                "phrasing does not transfer between scenes, so probe before "
                "committing.\n" + "\n".join(lines))


def _weight(record: PhraseRecord) -> float:
    """Rank evidence. Something that actually acquired beats a good probe."""
    score = record.max_conf
    if record.outcome == "acquired":
        score += 1.0
    elif record.outcome == "never_acquired":
        score -= 1.0
    elif record.outcome == "seed":
        score += 0.1
    return score
