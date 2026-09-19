"""CLIP attribute scoring: the difference between "a duck" and "the yellow one".

Three facts shape this stage.

1. **CLIP's raw similarities are too flat to threshold.** "a yellow duck" and
   "a duck" both score around 0.3 against the same crop, so an absolute cutoff
   picks up everything or nothing. Scoring is therefore contrastive: the phrase
   competes against a plain description of the class, and what matters is which
   one wins, not by how much.

2. **Per track, not per frame.** A track is the same object across hundreds of
   frames, so its colour does not need re-checking on each one. Results are
   cached against the tracker id and refreshed about once a second, which turns
   CLIP from a per-frame cost into a rounding error.

3. **Scored once, shared by everyone.** Several behaviours ask about the same
   track in the same frame. Every phrase any behaviour needs is encoded once,
   batched into a single forward pass.

The encoder is an interface so the whole stage is testable without weights.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import supervision as sv

from config import CFG

log = logging.getLogger("perception.attributes")


class Encoder(Protocol):
    """Whatever turns text and crops into comparable vectors."""

    dim: int

    def encode_text(self, texts: list[str]) -> np.ndarray: ...
    def encode_images(self, crops: list[np.ndarray]) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# 1. Scoring
# ---------------------------------------------------------------------------


def contrastive_scores(
    image_vecs: np.ndarray, phrase_vecs: np.ndarray, baseline_vecs: np.ndarray
) -> np.ndarray:
    """Probability each phrase beats its own plain-class baseline.

    For crop i and phrase j, softmax over just two logits: the phrase and
    `f"a {class}"`. A yellow duck scores near 1.0 for "a yellow duck" and a
    brown one near 0.0, even though their raw similarities barely differ.

    Returns an (n_crops, n_phrases) array in [0, 1].
    """
    if len(image_vecs) == 0 or len(phrase_vecs) == 0:
        return np.zeros((len(image_vecs), len(phrase_vecs)), dtype=np.float32)

    # 100 is CLIP's own trained logit scale; without it the softmax is mush.
    pos = image_vecs @ phrase_vecs.T * 100.0
    base = np.einsum("ij,ij->i", image_vecs, baseline_vecs)[:, None] * 100.0
    # Two-way softmax, written as a sigmoid of the difference. Clipped because
    # a confident score overflows exp long before it changes the answer.
    return (1.0 / (1.0 + np.exp(-np.clip(pos - base, -60.0, 60.0)))).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Similarity between every row of `a` and every row of `b`.

    Both sides are expected normalized; the encoder is responsible for that.
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    return (a @ b.T).astype(np.float32)


def crops_from(frame: np.ndarray, dets: sv.Detections, pad: float = 0.0,
               tighten: bool = False) -> list[np.ndarray]:
    """Cut each detection out of the frame, clipped to the frame bounds.

    With `tighten`, the crop is narrowed to the part of the box that actually
    carries a clothing attribute. This is not cosmetic: measured on real
    footage, a full-body crop of a man in a cream coat scored 0.95 for "a
    person wearing a dark jacket", because the crop was mostly dark jeans,
    dark shoes and a dark bus behind him. The jacket was a quarter of the
    pixels and CLIP scored what it was shown.
    """
    h, w = frame.shape[:2]
    out = []
    for x1, y1, x2, y2 in dets.xyxy:
        if pad:
            dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
            x1, y1, x2, y2 = x1 - dx, y1 - dy, x2 + dx, y2 + dy
        if tighten:
            x1, y1, x2, y2 = torso(x1, y1, x2, y2)
        xa, ya = max(0, int(x1)), max(0, int(y1))
        xb, yb = min(w, int(x2)), min(h, int(y2))
        # A degenerate box would crash the encoder; one pixel keeps shapes sane.
        out.append(frame[ya:max(yb, ya + 1), xa:max(xb, xa + 1)])
    return out


#: Boxes at least this much taller than wide are treated as upright bodies.
UPRIGHT_RATIO = 1.3
#: Fraction of an upright box, from the top, that holds head and torso.
TORSO_FRAC = 0.55
#: Horizontal trim, each side, to shed background inside the box.
SIDE_TRIM = 0.08


def torso(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
    """Narrow a box to the region an appearance word is actually about.

    Upright boxes lose their lower half, because "jacket", "shirt" and "hoodie"
    all live above the waist. Everything else keeps its height, since a duck
    has no torso. Both lose a sliver from each side, which is background the
    detector included to make the box rectangular.
    """
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return x1, y1, x2, y2
    trim = bw * SIDE_TRIM
    x1, x2 = x1 + trim, x2 - trim
    if bh / max(bw, 1e-6) >= UPRIGHT_RATIO:
        y2 = y1 + bh * TORSO_FRAC
    return x1, y1, x2, y2


# ---------------------------------------------------------------------------
# 2. The cache
# ---------------------------------------------------------------------------


@dataclass
class Scored:
    """What is known about one track's appearance."""

    scores: dict[str, float] = field(default_factory=dict)
    ref_sims: dict[str, float] = field(default_factory=dict)
    embedding: np.ndarray | None = None
    at: float = 0.0
    label: str = ""


class AttributeBank:
    """Scores tracks against phrases, caches per track, refreshes on a timer.

    Owned by the inference loop. Text encoding happens on the loader worker and
    arrives through `set_texts`; only image encoding runs in the loop, and only
    for tracks whose entry has expired.
    """

    def __init__(self, encoder: Encoder | None = None, ttl: float | None = None,
                 batch: int | None = None, tighten: bool | None = None) -> None:
        self.encoder = encoder
        self.ttl = CFG.VERIFY_TTL_S if ttl is None else ttl
        self.batch = CFG.VERIFY_BATCH if batch is None else batch
        self.tighten = CFG.ATTR_TIGHTEN if tighten is None else tighten
        self._text: dict[str, np.ndarray] = {}      # phrase -> vector
        self._baseline: dict[str, np.ndarray] = {}  # class name -> vector
        self._refs: dict[str, np.ndarray] = {}      # reference id -> vector
        self._cache: dict[int, Scored] = {}
        self.encodes = 0

    # 2a. Vocabulary, filled by the worker -----------------------------
    def set_texts(self, phrases: dict[str, np.ndarray], baselines: dict[str, np.ndarray]) -> None:
        self._text, self._baseline = phrases, baselines

    def set_references(self, refs: dict[str, np.ndarray]) -> None:
        self._refs = refs

    @property
    def phrases(self) -> list[str]:
        return list(self._text)

    # 2b. Per frame ----------------------------------------------------
    def update(self, frame: np.ndarray, dets: sv.Detections, now: float) -> None:
        """Refresh whatever has gone stale. Cheap on most frames.

        Only tracks whose entry expired are re-encoded, and never more than
        `batch` of them at once, so one crowded frame cannot stall the loop.
        """
        # Encode whenever there is an encoder, not only when a phrase was
        # asked for. The embedding is what re-identification compares, so
        # gating on attributes meant a follow-cam could never reacquire its
        # target. Scoring phrases on top of the embedding is free, and the
        # per-track cache keeps the forward pass rare.
        if self.encoder is None or len(dets) == 0:
            return
        ids = dets.tracker_id
        if ids is None:
            return

        stale = [i for i in range(len(dets)) if self._stale(int(ids[i]), now)][: self.batch]
        if not stale:
            return

        subset = dets[np.array(stale)]
        # Attribute scoring reads the tightened crop; re-identification reads
        # the same one, which is fine and arguably better: it compares what
        # someone is wearing rather than the floor behind them.
        crops = crops_from(frame, subset, tighten=self.tighten)
        try:
            vecs = self.encoder.encode_images(crops)
        except Exception as exc:
            # An attribute failure must never take the pipeline down: behaviours
            # that need attributes simply stop matching.
            log.warning("image encode failed: %s", exc)
            return
        self.encodes += 1

        names = subset.data.get("class_name")
        phrases = list(self._text)
        phrase_mat = np.stack([self._text[p] for p in phrases]) if phrases else np.zeros((0, 1))
        for row, i in enumerate(stale):
            label = str(names[row]) if names is not None else ""
            entry = Scored(embedding=vecs[row], at=now, label=label)
            if phrases:
                base = self._baseline.get(label)
                if base is None:
                    base = np.zeros_like(vecs[row])
                probs = contrastive_scores(vecs[row : row + 1], phrase_mat, base[None, :])[0]
                entry.scores = {p: float(v) for p, v in zip(phrases, probs)}
            if self._refs:
                ref_ids = list(self._refs)
                sims = cosine(vecs[row : row + 1], np.stack([self._refs[r] for r in ref_ids]))[0]
                entry.ref_sims = {r: float(v) for r, v in zip(ref_ids, sims)}
            self._cache[int(ids[i])] = entry

        self._evict(set(int(v) for v in ids))

    def _stale(self, track_id: int, now: float) -> bool:
        """A track never scored is always stale; one scored recently is not."""
        entry = self._cache.get(track_id)
        return entry is None or now - entry.at > self.ttl

    def _evict(self, live: set[int]) -> None:
        """Forget tracks that are long gone. A live stream never ends."""
        if len(self._cache) <= 256:
            return
        for tid in [t for t in self._cache if t not in live]:
            del self._cache[tid]

    # 2c. Reading ------------------------------------------------------
    def scores_for(self, track_id: int) -> dict[str, float]:
        entry = self._cache.get(int(track_id))
        return dict(entry.scores) if entry else {}

    def get(self, track_id: int) -> Scored | None:
        return self._cache.get(int(track_id))

    def matches(self, track_id: int, include, exclude, min_score: float) -> bool | None:
        """Does this track satisfy the selector's attribute rules?

        Returns None when the track has not been scored yet. The caller treats
        that as "not yet", never as "yes": a behaviour must not act on a track
        whose appearance is still unknown.

        Two regimes, because they behave very differently in practice.

        **Both include and exclude given:** decided by which phrase wins. This
        is by far the more reliable signal. Measured on real footage, a man in
        a cream coat scored 0.78 for "wearing a dark jacket" — over any sane
        absolute bar — but 1.00 for "wearing a cream coat". The ranking was
        right even where the absolute number was not, because lighting moves
        every score together and a comparison cancels that out.

        **Only one side given:** falls back to an absolute bar, which is worth
        calibrating per clip with scripts/calibrate.py.
        """
        if not include and not exclude:
            return True
        entry = self._cache.get(int(track_id))
        if entry is None or not entry.scores:
            return None
        best_in = max((entry.scores.get(p, 0.0) for p in include), default=None)
        best_ex = max((entry.scores.get(p, 0.0) for p in exclude), default=None)

        if best_in is not None and best_ex is not None:
            return best_in > best_ex and best_in >= min_score
        if best_in is not None:
            return best_in >= min_score
        return best_ex < min_score

    def ref_match(self, track_id: int, reference_ids, min_sim: float) -> bool | None:
        """Does this track look like an uploaded reference, more than any other?

        Not just "close enough": the track must also be the clear winner among
        everything on screen. In a frame with four people CLIP scores all of
        them near any one person, so a bare threshold matches everybody or
        nobody depending on the light.
        """
        if not reference_ids:
            return True
        entry = self._cache.get(int(track_id))
        if entry is None or not entry.ref_sims:
            return None
        from attributes.references import best_match

        for ref in reference_ids:
            sims = {tid: e.ref_sims.get(ref, 0.0) for tid, e in self._cache.items()
                    if e.ref_sims}
            if best_match(sims, min_sim) == int(track_id):
                return True
        return False
