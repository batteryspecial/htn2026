"""Reference images: "track *that* human".

An uploaded photo becomes a CLIP embedding, and a track matches it when its
appearance is close enough. The subtlety is the second condition:

    cosine >= threshold  AND  >= 0.05 clear of the runner-up

Absolute similarity alone is not enough. In a frame with four people, CLIP
scores all of them fairly close to any one person, so a bare threshold either
matches everybody or nobody depending on the lighting. Requiring a clear
winner is what makes "that one, not the others" mean something, and it fails
safe: when two people look equally like the photo, nothing matches, rather
than the wrong one matching.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger("perception.references")

#: How far ahead of the runner-up a match must be.
MARGIN = 0.05


@dataclass
class Reference:
    ref_id: str
    vector: np.ndarray
    label: str | None = None
    thumb: bytes | None = None


@dataclass
class ReferenceStore:
    """Registered references. Written by the worker, read by the loop."""

    encoder: object | None = None
    items: dict[str, Reference] = field(default_factory=dict)

    def add_image(self, image: np.ndarray, label: str | None = None) -> Reference:
        """Register an uploaded photo. Worker thread: this runs CLIP."""
        if self.encoder is None:
            raise RuntimeError("no encoder; references need CLIP loaded")
        vector = self.encoder.encode_images([image])[0]
        ref = Reference(ref_id=f"r{uuid.uuid4().hex[:8]}", vector=vector,
                        label=label, thumb=_thumb(image))
        self.items[ref.ref_id] = ref
        log.info("registered reference %s (%s)", ref.ref_id, label or "unlabelled")
        return ref

    def add_vector(self, vector: np.ndarray, label: str | None = None,
                   thumb: bytes | None = None) -> Reference:
        """Register from an existing embedding, e.g. the largest person on
        screen, so "track that one" works with no upload."""
        ref = Reference(ref_id=f"r{uuid.uuid4().hex[:8]}", vector=vector,
                        label=label, thumb=thumb)
        self.items[ref.ref_id] = ref
        return ref

    def vectors(self) -> dict[str, np.ndarray]:
        return {r.ref_id: r.vector for r in self.items.values()}

    def __contains__(self, ref_id: str) -> bool:
        return ref_id in self.items

    def __len__(self) -> int:
        return len(self.items)

    def ids(self) -> list[str]:
        return list(self.items)


def best_match(sims: dict[int, float], threshold: float,
               margin: float = MARGIN) -> int | None:
    """The track that matches a reference, or None if there is no clear winner.

    `sims` maps track id to similarity. Returns None when the best is below
    the threshold, or when the runner-up is too close to call: matching the
    wrong person is worse than matching nobody.
    """
    if not sims:
        return None
    ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
    best_id, best = ranked[0]
    if best < threshold:
        return None
    if len(ranked) > 1 and best - ranked[1][1] < margin:
        return None
    return best_id


def _thumb(image: np.ndarray, height: int = 96) -> bytes | None:
    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return None
    small = cv2.resize(image, (max(int(w * height / h), 1), height))
    ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return buf.tobytes() if ok else None
