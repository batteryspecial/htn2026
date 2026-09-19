"""Encoders for the attribute stage.

`ClipEncoder` is the real one. `HashEncoder` is deterministic nonsense with the
same interface, so every test above this line runs without weights, without a
download and without a GPU.
"""

from __future__ import annotations

import hashlib
import logging

import numpy as np

from config import CFG, resolve_device

log = logging.getLogger("perception.encoders")


def _unit(v: np.ndarray) -> np.ndarray:
    """Normalize rows. Everything downstream assumes unit vectors."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-8)


class ClipEncoder:
    """open_clip ViT-B-32. Loaded once, shared by every behaviour.

    Text encoding is slow enough that it belongs on the loader worker; image
    encoding runs in the loop but only for tracks whose cache entry expired.
    """

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k") -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.dim = 512
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self.device = "cpu"

    def load(self) -> None:
        import open_clip
        import torch

        self.device = resolve_device()
        log.info("loading CLIP %s/%s on %s", self.model_name, self.pretrained, self.device)
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device,
            cache_dir=str(CFG.WEIGHTS_DIR),
        )
        model.eval()
        self._model, self._preprocess = model, preprocess
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        self.dim = model.text_projection.shape[-1] if hasattr(model, "text_projection") else 512
        self._torch = torch

    def encode_text(self, texts: list[str]) -> np.ndarray:
        import torch

        with torch.no_grad():
            tokens = self._tokenizer(texts).to(self.device)
            return _unit(self._model.encode_text(tokens).float().cpu().numpy())

    def encode_images(self, crops: list[np.ndarray]) -> np.ndarray:
        import torch
        from PIL import Image

        if not crops:
            return np.zeros((0, self.dim), np.float32)
        batch = torch.stack([
            # Crops arrive from OpenCV, so BGR to RGB before anything else.
            self._preprocess(Image.fromarray(c[:, :, ::-1])) for c in crops
        ]).to(self.device)
        with torch.no_grad():
            return _unit(self._model.encode_image(batch).float().cpu().numpy())


class HashEncoder:
    """A stand-in with CLIP's interface, none of its weights, and its geometry.

    Getting the geometry right is what makes it a fair test. Real CLIP puts a
    plain class description roughly *between* the attribute phrases, which is
    why contrastive scoring separates them. A fake built from unrelated random
    vectors puts every non-match at 0.5, right on the threshold, and would make
    the tests pass or fail on noise.

    So: each attribute word gets its own direction, an image is encoded as the
    direction of its dominant colour, and any other text (notably the plain
    `"a {class}"` baseline) lands on the centroid of all of them. A matching
    phrase then beats the baseline decisively and a non-matching one loses
    decisively, exactly as real CLIP behaves.
    """

    WORDS = ("red", "green", "blue", "yellow")

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim
        self._basis = {w: self._seeded(w, dim) for w in self.WORDS}
        self._centroid = _unit(np.mean(np.stack(list(self._basis.values())), axis=0))

    @staticmethod
    def _seeded(key: str, dim: int) -> np.ndarray:
        digest = hashlib.sha256(key.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        return _unit(rng.standard_normal(dim).astype(np.float32))

    def _for_text(self, text: str) -> np.ndarray:
        lowered = text.lower()
        hits = [self._basis[w] for w in self.WORDS if w in lowered]
        # No attribute word: this is a plain class description, so it sits
        # between the attributes rather than off on its own.
        return _unit(np.sum(hits, axis=0)) if hits else self._centroid

    def encode_text(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        return np.stack([self._for_text(t) for t in texts])

    def encode_images(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.dim), np.float32)
        out = []
        for c in crops:
            bgr = c.reshape(-1, c.shape[-1]).mean(axis=0) if c.size else np.zeros(3)
            out.append(self._basis[("blue", "green", "red")[int(np.argmax(bgr))]])
        return np.stack(out)
