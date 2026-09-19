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
    """A stand-in with CLIP's interface and none of its weights.

    Deterministic: the same text always gives the same vector, and an image is
    encoded from its mean colour, so a "red" crop really does land nearer the
    vector for "red" than for "blue". That is enough to test every code path
    around the encoder, and it never pretends to be a real model.
    """

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def _seeded(self, key: str) -> np.ndarray:
        digest = hashlib.sha256(key.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        return _unit(rng.standard_normal(self.dim).astype(np.float32))

    def encode_text(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), np.float32)
        return np.stack([self._seeded(t) for t in texts])

    def encode_images(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, self.dim), np.float32)
        out = []
        for c in crops:
            bgr = c.reshape(-1, c.shape[-1]).mean(axis=0) if c.size else np.zeros(3)
            # Name the dominant channel so a colour word and a colour agree.
            word = ("blue", "green", "red")[int(np.argmax(bgr))]
            out.append(self._seeded(word))
        return np.stack(out)
