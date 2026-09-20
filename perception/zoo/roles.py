"""What kinds of model exist, and how to build one.

This table is the answer to "I want to add a second pose model, where does it
go". It goes here, as one entry, and nothing else changes: the registry, the
availability rules, the loading, the preloading and `GET /models` are all
role-agnostic and already work.

A **role** is what a model is *for*. Exactly one model per role is active at a
time, so "switch to the full-body pose model" is the same operation as "switch
to RF-DETR" — a different row in the same table.

A **type** is how a model is *built*. Types are globally unique and each one
declares the role it serves, so a typo in models.yaml is caught at boot rather
than at the moment a behaviour needs it.

Implementations live with their domain — detectors in `detectors/`, the
embedder in `attributes/`, aux models in `skills/` — because that is where
their code belongs. This file only knows how to summon them.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

#: Every role the pipeline understands.
#:
#: detector  finds things and names them. The frame loop's main cost.
#: embedder  turns crops and phrases into comparable vectors: attributes,
#:           reference matching, re-identification.
#: pose      body keypoints, for gesture triggers.
#: hands     hand landmarks, for finger gestures. Separate from `pose` because
#:           one model per role is active at a time, and sharing the role
#:           would make "raise your hand" and "raise your index finger"
#:           mutually exclusive.
#: wholebody body, feet, face and both hands at once, 133 points. Overlaps
#:           `pose` and `hands` in what it sees but not in what it fills:
#:           a gesture needs the role it is registered under.
#: ocr       reads text in the frame, for the keyboard skill.
Role = Literal["detector", "embedder", "pose", "hands", "wholebody", "ocr"]

ROLES: tuple[Role, ...] = (
    "detector", "embedder", "pose", "hands", "wholebody", "ocr")


@dataclass(frozen=True)
class Kind:
    """How to build one type of model, and what role it fills."""

    role: Role
    build: Callable[[Any], Any]
    #: Python module that must import for this type to work at all. Missing
    #: means the entry reports unavailable with a readable reason instead of
    #: exploding the first time something needs it.
    requires: str | None = None


def _yoloe(entry):
    from detectors.yoloe import YoloeDetector

    return YoloeDetector(entry.name, entry.weights_path)


def _ultralytics_fixed(entry):
    from detectors.ultralytics_fixed import UltralyticsFixedDetector

    return UltralyticsFixedDetector(entry.name, entry.weights_path)


def _fake(entry):
    from detectors.fake import FakeDetector

    return FakeDetector(entry.name)


def _open_clip(entry):
    from attributes.encoders import ClipEncoder

    # "ViT-B-32/laion2b_s34b_b79k" -> architecture and pretrained tag.
    spec = entry.weights or "ViT-B-32/laion2b_s34b_b79k"
    arch, _, pretrained = spec.partition("/")
    return ClipEncoder(arch, pretrained or "laion2b_s34b_b79k")


def _hash_encoder(entry):
    from attributes.encoders import HashEncoder

    return HashEncoder()


def _ultralytics_pose(entry):
    from skills.pose import UltralyticsPose

    return UltralyticsPose(entry.name, entry.weights_path)


def _mediapipe_hands(entry):
    from skills.hands import MediaPipeHands

    # No weights: the graph ships inside the package.
    return MediaPipeHands(entry.name)


def _easyocr(entry):
    from skills.ocr import EasyOCR

    # No weights: easyocr fetches its own on first load.
    return EasyOCR(entry.name)


def _rtmpose_wholebody(entry):
    from skills.wholebody import RTMPoseWholeBody

    # No weights: rtmlib fetches its own ONNX on first load.
    return RTMPoseWholeBody(entry.name)


#: type -> how to build it. Adding a model family is one entry here plus a
#: class in the folder its role belongs to.
KINDS: dict[str, Kind] = {
    "yoloe": Kind("detector", _yoloe, requires="ultralytics"),
    "ultralytics_fixed": Kind("detector", _ultralytics_fixed, requires="ultralytics"),
    "fake": Kind("detector", _fake),
    "open_clip": Kind("embedder", _open_clip, requires="open_clip"),
    "hash_encoder": Kind("embedder", _hash_encoder),
    "ultralytics_pose": Kind("pose", _ultralytics_pose, requires="ultralytics"),
    "mediapipe_hands": Kind("hands", _mediapipe_hands, requires="mediapipe"),
    "rtmpose_wholebody": Kind("wholebody", _rtmpose_wholebody, requires="rtmlib"),
    "easyocr": Kind("ocr", _easyocr, requires="easyocr"),
}


def kind_of(type_name: str) -> Kind | None:
    return KINDS.get(type_name)


def types_for(role: Role) -> list[str]:
    return sorted(t for t, k in KINDS.items() if k.role == role)
