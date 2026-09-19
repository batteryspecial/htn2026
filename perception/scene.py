"""Turning a frame into words: the grounding half of `describe()`.

"What's on the table right now?" is a different shape from everything else the
pipeline does. Every other question names its subject — *the duck*, *people* —
and an open-vocabulary detector only finds what you name it. An open question
names nothing, so there is nothing to detect.

Two things close that gap, and neither of them is a language model:

1. **A sweep.** Ask the detector about a broad everyday vocabulary in one pass
   and see what answers. That turns "what is there" into "which of these is
   there", which is a question the detector can actually take.
2. **Spatial language.** The agent and a vision model both want a sentence,
   not coordinates. "a laptop, centre-left, large" is usable where
   `cx=-0.31, area=0.18` is not, and it makes the spoken answer agree with
   what is drawn on the video.

Perception stops there. It does not call a vision model — it has no LLM and
should not grow one — so it returns the frame, what it can see, and a prompt
built from both. The agent supplies the model.
"""

from __future__ import annotations

from contracts import SceneObject, TrackView

#: An everyday vocabulary for the sweep, phrased the way SELECTORS.md says to
#: phrase things: what a person would say, not a dataset label. Measured on
#: this project's clips, "yellow rubber duck" is found where "duck" is not.
SCENE_VOCAB: tuple[str, ...] = (
    # people and pets
    "person", "face", "hand", "dog", "cat",
    # desk
    "laptop", "computer monitor", "computer keyboard", "computer mouse",
    "mobile phone", "tablet", "pen", "pencil", "eraser", "notebook", "book",
    "sheet of paper", "backpack", "handbag", "wallet", "pair of glasses",
    "headphones", "camera", "remote control",
    # table
    "coffee mug", "drinking glass", "water bottle", "soda can", "plate",
    "bowl", "fork", "knife", "spoon", "banana", "apple", "sandwich",
    # room
    "office chair", "desk", "potted plant", "cardboard box", "clock",
    "television", "umbrella", "scissors", "teddy bear", "yellow rubber duck",
)

#: Above this fraction of the frame an object is "very large", and so on down.
SIZE_BANDS = ((0.25, "very large"), (0.08, "large"), (0.015, "medium"))

#: The sweep asks about a lot of classes at once, so a low bar produces a
#: scene full of things that are not there. Higher than the tracking default.
SWEEP_CONF = 0.35


def where(cx: float, cy: float) -> str:
    """Normalized offsets to a phrase a person would use.

    Deliberately coarse: "upper left" survives the box jitter that "at x=0.31"
    does not, and it is what someone would say out loud.
    """
    horizontal = ("far left" if cx < -0.55 else "left" if cx < -0.18
                  else "far right" if cx > 0.55 else "right" if cx > 0.18
                  else "centre")
    vertical = "top" if cy < -0.3 else "bottom" if cy > 0.3 else ""
    if not vertical:
        return horizontal
    if horizontal == "centre":
        return f"{vertical} centre"
    return f"{vertical} {horizontal}"


def size_of(area: float) -> str:
    for threshold, name in SIZE_BANDS:
        if area >= threshold:
            return name
    return "small"


def to_objects(tracks: list[TrackView], merge: bool = True) -> list[SceneObject]:
    """Describe each track, optionally merging duplicates.

    Three people standing together is worth saying once as "three people, left"
    rather than three times, because that is how the answer will be read out.
    """
    objects = [
        SceneObject(label=t.label, where=where(t.cx, t.cy), size=size_of(t.area),
                    conf=round(t.conf, 2), cx=round(t.cx, 3), cy=round(t.cy, 3),
                    area=round(t.area, 4))
        for t in tracks
    ]
    objects.sort(key=lambda o: -o.area)
    if not merge:
        return objects

    grouped: dict[tuple[str, str], SceneObject] = {}
    for o in objects:
        key = (o.label, o.where)
        if key in grouped:
            grouped[key].count += 1
            grouped[key].conf = max(grouped[key].conf, o.conf)
        else:
            grouped[key] = o
    return sorted(grouped.values(), key=lambda o: -o.area)


def plural(label: str, n: int) -> str:
    if n == 1:
        return label
    if label == "person":
        return "people"
    return label + ("es" if label.endswith(("s", "x", "ch", "sh")) else "s")


NUMBERS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def summarize(objects: list[SceneObject]) -> str:
    """One sentence. Answers simple questions without a vision model at all."""
    if not objects:
        return "I cannot see anything I recognise."
    parts = []
    for o in objects[:8]:
        count = NUMBERS.get(o.count, str(o.count)) if o.count > 1 else "a"
        parts.append(f"{count} {plural(o.label, o.count)} ({o.where})")
    if len(parts) == 1:
        return f"I can see {parts[0]}."
    return f"I can see {', '.join(parts[:-1])} and {parts[-1]}."


def build_prompt(question: str, objects: list[SceneObject],
                 behaviors: list[str]) -> str:
    """A prompt for the agent's vision model, carrying the grounding.

    Telling the model what the detector already found keeps its answer and the
    overlay talking about the same scene, and stops it inventing things that
    are not there. It is told the list may be incomplete, because the sweep
    only knows the words it was given.
    """
    lines = [
        "You are looking at a single frame from a live camera.",
        f"Question: {question}",
        "",
    ]
    if objects:
        lines.append("An object detector found the following in this frame. "
                     "This list is reliable but not exhaustive - it only "
                     "recognises things it has words for, so trust the image "
                     "for anything not listed:")
        for o in objects:
            count = f"{o.count}x " if o.count > 1 else ""
            lines.append(f"  - {count}{o.label} ({o.where}, {o.size})")
    else:
        lines.append("An object detector found nothing it recognises in this "
                     "frame. Describe what you see from the image alone.")
    if behaviors:
        lines += ["", f"The camera is currently: {', '.join(behaviors)}."]
    lines += ["", "Answer the question in one or two plain sentences. "
                  "Do not describe the image in general; answer what was asked."]
    return "\n".join(lines)


class NoSweepDetector(RuntimeError):
    """Nothing spare to sweep with, so the question is answered narrowly."""


def sweep(registry, frame, candidates: list[str], shape) -> list[TrackView]:
    """One detection pass over a broad vocabulary.

    Runs on a spare detector, never the live one, and the choice is made now
    rather than at boot because the active model changes under us: the agent
    can swap it mid-run. Re-pointing whatever the camera is currently tracking
    with would break every running behaviour for the sake of a question.

    Blocking for a moment is fine here — a query is request/response, not part
    of the frame loop.

    Returns TrackViews with no tracker ids, because nothing is being tracked:
    this is one look, not a subscription.
    """
    name = registry.sweep_detector()
    if name is None:
        raise NoSweepDetector(
            "no spare open-vocabulary detector to sweep with; add one to "
            "models.yaml with sweep: true")
    detector = registry.get(name)
    detector.apply(detector.prepare(candidates))
    # Forty classes at the tracking threshold hallucinates a whole room, so
    # the sweep asks for a higher bar. Passed per call, not set globally.
    dets = detector.infer(frame, conf=SWEEP_CONF)

    h, w = shape[:2]
    names = dets.data.get("class_name")
    conf = dets.confidence
    out = []
    for i in range(len(dets)):
        x1, y1, x2, y2 = dets.xyxy[i]
        out.append(TrackView(
            track_id=-1,
            label=str(names[i]) if names is not None else "?",
            conf=float(conf[i]) if conf is not None else 0.0,
            cx=float((x1 + x2) / 2 / w * 2 - 1),
            cy=float((y1 + y2) / 2 / h * 2 - 1),
            area=float((x2 - x1) * (y2 - y1) / (w * h)),
        ))
    return out


def candidates_for(requested: list[str] | None) -> list[str]:
    """What to sweep for. The caller's list wins; otherwise the everyday one."""
    if requested:
        return list(dict.fromkeys(requested))[:64]
    return list(SCENE_VOCAB)
