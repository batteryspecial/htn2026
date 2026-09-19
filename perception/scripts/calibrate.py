"""Find the right attribute threshold for a clip, by measuring rather than guessing.

    python scripts/calibrate.py clips/duck.mp4 "a yellow duck" --detect duck
    python scripts/calibrate.py clips/me.mp4 "a person in a red hoodie" \\
        "a person in a dark jacket" --detect person --dump out/

Scores every phrase against every track, prints the spread, and suggests a
`min_score`. It also writes the crops, because the number that matters is
whether *you* agree with the ranking when you look at what CLIP looked at.

Why this exists: contrastive scores are comparable within a clip and not
across clips. Lighting moves them. A threshold tuned on someone else's footage
is a guess, and guessing is how a demo fails in a room with different lights.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attributes.clip_cache import AttributeBank  # noqa: E402
from attributes.encoders import ClipEncoder  # noqa: E402
from config import setup_logging  # noqa: E402
from zoo.registry import Registry  # noqa: E402
from runtime.world import new_tracker  # noqa: E402


def parse(argv):
    clip, phrases, detect, dump, every = None, [], ["person"], None, 15
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--detect":
            detect = argv[i + 1].split(",")
            i += 2
        elif a == "--dump":
            dump = Path(argv[i + 1])
            i += 2
        elif a == "--every":
            every = int(argv[i + 1])
            i += 2
        elif clip is None:
            clip, i = a, i + 1
        else:
            phrases.append(a)
            i += 1
    return clip, phrases, detect, dump, every


def main() -> int:
    setup_logging()
    clip, phrases, detect, dump, every = parse(sys.argv[1:])
    if not clip or not phrases:
        print(__doc__)
        return 2

    registry = Registry.from_yaml()
    registry.preload()
    det = registry.get("yoloe")
    det.apply(det.prepare(detect))
    enc = ClipEncoder()
    enc.load()

    bank = AttributeBank(enc, ttl=0.0, tighten=True)
    vecs = enc.encode_text(phrases)
    base = enc.encode_text([f"a {c}" for c in detect])
    bank.set_texts({p: vecs[i] for i, p in enumerate(phrases)},
                   {c: base[i] for i, c in enumerate(detect)})

    cap = cv2.VideoCapture(clip)
    if not cap.isOpened():
        print(f"cannot open {clip}")
        return 2
    # Must match the loop's tracker: a stricter gate here would silently sample
    # only high-confidence tracks and bias every phrase score it reports.
    tracker = new_tracker()
    samples: dict[str, list[float]] = {p: [] for p in phrases}
    per_track: dict[int, dict[str, list[float]]] = {}
    crops_saved = 0
    n = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        tracks = tracker.update_with_detections(det.infer(frame))
        if len(tracks) == 0 or n % every:
            continue
        bank.update(frame, tracks, now=time.time() + n)
        ids = tracks.tracker_id
        if ids is None:
            continue
        for i in range(len(tracks)):
            tid = int(ids[i])
            scores = bank.scores_for(tid)
            if not scores:
                continue
            row = per_track.setdefault(tid, {p: [] for p in phrases})
            for p in phrases:
                samples[p].append(scores.get(p, 0.0))
                row[p].append(scores.get(p, 0.0))
            if dump:
                crops_saved += _save_crop(dump, frame, tracks, i, tid, n, scores)
    cap.release()

    if not per_track:
        print("no tracks matched; is --detect right for this clip?")
        return 1

    _report(phrases, per_track, samples)
    if dump:
        print(f"\n{crops_saved} crops written to {dump} — check the ranking by eye")
    return 0


def _save_crop(dump: Path, frame, tracks, i, tid, n, scores) -> int:
    from attributes.clip_cache import crops_from

    out = dump / "calibrate"
    out.mkdir(parents=True, exist_ok=True)
    crop = crops_from(frame, tracks[np.array([i])], tighten=True)[0]
    if crop.size == 0:
        return 0
    best = max(scores.items(), key=lambda kv: kv[1])
    name = f"t{tid:03d}_f{n:05d}_{best[1]:.2f}.jpg"
    cv2.imwrite(str(out / name), crop)
    return 1


def _report(phrases, per_track, samples) -> None:
    print(f"\n{'track':>6}  " + "  ".join(f"{p[:26]:>26}" for p in phrases))
    for tid, rows in sorted(per_track.items()):
        cells = []
        for p in phrases:
            v = rows[p]
            cells.append(f"{statistics.median(v):>10.3f} "
                         f"({min(v):.2f}-{max(v):.2f})".rjust(26))
        print(f"{tid:>6}  " + "  ".join(cells))

    print("\nsuggested min_score")
    for p in phrases:
        medians = sorted(statistics.median(rows[p]) for rows in per_track.values())
        gap, at = 0.0, None
        for a, b in zip(medians, medians[1:]):
            if b - a > gap:
                gap, at = b - a, (a + b) / 2
        if at is None or gap < 0.15:
            print(f"  {p!r}: no clean split (widest gap {gap:.2f}). Either every "
                  f"track is the same, or the phrase is not discriminating.")
        else:
            above = sum(1 for m in medians if m >= at)
            print(f"  {p!r}: {at:.2f}  (gap {gap:.2f}; "
                  f"{above} of {len(medians)} tracks above it)")
    print("\nA phrase only works when its tracks split into two clear groups. "
          "If they do not, put the attribute straight into the detector prompt "
          "instead, e.g. detect=['yellow duck'].")


if __name__ == "__main__":
    raise SystemExit(main())
