"""Write a run to a folder a human can actually check.

Numbers in a terminal say a filter ran. They do not say it was *right*. This
writes the crops CLIP actually scored, next to the scores it gave them, so a
wrong answer is obvious instead of hiding behind a passing assertion.

Output:

    <dir>/index.html        open this
    <dir>/frames/*.jpg      annotated frames, sampled
    <dir>/crops/*.jpg       what CLIP saw, one file per track per sample
    <dir>/attributes.csv    track, phrase, score, verdict
    <dir>/report.json       machine-readable summary
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from attributes.clip_cache import crops_from


class Dumper:
    def __init__(self, root: Path, name: str, every: int = 15) -> None:
        self.root = root / "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        self.name = name
        self.every = every
        self.frames_dir = self.root / "frames"
        self.crops_dir = self.root / "crops"
        for d in (self.frames_dir, self.crops_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict] = []
        self.frame_files: list[str] = []
        self.n = 0

    def maybe(self, loop, jpeg: bytes | None) -> None:
        """Called once per published frame. Samples rather than saving all."""
        self.n += 1
        if self.n % self.every:
            return
        view = loop.view
        if jpeg:
            path = self.frames_dir / f"{self.n:05d}.jpg"
            path.write_bytes(jpeg)
            self.frame_files.append(path.name)
        self._crops(loop, view)

    def _crops(self, loop, view) -> None:
        """Save each track's crop and whatever was scored about it.

        `claimed_by` is the point: it shows which behaviour accepted the track,
        so an exclusion that kept the wrong person is visible at a glance.
        """
        tracks = view.tracks
        if tracks is None or len(tracks) == 0 or view.frame is None:
            return
        bank = loop.shared.bank
        crops = crops_from(view.frame, tracks)
        ids = tracks.tracker_id
        names = tracks.data.get("class_name")
        claims: dict[int, list[str]] = {}
        for bid, outcome in view.outcomes.items():
            for i in outcome.matches:
                claims.setdefault(int(i), []).append(bid)

        for i, crop in enumerate(crops):
            if crop.size == 0:
                continue
            tid = int(ids[i]) if ids is not None else -1
            fname = f"{self.n:05d}_t{tid}.jpg"
            cv2.imwrite(str(self.crops_dir / fname), self._thumb(crop))
            scores = bank.scores_for(tid)
            self.rows.append({
                "frame": self.n,
                "crop": fname,
                "track": tid,
                "label": str(names[i]) if names is not None else "?",
                "mean_bgr": [round(float(v)) for v in crop.reshape(-1, 3).mean(0)],
                "claimed_by": ",".join(claims.get(i, [])) or "-",
                "scores": {k: round(v, 3) for k, v in scores.items()},
            })

    @staticmethod
    def _thumb(crop: np.ndarray, height: int = 180) -> np.ndarray:
        h, w = crop.shape[:2]
        if h == 0 or w == 0:
            return crop
        scale = height / h
        return cv2.resize(crop, (max(int(w * scale), 1), height))

    # --- output ---------------------------------------------------------
    def finish(self, summary: dict) -> Path:
        (self.root / "report.json").write_text(
            json.dumps({"name": self.name, **summary, "samples": self.rows}, indent=1))
        self._csv()
        self._html(summary)
        return self.root

    def _csv(self) -> None:
        with (self.root / "attributes.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["frame", "track", "label", "claimed_by", "phrase", "score"])
            for r in self.rows:
                for phrase, score in (r["scores"] or {"-": 0.0}).items():
                    w.writerow([r["frame"], r["track"], r["label"],
                                r["claimed_by"], phrase, score])

    def _html(self, summary: dict) -> None:
        phrases = sorted({p for r in self.rows for p in r["scores"]})
        cards = []
        for r in self.rows:
            bits = "".join(
                f"<div class=s><span>{p}</span>"
                f"<b class='{'hi' if r['scores'].get(p, 0) >= 0.6 else 'lo'}'>"
                f"{r['scores'].get(p, 0):.2f}</b></div>"
                for p in phrases)
            cards.append(
                f"<figure><img src='crops/{r['crop']}' loading=lazy>"
                f"<figcaption><b>{r['label']} #{r['track']}</b>"
                f"<div class=by>kept by: {r['claimed_by']}</div>{bits}</figcaption></figure>")

        frames = "".join(f"<img src='frames/{f}' loading=lazy>" for f in self.frame_files)
        (self.root / "index.html").write_text(f"""<!doctype html><meta charset=utf-8>
<title>{self.name}</title>
<style>
 :root {{ color-scheme: dark; }}
 body {{ background:#111; color:#e8e8e8; font:14px ui-monospace,Menlo,monospace; margin:0; padding:20px; }}
 h1 {{ font-size:18px; }} h2 {{ font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:#888; margin-top:28px; }}
 pre {{ background:#0b0b0b; border:1px solid #2a2a2a; border-radius:6px; padding:12px; overflow:auto; }}
 .grid {{ display:grid; gap:12px; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); }}
 figure {{ margin:0; background:#1a1a1a; border:1px solid #2a2a2a; border-radius:6px; overflow:hidden; }}
 figure img {{ width:100%; display:block; background:#000; }}
 figcaption {{ padding:8px; font-size:11px; }}
 .by {{ color:#888; margin:2px 0 6px; }}
 .s {{ display:flex; justify-content:space-between; gap:6px; }}
 .s span {{ color:#888; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
 .hi {{ color:#6c6; }} .lo {{ color:#c66; }}
 .frames {{ display:grid; gap:10px; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); }}
 .frames img {{ width:100%; border:1px solid #2a2a2a; border-radius:6px; }}
</style>
<h1>{self.name}</h1>
<pre>{json.dumps(summary, indent=1)}</pre>
<h2>what CLIP actually saw ({len(self.rows)} crops) — green passes 0.60</h2>
<div class=grid>{''.join(cards)}</div>
<h2>annotated frames</h2>
<div class=frames>{frames}</div>
""")
