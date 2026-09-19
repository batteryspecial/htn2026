# For Qinkai — what to look at

Everything is merged into `main` and running. This is the short list; the
reasoning and measurements are in `HANDOFF-TO-QINKAI.md`.

`main` now has your rewrite + the frontend + one fix in your code.
`kevin/tracker-threshold` is that fix on its own if you want to review it in
isolation.

---

## Please look at these five

### 1. The tracker fix, in your code — `runtime/world.py`, `config.py`

`ByteTrack` gates new tracks on `det_thresh`, which it sets to
`track_activation_threshold + 0.1`. The stock constructor therefore refuses to
create a track below **0.35**. `Shared`'s default, `Shared.reset()` and
`scripts/calibrate.py` all built a bare `ByteTrack()`.

Measured on `yoloe-11s`, same frames per row, subject genuinely in frame:

| prompt | detected | score | tracked, gate 0.35 | tracked, gate 0.12 |
|---|---|---|---|---|
| `"yellow duck"` | 10/10 | 0.15–0.27 | **0/10** | 4/10 |
| `"person's face"` | 10/10 | 0.12–0.28 | **0/10** | 3/10 |
| `"person"` | 10/10 | 0.85–0.88 | 10/10 | 10/10 |

Everything is found and then discarded, so behaviours see nothing while the
logs show inference succeeding. `"person"` clears 0.35 at 0.87, which is why
testing with a person never shows it.

Now behind `new_tracker()`, the only constructor. 321 tests pass; three new ones
in `tests/test_tracking_threshold.py` fail without it.

### 2. A decision that should be yours — `CONF_THRESHOLD` 0.25 → 0.15

Needed, or the detector filters the same detections before the tracker sees
them. But it cuts both ways: `track` and `highlight` want it low, `count_line`
wants it high because texture boxes inflate counts. `MIN_BOX_FRAC` catches junk
that is *small*, not junk that is *uncertain*. Probably wants to be per
behaviour kind. Change it or split it as you see fit.

### 3. Re-run your calibration

`scripts/calibrate.py` built its own bare `ByteTrack`, so every phrase score it
has reported was sampled only from tracks above 0.35 — real numbers, but with
the weak-but-correct band cut off, which is the band calibration is meant to
characterise. Fixed to use `new_tracker()`; worth re-running.

### 4. Don't bother with a larger YOLOE

Measured, because I assumed the opposite:

| weights | `"person's face"` | `"yellow duck"` (absent) | `"person"` | ms/frame |
|---|---|---|---|---|
| `yoloe-11s-seg` | 0.14 | 0.12 | 0.93 | 38 |
| `yoloe-11m-seg` | 0.10 | 0.05 | 0.94 | 47 |
| `yoloe-11l-seg` | 0.12 | **0.02** | 0.96 | 62 |

Bigger is better *calibrated*, not more sensitive. Note the duck was absent
there, so `11s` at 0.12 is a false positive `11l` gets right.

Related and worth knowing: with `yoloe-11s`, the duck scores 0.18–0.27 present
and 0.10–0.14 absent. **About 0.06 between signal and noise.** That is the real
reason not to drop thresholds further — below ~0.15 it starts locking onto wall
texture, and `pick: largest` will happily choose a big patch of it.

### 5. Worth a line in `CLAUDE.md`

Your frame-loop doc says "TRACK: supervision.ByteTrack over all detections",
which is what I believed too. The `+0.1` is undocumented in the library's own
constructor docstring. Anyone adding a detector will hit it again.

---

## Things I checked that you had already done

So you don't re-explain them: CORS is on your branch and correct;
`zoo/registry.py::check_available` reports missing packages up front rather than
accepting and failing late; `server/legacy.py` works — the frontend drives your
rewrite through it unchanged.

We also independently found the same prompt effect. Your `CLAUDE.md`:
*"`yellow duck` 12/12 frames, `duck` 0/12."* From the frontend: `"duck"` 0.00,
`"yellow duck"` 0.22. Your 12/12 was counted at the detector — those detections
were real, and were being dropped one stage later by the tracker.

---

## One question for you

**Does the frontend move to `BehaviorSpec`, or does `legacy.py` stay?**

The shim works and costs nothing today. But it maps every target to a `track`
behaviour, so the UI cannot reach `watch`, `count_line`, `privacy`, `pan_to` or
`pose_trigger` — most of what you built. If the demo wants that breadth, the
compiler needs to emit `BehaviorSpec` and that is frontend work I should start.

Your call on timing, since the contract is still moving.

---

## Running it

```bash
cd perception && VIDEO_SOURCE=0 python main.py     # :8001
cd frontend  && python -m http.server 5174         # :5174, Chrome or Edge
```

`frontend/config.local.js` holds an OpenAI key, is gitignored, and is not in any
commit. Copy `config.local.example.js` and paste your own.

Last verified on merged `main`: LIVE video, 31 fps, yoloe on cuda, a spec
compiled in 4.0 s and accepted as one behaviour, phase coming off `/ws/state`.
