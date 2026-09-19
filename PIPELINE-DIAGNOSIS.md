# Why open-vocabulary targets never acquire

**Status: fixed, on `kevin/tracker-threshold`, branched from
`qinkai/perception`.** Awaiting Qinkai's review — it is his folder.

This was first diagnosed and fixed against `main`. `qinkai/perception` then
landed a 9,000-line rewrite that deletes `runtime/state.py`, `runtime/task.py`,
`stages/` and `detectors/registry.py`, so those fixes were rebuilt against the
new structure. **The bug survived the rewrite**: `runtime/world.py` still
constructed a bare `ByteTrack()`, in `Shared`'s default and again in
`Shared.reset()`, and `scripts/calibrate.py` did too.

Section 3 below describes the fix against the *old* layout and is kept for the
reasoning. The shipped version is in `runtime/world.py::new_tracker()`.

Verified before the rewrite: `person's face` reached `TRACKING` 1.4 s after the
spec was posted, where it had never acquired at all. On the new branch the full
suite passes, 321 tests, with three new ones that fail without the change.

**Symptom:** `track the yellow duck`, `follow the person's face` and similar
compile cleanly, reach `applied`, and then sit in `ACQUIRING` until the timeout
fires `no_target`. `track the person` works perfectly. The annotated stream shows
no boxes for the failing prompts.

---

## 1. Root cause

`ByteTrack` refuses to create a track for any detection scoring below
`track_activation_threshold + 0.1`. With the default `0.25`, the real gate is
**0.35**. Open-vocabulary detections for anything that isn't a common COCO object
score **0.12–0.28**. Every one of them is discarded before it can become a track.

`supervision/tracker/byte_tracker/core.py`:

```python
# line 78 — the +0.1 is not documented in the constructor docstring
self.det_thresh = self.track_activation_threshold + 0.1

# line 204 — only scores above the threshold become STrack candidates at all
remain_inds = scores >= self.track_activation_threshold

# line 322 — and then they must clear det_thresh to be activated
for inew in u_detection:
    track = detections[inew]
    if track.score < self.det_thresh:
        continue
    track.activate(self.kalman_filter, self.frame_id)
```

`perception/runtime/task.py`, `PreparedTask.build()` constructs the tracker with
every default:

```python
tracker=ByteTrack(),
```

So `CFG.CONF_THRESHOLD` governs only YOLOE's own cutoff. The tracker applies a
second, stricter, invisible gate that no config value reaches. Lowering
`CONF_THRESHOLD` makes the situation *worse-looking*, not better: more
detections pass YOLOE and are then silently eaten, so the logs show inference
succeeding while `visible` stays `false`.

### Measured, on live frames, same frames for each row

| prompt | raw detections | score range | gate 0.35 (current) | gate 0.12 |
|---|---|---|---|---|
| `yellow duck` | 10/10 frames | 0.15–0.27 | **0/10 tracked** | 4/10 |
| `person's face` | 10/10 frames | 0.12–0.28 | **0/10 tracked** | 3/10 |
| `person` | 10/10 frames | 0.85–0.88 | 10/10 | 10/10 |

The detector is doing its job on every frame. The tracker is throwing the result
away on every frame.

---

## 2. Ruled out

Recorded so nobody re-walks these:

- **fp16.** `quantize()` returns `16` on CUDA while an ad-hoc probe defaults to
  fp32. Measured both on identical frames: `avg=0.162` vs `avg=0.162`. No effect.
- **`class_name` string matching.** `candidates_for` does an exact `in set(...)`
  match. Verified the detector emits exactly `"person's face"`, apostrophe
  intact, and the filter matches it. Not the problem.
- **CLIP verify.** These specs now carry `verify: null`, so the stage does not
  run. (It *was* a real bug earlier — see §5 — but it is not this one.)
- **`CONF_THRESHOLD`.** Already lowered to `0.15` in the running process. No
  effect, for the reason above.
- **CORS.** Fixed locally, unrelated to acquisition. See §6.

---

## 3. Fix, part 1 — make the tracker's gate configurable and correct

**File:** `perception/config.py`

```python
    # 3. Timing / state machine
    ...
    # ByteTrack will not create a track below (this + 0.1). Open-vocabulary
    # detections for uncommon objects land at 0.12-0.28, so the stock 0.25
    # (a real gate of 0.35) discards them all. Keep this at or below
    # CONF_THRESHOLD - 0.1 so the detector's cutoff is the only one that bites.
    TRACK_ACTIVATION_THRESHOLD: float = _f("TRACK_ACTIVATION_THRESHOLD", 0.02)
    LOST_TRACK_BUFFER: int = _i("LOST_TRACK_BUFFER", 60)
    MIN_MATCHING_THRESHOLD: float = _f("MIN_MATCHING_THRESHOLD", 0.8)
    TRACK_FRAME_RATE: int = _i("TRACK_FRAME_RATE", 30)
```

**File:** `perception/runtime/task.py`, in `PreparedTask.build()`

```python
            tracker=ByteTrack(
                track_activation_threshold=CFG.TRACK_ACTIVATION_THRESHOLD,
                lost_track_buffer=CFG.LOST_TRACK_BUFFER,
                minimum_matching_threshold=CFG.MIN_MATCHING_THRESHOLD,
                frame_rate=CFG.TRACK_FRAME_RATE,
            ),
```

Add `from config import CFG` to that module.

Note on `frame_rate`: ByteTrack computes `max_time_lost` from it, so it controls
how long a lost track survives *in frames*. The loop runs at ~10 fps with YOLOE,
not 30. Leaving it at 30 makes lost tracks persist roughly three times longer in
wall-clock, which currently helps with flapping. Do not "correct" it to 10
without also raising `LOST_TRACK_BUFFER`, or targets will drop faster.

### Why 0.02 and not, say, 0.15

`det_thresh = tat + 0.1`. For a detection at 0.15 to activate, `tat` must be
≤ 0.05. `0.02` gives a real gate of `0.12`, just under the observed floor. The
detector's own `CONF_THRESHOLD` then becomes the single meaningful cutoff, which
is the behaviour the config already implies.

---

## 4. Fix, part 2 — the scores are marginal, which is a separate problem

Even with the gate at 0.12, the duck is visible on only **4/10** frames and the
face on **3/10**. `CFG.ACQUIRE_HITS = 3` requires **three consecutive** visible
frames to reach `TRACKING`, and `LOST_MISSES = 5` drops out after five misses.
Flapping detections will therefore acquire late, or oscillate
`ACQUIRING → TRACKING → LOST`, which matches the earlier run that hit `active`
at 4.30 s and then went `LOST`.

Three options, roughly in order of value:

1. ~~**Use a larger YOLOE.**~~ **Tested, and it makes things worse.** Measured on
   identical live frames:

   | weights | `"person's face"` | `"yellow duck"` (not in frame) | `"person"` | ms/frame |
   |---|---|---|---|---|
   | `yoloe-11s-seg` | 0.14 | 0.12 | 0.93 | 38 |
   | `yoloe-11m-seg` | 0.10 | 0.05 | 0.94 | 47 |
   | `yoloe-11l-seg` | 0.12 | **0.02** | 0.96 | 62 |

   Larger models are *better calibrated*, not more sensitive: they are more
   confident on canonical classes and correctly less confident on unusual text
   prompts. Note the duck was **absent** for this run — `yoloe-11s` scoring it
   0.12 on empty background is a false positive that `-11l` gets right.

   The real conclusion is about margins, not model size. With the duck present
   `yoloe-11s` scores 0.18–0.27; absent, 0.10–0.14. Signal and noise are about
   0.06 apart. `CONF_THRESHOLD = 0.15` sits between them, which is why it is the
   new default — but it is a thin margin, and pushing thresholds lower buys
   false locks on background rather than better tracking.

2. **Debounce on a window, not a run.** `ACQUIRE_HITS = 3` consecutive is
   brittle against a 40 %-hit-rate detector. "3 of the last 5" would acquire on
   the same evidence without waiting for a lucky streak. This is a change to
   `Machine.on_frame` in `runtime/state.py` — keep a small deque of recent
   visibility instead of `self._hits` / `self._misses`.

3. **Raise `ACQUIRE_TIMEOUT_S`.** Currently 3 s. At 10 fps and a 40 % hit rate a
   run of three takes a while. This only masks the problem and costs demo time,
   so prefer 1 and 2.

---

## 5. Already fixed in `frontend/` (context, no action needed)

Two compile-side bugs found on the way, both fixed and unpushed:

- **`detect` was being stripped of adjectives.** Correct for a fixed-vocabulary
  detector, wrong for an open-vocabulary one. Measured: `"duck"` → 0.00,
  `"yellow duck"` → 0.22; `"face"` → 0.00, `"person's face"` → 0.48. The
  compile prompt now keeps the full phrase for open-vocab models and strips it
  for fixed-vocab ones, selected from `open_vocab` in `GET /models`.

- **`verify` was self-defeating.** The pipeline scores CLIP softmax over
  `[verify.text, f"a {cls}"]`. With `detect: ["yellow duck"]` the model also set
  `verify.class: "yellow duck"`, so CLIP was asked to choose between
  *"a yellow duck"* and *"a yellow duck"* — 0.5 by construction, under any
  `min_score`, discarding every detection. `verify` is now `null` for
  open-vocabulary models, since the detector already filters on the full phrase,
  plus two deterministic guards that strip a degenerate verify if one reappears.

Minor outstanding frontend nit: if a `no_target` event arrives before `active`,
the badge ends up reading `TRACKING` while the timer label reads `NO TARGET`.
Ordering issue in `onStatusEvent` / `settle` in `frontend/app.js`.

---

## 6. Also outstanding: CORS

`perception/server/api.py` has no CORS middleware, so a browser can load
`/video` (an `<img>` is exempt) but every `fetch` to `/spec`, `/health` and
`/models` is blocked. The running process currently has a local patch applied
that is **not committed**; restarting it without this restores the breakage.

```python
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
```

Related, lower priority: `GET /models` reports `available: true` for `yoloe` and
`coco` even when preload failed with `ModuleNotFoundError`. `POST /spec`
therefore accepts the spec with a 202 and the failure surfaces seconds later as
a `failed` event, rather than an immediate 422 naming the missing package. An
import error will never succeed on retry, so the registry should mark such an
entry unavailable.

---

## 7. How to verify a fix

Reproduction is deterministic if you hold an object in frame:

```bash
cd perception
VIDEO_SOURCE=0 DEVICE=auto IMGSZ=640 CONF_THRESHOLD=0.10 python main.py
```

Then from the console, or `curl`, send a spec with
`detect: ["yellow duck"]` and watch `/ws/target`. Before the fix: `visible`
never becomes `true`, `phase` goes `ACQUIRING → NO_TARGET`. After: `phase`
reaches `TRACKING` and `cx`/`area` populate.

A regression test belongs in `perception/tests/test_pipeline.py` using the
`fake` detector with `set_script` to emit a fixed 0.20-confidence detection, and
asserting the pipeline reaches `TRACKING`. That test fails on `main` today and
is the cheapest guard against this class of bug returning — the whole failure is
invisible unless something asserts on confidences below 0.35.

---

## 8. Ownership

| Change | Where | State |
|---|---|---|
| Tracker gate + `new_tracker()` | `runtime/world.py`, `config.py` | on `kevin/tracker-threshold`, needs Qinkai's review |
| `calibrate.py` sampling bias | `scripts/calibrate.py` | same branch — every phrase score it has reported was sampled only from tracks above 0.35 |
| `CONF_THRESHOLD` 0.25 → 0.15 | `config.py` | same branch. **The one judgement call**: lower helps `track` and `highlight`, hurts `count_line` by admitting texture boxes. Worth making per behaviour kind. |
| Larger YOLOE | — | **don't**. Measured worse, see §4 |
| Compile prompt, verify guards | `frontend/api.js` | done, `kevin/frontend` |
| Badge/label ordering | `frontend/app.js` | done, `kevin/frontend` |

Superseded by the rewrite, not carried across: the windowed acquire debounce
(`runtime/state.py` is gone; behaviours have per-kind state machines now), the
registry import-error fix (`detectors/registry.py` → `zoo/registry.py`), and
CORS, which `server/api.py` already handles on Qinkai's branch.

## 9. Convergence worth noting

Qinkai independently measured the prompt-phrasing effect and wrote it into
`perception/CLAUDE.md`: *"Descriptive phrases beat bare nouns — `yellow duck`
12/12 frames, `duck` 0/12."* That matches what the frontend found from the other
side (`"duck"` 0.00, `"yellow duck"` 0.22) and is now enforced in the compile
prompt. Two independent measurements, same conclusion.

The tracker gate is the part neither of us had: his 12/12 was measured at the
detector, where the detections are real. They were being discarded one stage
later.
