# Handoff — frontend side, and one bug in yours

Kevin. Everything below came out of running your pipeline against a real webcam
with real objects, from the frontend side. Two branches are pushed; one needs
your review because it touches `perception/`.

---

## TL;DR

1. **`kevin/tracker-threshold` needs your review.** ByteTrack silently discards
   every open-vocabulary detection below **0.35**, which is most of them. It
   survived your rewrite — `runtime/world.py` still built a bare `ByteTrack()`.
   Full suite green, 321 passed, three new tests that fail without it.
2. **`scripts/calibrate.py` has been giving you biased numbers** for the same
   reason. Worth re-running after the merge.
3. **One judgement call is yours**: I dropped `CONF_THRESHOLD` 0.25 → 0.15 in
   that branch. It helps `track`, it hurts `count_line`. Details below.
4. You and I independently found the same prompt-phrasing effect. Nice.
5. `kevin/frontend` is frontend-only and touches nothing of yours.

---

## 1. The tracker bug

`ByteTrack` does not gate on `track_activation_threshold`. It gates on
`det_thresh`, which it sets to `track_activation_threshold + 0.1`, and Step 4 of
`update_with_tensors` skips anything under it:

```python
# supervision/tracker/byte_tracker/core.py
self.det_thresh = self.track_activation_threshold + 0.1   # line 78

for inew in u_detection:                                  # line 322
    track = detections[inew]
    if track.score < self.det_thresh:
        continue                                          # never becomes a track
    track.activate(self.kalman_filter, self.frame_id)
```

So the stock `ByteTrack()` refuses to create a track below **0.35**.

That is correct for COCO, where a real object scores 0.8+. It is wrong for
open-vocabulary matching. Measured on `yoloe-11s`, same frames for every row,
subject genuinely in frame:

| prompt | raw detections | score range | tracked (gate 0.35) | tracked (gate 0.12) |
|---|---|---|---|---|
| `"yellow duck"` | 10/10 frames | 0.15–0.27 | **0/10** | 4/10 |
| `"person's face"` | 10/10 frames | 0.12–0.28 | **0/10** | 3/10 |
| `"person"` | 10/10 frames | 0.85–0.88 | 10/10 | 10/10 |

The detector finds the subject on every frame. The tracker discards it on every
frame. Downstream sees no subjects, so behaviours report nothing while the logs
show inference succeeding — nothing anywhere points at the tracker.

**This is why it hid:** nothing in the suite exercised a confidence between the
detector cutoff and 0.35, and `"person"` — the obvious thing to test with — sails
over it at 0.87.

### The fix

`runtime/world.py` now has one `new_tracker()` and everything goes through it,
so the gate cannot drift back:

```python
def new_tracker() -> ByteTrack:
    return ByteTrack(
        track_activation_threshold=CFG.TRACK_ACTIVATION_THRESHOLD,  # 0.02 -> gate 0.12
        lost_track_buffer=CFG.LOST_TRACK_BUFFER,
        minimum_matching_threshold=CFG.MIN_MATCHING_THRESHOLD,
        frame_rate=CFG.TRACK_FRAME_RATE,
    )
```

Three call sites were bare: `Shared`'s `default_factory`, `Shared.reset()`, and
`scripts/calibrate.py`.

`0.02` and not something rounder because `det_thresh = tat + 0.1`, so a
detection at 0.15 needs `tat ≤ 0.05`. This puts the real gate at 0.12 and leaves
`CONF_THRESHOLD` as the only cutoff that bites, which is what your config
already implies.

### `calibrate.py` was sampling biased data

It built its own bare `ByteTrack`, so every phrase score it has reported was
drawn only from tracks that cleared 0.35. The numbers are real but truncated —
they systematically miss the weak-but-correct band, which is exactly the band
the calibration is supposed to characterise. Worth re-running.

---

## 2. The call that is yours: `CONF_THRESHOLD`

I set it 0.25 → 0.15 on that branch, because otherwise the detector filters the
same detections out before the tracker is reached and the fix does nothing.

The trade-off runs both ways and you are better placed to judge it:

- `track`, `highlight`, `pan_to` want it **low** — one subject, and missing it
  is total failure.
- `count_line` wants it **high** — texture boxes inflate counts, which is why
  you added `MIN_BOX_FRAC`. That catches junk that is *small*; it does not catch
  junk that is merely *uncertain*.

It probably wants to be per behaviour kind rather than one global number, but I
did not want to invent that API in your code. Flagged inline in `config.py`.

---

## 3. Findings you can use, or ignore

**We measured the same prompt effect independently.** Your `CLAUDE.md`:
*"`yellow duck` 12/12 frames, `duck` 0/12."* From the frontend: `"duck"` 0.00,
`"yellow duck"` 0.22; `"face"` 0.00, `"person's face"` 0.48. The compile prompt
now keeps the operator's full phrase for open-vocab models and strips it for
fixed-vocab ones, chosen from `open_vocab` in `GET /models`.

Worth noting your 12/12 was counted at the detector. Those detections were real;
they were being thrown away one stage later.

**Bigger YOLOE is worse, measured.** I expected the opposite and was wrong:

| weights | `"person's face"` | `"yellow duck"` (absent) | `"person"` | ms/frame |
|---|---|---|---|---|
| `yoloe-11s-seg` | 0.14 | 0.12 | 0.93 | 38 |
| `yoloe-11m-seg` | 0.10 | 0.05 | 0.94 | 47 |
| `yoloe-11l-seg` | 0.12 | **0.02** | 0.96 | 62 |

Larger models are better *calibrated*, not more sensitive: more confident on
canonical classes, correctly less confident on odd text. Note the duck was
**absent** for that run, so `11s` scoring 0.12 is a false positive that `11l`
gets right.

**The signal-to-noise margin is thin.** Same prompt, `yoloe-11s`: duck present
0.18–0.27, duck absent 0.10–0.14. About 0.06 apart. That is the real reason not
to push thresholds lower — below ~0.15 you start locking onto wall texture, and
`select: largest` will happily pick a big patch of it.

**fp16 is not a factor.** `quantize=16` on CUDA versus fp32 on identical frames:
`avg=0.162` both. I chased this and it was nothing; recording it so nobody else
does.

**A degenerate `verify` was discarding everything** (frontend-side, fixed). The
pipeline scores CLIP softmax over `[verify.text, f"a {cls}"]`. The compiler was
emitting `detect: ["yellow duck"]` with `verify.class: "yellow duck"`, so CLIP
was asked to choose between *"a yellow duck"* and *"a yellow duck"* — 0.5 by
construction, under any `min_score`. Now null for open-vocab, since `detect`
already carries the description.

---

## 4. Branches

| Branch | Base | Contents |
|---|---|---|
| `kevin/tracker-threshold` | `qinkai/perception` | The tracker fix. **Yours to review.** |
| `kevin/frontend` | `main` | Frontend only: compile prompt, verify guards, stage strip, camera pane, badge ordering. Plus `PIPELINE-DIAGNOSIS.md`. |

I had an earlier set of `perception/` fixes against `main` and **deleted them
rather than pushing** — they were against `runtime/state.py`, `runtime/task.py`
and `stages/`, which your rewrite removes. They would have been a nasty conflict
for no benefit. The only one worth carrying across was the tracker gate, which
is what `kevin/tracker-threshold` is.

**`main` currently has neither branch merged.**

---

## 5. Things I checked that you had already handled

So you know what I looked at and do not need to re-explain:

- **CORS** — present on your branch, `allow_origins=["*"]`, `allow_credentials=False`.
  It was missing on `main` and blocked the browser from everything except
  `/video` (an `<img>` is exempt from CORS, so live video sat next to a dead
  control surface, which is a genuinely confusing thing to debug).
- **Missing-package handling** — `zoo/registry.py::check_available` reports
  `kind.requires is not installed` up front, so `POST` gets a reason instead of
  a 202 followed by a late failure. On `main` it accepted and failed seconds
  later.
- **`server/legacy.py`** — thank you for this. The frontend speaks `TaskSpec`
  and your shim translates both directions, so the UI keeps working against the
  rewrite with no changes. Open question below.

---

## 6. Open question for both of us

Does the frontend move to `BehaviorSpec` natively, or does `legacy.py` stay?

The shim works and costs nothing right now. But it maps every target to a
`track` behaviour, so the UI cannot reach `watch`, `count_line`, `privacy`,
`pan_to` or `pose_trigger` — which is most of what you built. If the demo wants
those, the compiler needs to emit `BehaviorSpec` and I should do that work.

Your call on timing: it is a few hours of frontend work and it would be done
against a contract that is still moving.
