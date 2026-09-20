# ENRICH — where perception actually stands, and what's worth building next

Written after Qinkai's orchestration-layer testing pass, updated as the work
landed. Supersedes `FOR-QINKAI.md` and `HANDOFF-TO-QINKAI.md`.

The original finding still holds: the failures from testing looked like ten
separate bugs and were closer to **three**. One of the three is now closed and
two are open. Suite is at **423 passed, 10 skipped**, no weights or GPU needed.

| | |
|---|---|
| **Closed** | Root cause 3 — the capability manifest. Gesture vocabulary reaches the agent, four aux models registered, every behaviour kind in the contract implemented. |
| **Open** | Root cause 1 — query expansion for weak detections. Root cause 2 — image queries still route through text. |
| **Unverified** | Everything that needs a camera or an optional package. This is the real risk column. |

---

## What testing established

Verified working:

- arbitrary retargeting of the tracked subject
- general-purpose / abstract arguments through to the compiler
- directed behaviours (`watch`, guard-style specs) compiling and arming

Observed failures: dark Raspberry Pi box missed while a paper box is found;
"watch" tracking a wristband; an uploaded Hacker Badge photo never matching;
the keyboard key-pointing spec refused; the index-finger guard arming and then
never firing.

Of those five, **two are now fixed** (keyboard, finger guard) and three are
root causes 1 and 2 below.

---

## The threshold theory is closed

Worth stating up front so nobody re-derives it: the "one global threshold is
silently eating weak detections" hypothesis was correct, was found, measured
and fixed *before* this testing pass.

The gate was not in the detector — it was `supervision.ByteTrack`, which sets
`det_thresh = track_activation_threshold + 0.1` and so refuses to *create* a
track below 0.35. Detections were real and were discarded one stage later.

Current tree: `CONF_THRESHOLD = 0.15`, `TRACK_ACTIVATION_THRESHOLD = 0.02`
(real gate 0.12), routed through a single `new_tracker()`, with a regression
test asserting `gate <= CONF_THRESHOLD` so it cannot drift back.

**So the remaining detection failures are not threshold failures.**

---

# Part I — what shipped

## The capability manifest (root cause 3, closed)

The finger guard failed for a smaller reason than "finger tracking is hard".
`skills/pose.py` read, in full:

```python
GESTURES = {"hand_raised": hand_raised}
```

One gesture. And `/models` exposed `roles` and `/behaviors` exposed `kinds`,
but **nothing exposed `GESTURES`**. The agent knew `pose_trigger` existed, had
to fill in a `gesture` param, and the only implemented value was
`hand_raised`. So "report a raised index finger" silently became `hand_raised`,
validated cleanly, armed, and watched wrists forever.

Had it invented a name, `detect_gesture` would have rejected it loudly. The
silence is the proof it picked the real one.

**The refuse-to-arm machinery was never missing.** `needs_roles`,
`UnsupportedBehavior`, `NotYetBuilt`, `check_available` — all present and
wired. The vocabulary just never reached the compiler.

Now:

- `GET /models` reports `gestures`, spanning roles and narrowed to what is
  actually loaded. A rule whose model is missing is not a capability.
- `ground()` in the orchestrator pulls it off the round trip it was already
  making for `open_vocab` — no extra call.
- `situation()` states it every turn, in the same shape as the existing
  `FIXED VOCABULARY` warning: *these and no others, and do not install the
  nearest match and report it as done.*

It lives in the per-turn situation rather than `BEHAVIORS.md` because the
vocabulary is **code**, and it changed three times in one afternoon.

## Four aux models, one contract

`skills/pose.py` was written as the first of a series. There are now four, and
the pattern held without modification.

| role | model | points | notes |
|---|---|---|---|
| `pose` | `pose`, `pose_fullbody` | 17 | COCO-17. `pose_fullbody` is the larger checkpoint, same order, swappable |
| `hands` | MediaPipe Hands | 21/hand | CPU, no weights file, no GPU contention |
| `wholebody` | RTMPose COCO-WholeBody | 133 | body + feet + face + both hands, ONNX on CPU |
| `ocr` | EasyOCR | — | single characters; the only one too slow to run inline |

**Why `hands` is its own role and not a second `pose` entry.** Exactly one
model per role is active at a time. Sharing the role would make "raise your
hand" and "raise your index finger" mutually exclusive — installing one would
pause the other. Roles are a contract about keypoint *layout*, not just about
what a model can see.

**`needs_roles` is now per instance, not per kind.** This is the actual fix for
the silent guard. `pose_trigger` used to declare `needs_roles = ("pose",)` as a
class attribute, so a finger guard armed the moment a *body* model loaded. It
is now set in `validate()` from `gestures.ROLE_OF[...]`, so a finger gesture
needs `hands` and a body gesture needs `pose`, and the wrong one pauses with a
reason.

**The whole-body integration was much smaller than expected**, because
COCO-WholeBody's first 17 points are COCO order and its two hand blocks use the
same 21-point topology MediaPipe does. So `skills/wholebody.py` *slices* —
`body()`, `left_hand()`, `right_hand()` return arrays the existing rules
already accept. Nothing was reimplemented, and there are tests feeding
`hand_raised` and `index_finger_raised` those slices directly, using the same
shape builder the hands tests use, so a layout change fails loudly.

Gestures now available: `hand_raised`, `index_finger_raised`, `open_palm`,
`fist`, `mouth_open`, `both_hands_raised`.

One design note that matters on a demo floor: "finger extended" is measured as
**distance from the wrist**, not tip-above-joint. The obvious test silently
assumes an upright hand, and people point sideways and palm-down. There is a
parametrized test that flips the hand 180°.

## The keyboard (demo #17)

`keyboard` was the last unimplemented kind in the contract. `kinds.planned` is
now empty.

The refusal you hit was correct behaviour — the agent read a vocabulary that
genuinely lacked the primitive. What was missing was per-key coordinates, and
`BehaviorKind` already had the `keyboard` slot and `zoo/roles.py` already had
the `ocr` role, both waiting.

**OCR is the first aux model that cannot run inline.** Pose, hands and
wholebody are tens of milliseconds and the loop calls them directly. EasyOCR is
150–400ms and a loop that waited would run at 3fps. So the model owns a thread
and uses the same one-slot, newest-wins handover as `Capture`: the loop offers
the newest frame and takes the most recent finished read, neither blocking. A
keyboard does not move, so a 300ms-old read is still true.

**The fit is the interesting part.** OCR never reads every key and never the
same ones twice — but a keyboard is rigid and QWERTY is known, so fitting the
template to what *was* read supplies every key that wasn't. Six characters
locate all forty-seven, and there is a test asserting exactly that.

RANSAC matters more than the model does. A letter read off a mug or a sticker
is one bad correspondence, and a least-squares fit would bend the whole
keyboard to it; there is a test that adds a stray `m` at (2000, −900) and
asserts nothing moves more than 5px. Homography above four points, partial
affine at three to five — wrong for an angled keyboard, right for a top-down
one, and a slightly-off badge beats no badge. Smoothing is applied to the
matrix, normalised by the bottom-right term first, so a key OCR missed this
frame does not jump when it returns.

Repeated letters collapse into one badge carrying every position, so the `h` in
HACK THE NORTH reads `1·7·14` rather than three overlapping circles. All three
step modes work, and `POST /behaviors/{id}/advance` is tested over HTTP
including both refusal paths.

## Two bugs worth recording

**`CFG.MAX_HANDS` and friends never existed.** `skills/hands.py` was written
referencing three config attributes that were never added. The suite was green
because the only caller is `load()`, which never runs while `mediapipe` is
absent — so it would have been an `AttributeError` on the GPU host the moment
the dependency was installed, and nowhere before. Caught a turn later by
accident. The general shape — *code behind an optional-dependency guard is
untested code* — applies to `rtmlib` and `easyocr` equally.

**`self.step` shadowed `Behavior.step`.** The keyboard's cursor was named
`step`, which is the base class's per-frame entry point, so every frame raised
`'int' object is not callable`. Caught by the tests, renamed to `cursor`, and
there is a comment on the attribute so nobody re-does it.

---

# Part II — what is still open

## Root cause 1 — the signal-to-noise margin, not the threshold

**Untouched. Still the highest value-per-hour item on the page.**

This is the dark Pi box and the wristband.

The measured margin on `yoloe-11s` is about **0.06**: a present target scores
0.18–0.27, an absent one 0.10–0.14. Below roughly 0.15 the detector starts
locking onto wall texture, and `pick: largest` will cheerfully select a big
patch of it.

That gap is the actual ceiling. A dark, low-contrast, non-canonical object
lands inside it. "Box" also carries heavy prototype bias — the text embedding
sits near *cardboard carton*, and a matte black PCB enclosure is nowhere near
it. The wristband is the same mechanism with a different ending: not that the
wristband beat the watch, but that both were weak and the wrong one cleared
first.

**What actually helps, cheapest first:**

1. **Query expansion in the compiler, union the results.** The agent already
   produces the detect list. Emitting `["cardboard box", "small black plastic
   case", "electronics enclosure"]` instead of `["box"]` costs one prompt
   change and no perception work. Detection runs once on the union, so extra
   phrasings are free — the compiler already knows to keep the operator's full
   phrase for open-vocab models, so the plumbing is there.
2. **Lock camera exposure and gain.** Webcams hunt continuously, and a hunting
   exposure moves scores across the 0.06 band frame to frame. This is closer to
   the lighting intuition than anything else here.
3. **Dump the frame the detector actually receives.** Not the preview — the
   post-resize, post-conversion array. It settles the lighting question in
   five minutes instead of by argument.

Do not push thresholds lower. That trade was already made and the headroom is
gone.

## Root cause 2 — references re-rank, they do not detect

**Untouched.** This is the badge, and the diagnosis is sharper than "the model
is bad at it".

`attributes/references.py` works well: it CLIP-embeds an uploaded photo and
matches it against tracks with a cosine threshold *plus* a 0.05 margin over the
runner-up, so an ambiguous frame matches nobody rather than the wrong body.
That margin rule is good design and should stay.

But look at the signature:

```python
def best_match(sims: dict[int, float], threshold: float, margin: float = MARGIN)
```

`sims` is keyed by **track id**. References rank tracks that already exist.
They are a re-ranker over detections, not a detector.

So the badge path was: agent describes the photo as "purple digital device" →
YOLOE searches for that text → finds nothing → `sims` is empty → `best_match`
returns `None`. The CLIP embedding was computed correctly and had nothing to
rank. Every pixel of the badge was discarded at the text bottleneck before the
reference system was consulted.

**The fix is in the stack.** YOLOE takes visual prompts natively —
`get_visual_pe` with a box/mask prompt in place of `get_text_pe` — which makes
an image query a *detector input* rather than a post-hoc filter.

One caveat that is also a better demo: the uploaded photo is a straight-on,
isolated, evenly-lit product shot on a transparent background, and the live
feed is a hand-held badge at an angle under room light. Even a correct visual
prompt crosses that gap. **"Show it to me and I'll remember it"** — grab the
reference crop from our own camera, same sensor, same lighting, no gap.
`references.add_vector()` already exists for precisely this.

Note this needs the GPU host to develop against: the ultralytics visual-prompt
path is genuinely different from the text path (`YOLOEVPSegPredictor`, a
different `predict` signature), not a parallel of `get_text_pe`. Writing it
blind and shipping it unrun is the trap.

## One model instead of three

Loading `wholebody` does **not** satisfy a `pose` or `hands` gesture, even
though it can see everything they can. `needs_roles` is AND-semantics in
`world.revalidate`, so making one model substitute for another means changing
that to "any of these". Documented in the module docstring, with a test pinning
current behaviour. That is the obvious follow-up if the frame budget gets
tight.

## `orchestrator/documents/` is gitignored and absent

**This is a demo-day blocker and it is nobody's current task.**

`prompts.py` builds the agent's catalogue from `documents/SELECTORS.md` and
`documents/BEHAVIORS.md`. That folder is excluded by `orchestrator/.gitignore`
and is not in the clone. `_read()` raises a deliberate, well-written error
about it — which means **a fresh checkout has no agent prompt at all**.

Anyone who clones before Saturday gets a service that will not start. Restore
the files or un-ignore the folder.

---

## On swapping the detector

Our own measurements answer this, and they answer against it:

| weights | `"person's face"` | `"yellow duck"` (absent) | `"person"` | ms/frame |
|---|---|---|---|---|
| `yoloe-11s-seg` | 0.14 | 0.12 | 0.93 | 38 |
| `yoloe-11m-seg` | 0.10 | 0.05 | 0.94 | 47 |
| `yoloe-11l-seg` | 0.12 | **0.02** | 0.96 | 62 |

Bigger YOLOE is better *calibrated*, not more sensitive — more confident on
canonical classes, correctly less confident on odd text. Scaling up does not
buy the dark Pi box.

RF-DETR is a different objection: it would improve boxes on COCO classes and
**lose open vocabulary**, which is the entire premise of the device. That
trades the differentiator for accuracy on categories nobody will be impressed
by.

The useful framing is a **router**, and it is the same manifest that closed
root cause 3:

| query shape | backend |
|---|---|
| COCO noun (person, laptop, cup) | `ultralytics_fixed` — faster, better calibrated |
| novel / descriptive noun | `yoloe` text prompt |
| hand, body, face, gesture | aux model (`pose` / `hands` / `wholebody`) |
| reference image or held-up object | `yoloe` visual prompt — not built |

Three of the four rows now exist. Making the enriched frame name which backend
answered is still worth doing: visible model switching reads as sophistication,
and it is a label, not a feature.

---

## Verify before demo

Ranked by how badly it hurts if it is wrong on the day.

**1. None of the four optional packages are installed on the Mac.**
`mediapipe`, `rtmlib`, `onnxruntime` and `easyocr` are all absent, so `hands`,
`wholebody` and `ocr` report `available: false` with a reason and the service
boots fine — but **nothing any of those three models actually produces has ever
run**. The geometry over their output is thoroughly tested; the output itself
is not. Install them on the GPU host early and look at the enriched frame.

**2. Can EasyOCR read keys off a real keyboard?** That is the whole keyboard
demo and it is camera-only. If it reads too few characters to fit you get
`SEARCHING` rather than a wrong answer, which is the right failure — but find
out now. `OCR_HZ` and `OCR_MIN_CONF` are the knobs.

**3. Counting.** Judges will ask "how many people do you see."
`behaviors/count_line.py` exists and is unit-tested but has never run against a
real crowded frame. Two things that are known to interact:

- `CONF_THRESHOLD` is at 0.15, tuned for `track` (one subject, missing it is
  total failure). `count_line` wants it **higher** — uncertain texture boxes
  inflate counts, and `MIN_BOX_FRAC` (currently 0.0) only catches boxes that
  are *small*, not boxes that are *uncertain*. Counting is the behaviour that
  forces the per-behaviour threshold decision that was flagged and deferred.
- Track ID stability under occlusion. If IDs churn as people cross, the count
  inflates. `LOST_TRACK_BUFFER` is the knob.

**4. Tuning constants that are guesses.** `MOUTH_OPEN_RATIO = 0.35`,
`HAND_DETECTION_CONF = 0.4`, `WHOLEBODY_MODE = "balanced"`. All module
constants or config for exactly this reason.

---

## Demos, ranked by wow-per-hour

1. **Point-to-command** — point at an object, the device tracks it. Hand
   landmarks now exist, so this is wrist → index tip, extended, intersected
   with current detections. Composes with retargeting, which already works.
   Judges have not seen this at a hackathon table.
2. **"Show me, I'll remember it"** — hold up any object, device tracks it.
   Fixes the badge, no domain gap. Needs root cause 2.
3. **Keyboard spelling** — *built*. Point it at a keyboard, ask for HACK THE
   NORTH, get numbered badges in typing order with a path through them. Use
   `step_mode: auto` on the floor; it has a rhythm and a wall of badges is a
   photograph. Rehearse it — it is the most deterministic thing on the list
   once OCR locks.
4. **Absence guard** — "tell me if my bottle leaves the desk." Inverted
   predicate over detections we already have. Nearly free, and the
   someone-took-my-stuff alert lands.
5. **Mouth-open / both-hands-raised** — free now, and genuinely impossible
   before the 133-point model. Good filler that proves the model swap is real.

---

## What not to chase

Skip general complex-query handling. Right instinct, wrong week — that is the
startup, not the hackathon. Judges do not reward asymptotic generality; they
reward a legible capability boundary plus three demos that land clean.

And lean *into* the refusals. An agent that says "I only have `hand_raised`,
`index_finger_raised`, `open_palm` and `fist`" is more impressive than one that
bluffs — provided it names the alternative. Honest capability reporting is a
feature. The silent no-op guard was the only version of it that was a bug, and
that one is now closed.
