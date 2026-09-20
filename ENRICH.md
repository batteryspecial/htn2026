# ENRICH — where perception actually stands, and what's worth building next

Written after Qinkai's orchestration-layer testing pass. Supersedes
`FOR-QINKAI.md` and `HANDOFF-TO-QINKAI.md`, which are now merged and stale.

The short version: the failures from testing look like ten separate bugs and
are closer to **three**, and two of the three already have most of their
machinery in the tree. Nothing below proposes a rewrite.

---

## What testing established

Verified working:

- arbitrary retargeting of the tracked subject
- general-purpose / abstract arguments through to the compiler
- directed behaviours (`watch`, guard-style specs) compiling and arming

Not yet verified, and load-bearing for the demo:

- counting many people at once
- `count_line` — people crossing the frame
- complex multi-clause queries

Observed failures: dark Raspberry Pi box missed while a paper box is found;
"watch" tracking a wristband; an uploaded Hacker Badge photo never matching;
the keyboard key-pointing spec refused; the index-finger guard arming and then
never firing.

---

## The threshold theory is already closed

Worth stating up front so nobody re-derives it: the "one global threshold is
silently eating weak detections" hypothesis was correct, was found, measured
and fixed before this testing pass.

The gate was not in the detector — it was `supervision.ByteTrack`, which sets
`det_thresh = track_activation_threshold + 0.1` and so refuses to *create* a
track below 0.35. Detections were real and were discarded one stage later.

Current tree: `CONF_THRESHOLD = 0.15`, `TRACK_ACTIVATION_THRESHOLD = 0.02`
(real gate 0.12), routed through a single `new_tracker()`, with a regression
test in `tests/test_behaviors.py` that asserts `gate <= CONF_THRESHOLD` so it
cannot drift back.

**So the remaining detection failures are not threshold failures.** That
matters, because it rules out the cheap fix and points at the real ceiling.

---

## Root cause 1 — the signal-to-noise margin, not the threshold

This is the dark Pi box and the wristband.

The measured margin on `yoloe-11s` is about **0.06**: a present target scores
0.18–0.27, an absent one 0.10–0.14. Below roughly 0.15 the detector starts
locking onto wall texture, and `pick: largest` will cheerfully select a big
patch of it.

That gap is the actual ceiling. A dark, low-contrast, non-canonical object
lands inside it. "Box" also carries heavy prototype bias — the text embedding
sits near *cardboard carton*, and a matte black PCB enclosure is nowhere near
it. Both effects push the Pi box into the noise band, where no threshold
setting can separate it from the wall.

The wristband is the same mechanism with a different ending: it is not that
the wristband beat the watch by being a better match, it is that both were
weak and the wrong one happened to clear first.

**What actually helps, cheapest first:**

1. **Query expansion in the compiler, union the results.** The agent already
   produces the detect list. Emitting `["cardboard box", "small black plastic
   case", "electronics enclosure"]` instead of `["box"]` costs one prompt
   change and no perception work. This is the highest value-per-hour item on
   the page — the compiler already knows to keep the operator's full phrase
   for open-vocab models, so the plumbing is there.
2. **Lock camera exposure and gain.** Webcams hunt continuously, and a hunting
   exposure moves scores across the 0.06 band frame to frame. This is closer
   to the lighting intuition than anything else here.
3. **Dump the frame the detector actually receives.** Not the preview — the
   post-resize, post-conversion array. It is routinely darker and blurrier
   than expected, and it settles the lighting question in about five minutes
   instead of by argument.

Do not push thresholds lower. That trade was already made and the remaining
headroom is gone.

---

## Root cause 2 — references re-rank, they do not detect

This is the badge, and the diagnosis is sharper than "the model is bad at it."

`attributes/references.py` works, and works well: it CLIP-embeds an uploaded
photo and matches it against tracks with a cosine threshold *plus* a 0.05
margin over the runner-up, so an ambiguous frame matches nobody rather than
the wrong body. That margin rule is good design and should stay.

But look at the signature:

```python
def best_match(sims: dict[int, float], threshold: float, margin: float = MARGIN)
```

`sims` is keyed by **track id**. References rank tracks that already exist.
They are a re-ranker over detections, not a detector.

So the badge path was: agent describes the photo as "purple digital device" →
YOLOE searches for that text → finds nothing → `sims` is empty → `best_match`
returns `None`. The CLIP embedding of the badge was computed correctly and
then had nothing to rank. Every pixel of the actual badge was discarded at the
text bottleneck before the reference system was ever consulted.

**The fix is already in the stack.** YOLOE takes visual prompts natively —
`get_visual_pe` with a box/mask prompt, in place of `get_text_pe`. That makes
an image query a *detector input* rather than a post-hoc filter.
`detectors/yoloe.py` already splits the text path across the thread boundary
exactly the way the visual path needs (encoder forward on the worker, tensor
assignment on the loop), so it follows a shape that is already written and
tested.

**One caveat that is also a better demo.** The uploaded photo is a straight-on,
isolated, evenly-lit product shot on a transparent background. The live feed is
a hand-held badge at an angle under room light. Even a correct visual prompt
has to cross that domain gap. The better interaction is **"show it to me and
I'll remember it"** — grab the reference crop from our own camera, same sensor,
same lighting, no gap. `references.add_vector()` already exists for precisely
this ("so 'track that one' works with no upload"). Turns the worst failure in
testing into the best demo on the list.

---

## Root cause 3 — the capability manifest

This is the keyboard and the finger guard, and they are the same missing piece.

**The keyboard refusal was correct behaviour.** The keyboard tracked perfectly;
what was asked for — point at these specific keys, in sequence — needs
sub-object localisation, and no `BehaviorSpec` kind expresses it. The agent
was not being timid, it was reading a vocabulary that genuinely lacks the
primitive. `pan_to` already has pan odometry, so the *motion* half is done;
what is missing is per-key coordinates.

**The finger guard is the same gap failing much worse.** `skills/pose.py` is
COCO-keypoint based — `NOSE`, shoulders, elbows, `L_WRIST`, `R_WRIST`. There
are no finger keypoints in COCO. "Index finger raised" is not a hard query, it
is an *unrepresentable* one. The agent saying "finger tracking is hard" was
accurate.

What is not acceptable is what happened next: it armed the guard anyway, and
the guard then no-op'd forever. That is the worst of both outcomes — the
pessimism of a refusal and the silence of a false promise. The rule already
written into the aux-model design says what should have happened:

> a behaviour whose model is missing is paused with a reason instead of sitting
> there looking healthy — `tests/test_pose.py`

A spec no registered model can satisfy should be **rejected at compile time**
with the list of what *is* available, never compiled into a silent guard. The
zoo already has the data for that list: `zoo/roles.py` knows every kind, its
role, and its `requires`. Feeding that manifest into the compile prompt is a
small change and it fixes refusal quality and routing at the same time.

### The specialised tool you asked about

**MediaPipe Hands.** 21 keypoints, CPU, 30fps, no GPU contention with YOLOE.
"Index finger raised" becomes a two-line geometric predicate — index tip above
index PIP while the other tips sit below theirs.

The important part is that this is **not new architecture**. `skills/pose.py`
is explicitly written as the first of a series:

> The pose model is the first aux model, so these also pin the rules that every
> later one inherits.

A hand model is aux model #2, in an existing tested pattern, with an existing
`role` slot in `zoo/roles.py`. Much smaller than it sounds.

And the unlock is bigger than the guard: hand keypoints give a **pointing ray**.
Wrist → index tip, extended, intersected with current detections. That is
"point at what you want me to track" as an input modality — and it composes
with the retargeting that already works.

---

## On swapping the detector

Our own measurements already answer this, and they answer against it:

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

The useful framing is not a swap but a **router** — and it is the same
capability manifest from root cause 3:

| query shape | backend |
|---|---|
| COCO noun (person, laptop, cup) | `ultralytics_fixed` — faster, better calibrated |
| novel / descriptive noun | `yoloe` text prompt |
| hand, body, gesture | aux model (`pose`, + hands) |
| reference image or held-up object | `yoloe` visual prompt |

One registry the agent reads to route *and* to refuse honestly. Two problems,
one small abstraction, and `zoo/registry.py::check_available` already reports
what is installed up front.

Agreed on making the enriched frame name which backend answered — visible
model switching reads as sophistication, and it is a label, not a feature.

---

## Verify before demo

**Counting is the biggest unverified risk.** Judges will ask "how many people
do you see." `behaviors/count_line.py` exists and `tests/test_watch.py` covers
it, so this is verification, not construction — but it has never run against a
real crowded frame.

Two specific things to check, because they are known to interact:

- `CONF_THRESHOLD` is at 0.15, which is tuned for `track` (one subject, missing
  it is total failure). `count_line` wants it **higher** — uncertain texture
  boxes inflate counts, and `MIN_BOX_FRAC` (currently 0.0) only catches boxes
  that are *small*, not boxes that are *uncertain*. This is the per-behaviour
  threshold decision that was flagged and deferred. Counting is the behaviour
  that forces it.
- Track ID stability under occlusion. If IDs churn as people cross, the count
  inflates. `LOST_TRACK_BUFFER` is the knob.

Find out which world we are in before Saturday, not during.

---

## Demos, ranked by wow-per-hour

1. **Point-to-command** — point at an object, the device tracks it. Hand
   keypoints + ray intersection over existing detections. Composes with
   retargeting, which already works.
2. **"Show me, I'll remember it"** — hold up any object, device tracks it.
   Fixes the badge, no domain gap, and `add_vector()` is already there.
3. **Keyboard spelling** — we were closer than it looked. Track keyboard, OCR
   the crop **once**, cache per-key boxes, then it is `pan_to` choreography
   spelling HACK THE NORTH. Deterministic and rehearsable, which matters more
   on a demo floor than it sounds.
4. **Absence guard** — "tell me if my bottle leaves the desk." Inverted
   predicate over detections we already have. Nearly free, and the
   someone-took-my-stuff alert lands.

---

## What not to chase

Skip general complex-query handling. Right instinct, wrong week — that is the
startup, not the hackathon. Judges do not reward asymptotic generality; they
reward a legible capability boundary plus three demos that land clean.

And lean *into* the refusals. An agent that says "I can't do finger tracking,
but I can do hand-raise, pointing, and object-absence" is more impressive than
one that bluffs — provided it names the alternative. Honest capability
reporting is a feature. The silent no-op guard is the only version of it that
is a bug, and that is root cause 3, which is fixable.
