# Retask — a compiler from plain language to running computer vision

Hack the North 2026. Two people, one webcam, one GPU.

This document starts with what the thing does and why anyone should care, then
goes progressively deeper into how it works, what we decided and why, what
broke, and what is genuinely still wrong with it. The later sections assume
you are comfortable with detection, tracking and async Python.

---

# Part I — The plain version

## The idea

A webcam is a fixed-function device. It does whatever it was programmed to do.
If you want it to count people you buy people-counting software; if you want it
to blur faces you buy something else; if you want it to watch your bag you
probably do not get to have that at all.

The limitation is not the hardware. A 1080p USB webcam and a modern GPU can
already see everything you would need for any of those products. The limitation
is that turning "what the camera sees" into "a job the camera does" requires a
developer.

**Retask removes the developer from that loop.** You describe the job in a
sentence, and the camera starts doing it — and keeps doing it after the
conversation ends.

> "Highlight everyone."
> "Now track the pencil instead."
> "Watch my phone and tell me if it disappears."
> "Blur everyone's face except mine."
> "Count people crossing this line."

Same camera, same models, no retraining. Each sentence turns it into a
different product.

## The one-line framing

**A compiler for natural language, targeting computer vision.**

That is a precise claim, not a slogan. A compiler takes something a human wrote
and emits a program that a machine executes continuously. Ours takes English and
emits a **behaviour specification** — a structured, validated, typed object that
a perception pipeline runs at thirty frames per second until told otherwise.

The analogy holds all the way down. There is a source language (English), a
front end that parses intent (an LLM agent), an intermediate representation (our
shared schema), a validator that rejects malformed programs with specific error
messages, and a runtime that executes them. There is even an optimiser: multiple
behaviours that need the same detections are compiled into a single forward
pass.

## The thing that makes it work

**Two clocks.**

The agent thinks in seconds. The camera acts every frame.

Most "AI plus camera" projects put a language model inside the frame loop —
every frame gets described, sent to a model, and reasoned about. That runs at
two or three frames per second, costs money per frame, and stops entirely when
the network hiccups.

We do the opposite. The agent runs **once**, when you give an instruction. It
configures the pipeline and gets out of the way. The pipeline then runs
detection, tracking, attribute checks, trigger evaluation and rendering every
frame, locally, with no model call at all. The agent is woken again only when
something it asked to be told about actually happens.

That separation is the whole architecture, and everything else follows from it.

---

# Part II — How it works, conceptually

## Four primitives

Breadth comes from composition, not from features. There are four ideas, and
every demo we can do is a combination of them.

**Selector — which things in the frame do I care about?**

```json
{
  "detect": ["duck", "rubber duck", "toy"],
  "include": ["a yellow rubber duck"],
  "exclude": ["a cardboard box"],
  "ref_id": null,
  "pick": "largest",
  "min_score": 0.75
}
```

`detect` is what the open-vocabulary detector searches for. `include` and
`exclude` are CLIP checks on each candidate's crop. `ref_id` matches against a
photo you uploaded. `pick` decides which match to act on when several qualify.

**Behavior — what should the camera do about those things?**

```json
{
  "kind": "watch",
  "subject": { "...Selector..." },
  "params": { "triggers": [ { "type": "missing", "after_s": 2 } ] },
  "render": { "color": "#FFD400", "mask": true },
  "notify": true
}
```

Eight kinds exist: `highlight`, `track`, `watch`, `count_line`, `privacy`,
`pan_to`, `pose_trigger`, `keyboard`.

**Trigger — what counts as something happening?** Lives inside `watch`, and
nests its own Selector, so "tell me if anyone *who isn't wearing a staff badge*
comes near my bag" is expressible without a new feature.

**Event — what just happened?** Flows the other way, from pipeline to agent,
carrying a cropped snapshot of the moment.

The nesting is `Behavior → Selector` and `Behavior → Trigger → Selector`.
Selector is the reusable atom.

## One instruction, end to end

> "Watch over this yellow duck but ignore anybody wearing a black jacket."

1. The console POSTs the sentence to the orchestrator.
2. The agent is handed the current situation — what is already running, which
   detector is active, which gestures exist, and what phrasings have been
   measured before for subjects like this one.
3. It emits a tool call whose arguments *are* a `BehaviorSpec`: kind `watch`,
   subject the duck, with a `missing` trigger and a `near` trigger whose `other`
   selector is people **excluding** black jackets.
4. The orchestrator POSTs that JSON to the pipeline. The pipeline validates it
   against the shared schema and a per-kind parameter check. A malformed spec
   comes back as a 422 with a sentence naming exactly what was wrong, and the
   agent fixes and retries.
5. The pipeline builds the behaviour on a background thread and installs it
   **between two frames**. From that moment it evaluates every frame with no
   further agent involvement.
6. Someone reaches for the duck. The `near` trigger fires, the pipeline emits an
   Event with a snapshot crop.
7. That event wakes the agent, which looks at the crop and says one sentence out
   loud.

Steps 5 and 6 are separated by however long you like. The conversation is over;
the job is not.

---

# Part III — The technical architecture

## Services

| Service | Port | Language | Responsibility |
|---|---|---|---|
| Perception | 8001 | Python 3.14 | Every frame: detect, track, score attributes, run behaviours, actuate, render, emit events |
| Orchestrator | 8000 | Python 3.14 | The agent loop, tools, phrase memory, event consumption |
| Console | 3000 | React 19 / TS 5.9 | Enriched feed, chat, agent trace, events, behaviours, settings |
| Camera service | 8003 | Python | Optional: frame source and, later, stepper control over serial |

`linker/schemas.py` is the shared contract — pydantic v2 with
`extra="forbid"`, imported by both Python services and mirrored by hand into
TypeScript. One file defines the wire format; two copies would drift, and the
drift would only surface during integration.

## The frame loop

```
frame ─▶ ops queue ─▶ DETECT ─▶ TRACK ─▶ ATTRIBUTES ─▶ BEHAVIOURS ─▶ ACTUATOR ─▶ RENDER
         (single         │         │          │             │
          writer)        │         │          │             └─▶ EVENTS ─▶ WS /ws/events
                         │         │          └─ CLIP cache, one score per track, TTL'd
                         │         └─ ByteTrack, shared across every behaviour
                         └─ one forward pass on the UNION of every behaviour's prompts
```

Three properties matter here.

**Detection runs once for all behaviours.** `World.prompt_union()` collects
every active behaviour's classes into one prompt list. Two behaviours watching
people cost one forward pass. Measured: going from one class to eleven left
`person` confidence unchanged at 0.949 → 0.950 while tracks went 1 → 6.

**Attributes are scored once per track and cached with a TTL.** CLIP is the
expensive stage; re-scoring the same track every frame for three behaviours
would triple it.

**Behaviours never draw.** They return layers; the renderer composes them. A
behaviour that drew directly would fight every other behaviour for the frame,
and generated drawing code is the fastest way to crash on stage.

## Atomic world swaps

Changing what the pipeline is doing is the operation most likely to break a live
demo, so it is the one with the most machinery behind it.

A `Builder` thread owns everything expensive: loading a model, encoding prompt
vocabularies, encoding CLIP phrases. It assembles an entire new `World` object
off the frame loop. Only when that world is **complete** does it post an outcome
to a queue the frame loop drains at the top of a frame.

Consequences:

- A failure during preparation is a **no-op**. Whatever was running keeps
  running. There is no half-built state to observe.
- The frame loop is the only thread that installs anything, so there is exactly
  one writer.
- Ids are assigned by the builder, not by the caller, so two agents cannot
  collide on a name.
- The HTTP route returns the id immediately; the behaviour starts a few frames
  later. `GET /behaviors/{id}/status` distinguishes *accepted*, *pending*,
  *installed* and *failed*, because accepting a request is not the same as
  running it.

That last distinction turned out to matter enormously — see Part V.

## The agent loop

LangGraph `StateGraph`, four nodes:

```
START ─▶ ground ─▶ think ─▶ (tool_calls?) ─▶ act ─┐
                     ▲                             │
                     └─────────────────────────────┘   capped at max_steps
                     │
                     └─▶ finish ─▶ END
```

**`ground`** is what makes this better than a one-shot compiler. Before the
model sees the instruction it is handed:

- what is currently running, with state and match counts — including an explicit
  warning when a behaviour is `ACTIVE` with zero matches, which is the failure
  that looks like success
- which detector is live, and whether it is open-vocabulary or fixed — a fixed
  vocabulary makes descriptive phrases a guaranteed 422, so the agent is told
  up front
- which gestures actually exist, narrowed to models that are loaded
- a **prior**: phrasings previously measured for subjects like this one

**`think`** is one model call with tools bound. **`act`** runs the requested
tools *sequentially* — "stop the old one, then start the new one" is a common
pair and running those concurrently is a race that sometimes leaves the pipeline
empty. **`finish`** publishes the reply and the elapsed time.

The loop is step-capped. A demo that thinks for twenty steps has already lost.

### Tools

Twelve, and the interesting design decision is that **the argument schemas are
the shared contracts themselves**. `start_behavior` takes `BehaviorSpec`;
`count_objects` takes `Selector`. The model is shown the same pydantic model the
pipeline validates against, including the class docstring with its five worked
examples. There is no second copy of the schema to drift.

Tools return **prose, never exceptions**. A tool that raises ends the turn; a
tool that returns "REFUSED: watch needs a non-empty 'triggers' list — fix the
call and try again" lets the agent recover. Perception goes to trouble making
its 422s specific precisely so they can be acted on, and throwing that away
would waste it.

### Retrieval

A LanceDB store of measured phrase outcomes — this wording scored 0.22 here,
that one scored 0.00, the bare noun never works.

Two decisions worth defending:

**The rules are not retrieved, they are in the system prompt.** SELECTORS.md is
~5k tokens of nine numbered rules, and Rules 4 and 5 are coupled — retrieving
one without the other produces worse specs than no retrieval at all. Rules
always apply; only *evidence* grows with use, so only evidence is retrieved.

**Search is hybrid, not vector.** The corpus is short, repetitive, and full of
near-synonyms that mean opposite things: "a person wearing a dark jacket" and "a
person wearing a cream coat" are neighbours in embedding space. Pure vector
search returned coats for a query about ducks. Pure full-text cannot match
"duck" to "rubber duck toy". Both retrievers run and fuse by reciprocal rank,
and with no embedding model configured it degrades to full-text, which needs no
network — the right mode for a venue with bad WiFi.

## Protocol discipline

HTTP is HTTP; WebSockets are WebSockets. `POST /chat` is a request that returns
a reply. `WS /ws/trace` is a stream the browser subscribes to. A turn's
intermediate steps go on the socket and are never dribbled into the HTTP
response. The only outbound socket in the orchestrator is the event consumer;
every tool call is HTTP. There are tests asserting that a WebSocket route does
not answer HTTP and vice versa, because mixing them is a common and expensive
mistake.

Channel semantics differ on purpose:

- `/ws/state` is **level-triggered and lossy** — one StateView per frame, newest
  wins. A client that falls behind gets the current truth, not a backlog.
- `/ws/events` is **edge-triggered and replayed** — things that happened must
  not be missed, so a late client gets the backlog.

## The model zoo

Models are entries in `models.yaml`, not code. Each declares a **role**, and
exactly one model per role is active at a time.

| role | models | points | notes |
|---|---|---|---|
| detector | `yoloe` (open-vocab), `coco` (fixed) | — | swappable at runtime |
| embedder | `clip` | — | attributes and references |
| pose | `pose`, `pose_fullbody` | 17 | COCO-17 |
| hands | MediaPipe Hands | 21/hand | CPU, no weights file |
| wholebody | RTMPose COCO-WholeBody | 133 | ONNX on CPU, leaves the GPU alone |
| ocr | EasyOCR | — | own thread; too slow to run inline |

**Roles are a contract about keypoint layout, not about capability.** A
wholebody model can see everything a hands model can, but its 133 points are not
the hands model's 21, and a behaviour indexing one as if it were the other is a
silent wrong answer. Hence separate roles.

`needs_roles` is resolved **per behaviour instance**, from the gesture, not per
kind. A finger guard needs `hands`; a hand-raise guard needs `pose`. Get this
wrong and a finger guard arms the moment a *body* model loads and then watches
wrists forever — which is exactly what happened, and is the bug that motivated
the fix.

Unavailable models report `available: false` with a reason and the service boots
fine. A capability whose model is missing is not a capability, and the agent is
told so.

## The keyboard behaviour

The one item that is a genuine skill rather than a combination of primitives,
and the best single piece of engineering in the repo.

OCR is the first aux model that cannot run inline — EasyOCR is 150–400ms and a
loop that waited would run at 3fps. So it owns a thread and uses the same
one-slot, newest-wins handover as the camera capture: the loop offers the newest
frame and takes the most recent finished read, neither blocking. A keyboard does
not move, so a 300ms-old read is still true.

The fit is the clever part. OCR never reads every key and never the same ones
twice — but a keyboard is rigid and QWERTY is known, so fitting the template to
what *was* read supplies every key that was not. **Six characters locate all
forty-seven**, and there is a test asserting exactly that.

RANSAC matters more than the model does. A letter read off a mug is one bad
correspondence, and least-squares would bend the whole keyboard to it; a test
adds a stray `m` at (2000, −900) and asserts nothing moves more than 5px.
Homography above four points, partial affine at three to five. Smoothing is
applied to the matrix, normalised by the bottom-right term first, so a key OCR
missed this frame does not jump when it returns.

## Testing

| suite | count |
|---|---|
| perception | **452 passed, 10 skipped** |
| orchestrator | **149 passed** |

No test opens a camera, loads real weights, or reaches a network. The rig drives
a whole pipeline by hand with a fake capture, a fake detector and a hash-based
encoder, so what is under test is the real loop, the real worker and the real
behaviour code with only the outside world faked. Tests that need real weights
are marked and skipped unless explicitly enabled.

The agent is tested against a scripted model that implements exactly the surface
the graph uses — `bind_tools` and `ainvoke` — rather than a `BaseChatModel`
subclass, which would drag in serialisation machinery the graph never touches.

---

# Part IV — The decisions, and why

## Why an agent loop instead of one JSON response

The first version was a one-shot compiler: sentence in, one `TaskSpec` out,
pipeline swaps to it. It worked, and it had a ceiling.

One instruction frequently needs several dependent steps. "Track that one" after
a photo upload is: register the reference, get an id back, *then* start a
behaviour carrying it. "Turn 45° left and count people" is: pan, wait for the
`reached` event, then count. "How many people are there?" installs nothing at
all and should answer from a query.

More importantly, the agent can **ask the pipeline things and use the answers**.
It can probe whether a wording finds anything in this room before committing to
it, read a 422 and fix its own call, or notice a behaviour is active with zero
matches and rephrase. A single JSON response cannot do any of that.

## Why the schema is shared, not per-service

`extra="forbid"` on every model means a typo is a 422, not a silently ignored
field. One file, imported by both services, mirrored by hand into TypeScript
with the mirror asserted by tests. The alternative — each service defining its
own view — drifts within a day, and the drift surfaces during integration, which
is the worst possible time.

## Why thresholds are low and why we will not lower them further

`CONF_THRESHOLD = 0.15`, which looks reckless if you are used to COCO detectors
where a real object scores 0.8+.

Open-vocabulary matching is not that. It is a text-image similarity on a
different scale. Measured on `yoloe-11s` with the subject genuinely in frame:

| phrase | score |
|---|---|
| "yellow duck" (present) | 0.18 – 0.27 |
| "person's face" | 0.12 – 0.28 |
| "person" | 0.85 – 0.88 |
| "yellow duck" (**absent**) | 0.10 – 0.14 |

The default 0.25 rejects about half of the correct detections for exactly the
objects a judge will name. But look at the last row: the noise floor with
nothing there is 0.10–0.14. **The usable margin is about 0.06.** Below roughly
0.15 the detector starts locking onto wall texture, and `pick: largest` will
cheerfully select a big patch of it.

That gap is the real ceiling on this system, and it is a property of the model,
not a tuning mistake.

## Why a bigger detector was rejected

The obvious move when detection is weak is a larger checkpoint. We measured it:

| weights | "person's face" | "yellow duck" (absent) | "person" | ms/frame |
|---|---|---|---|---|
| `yoloe-11s-seg` | 0.14 | 0.12 | 0.93 | 38 |
| `yoloe-11m-seg` | 0.10 | 0.05 | 0.94 | 47 |
| `yoloe-11l-seg` | 0.12 | **0.02** | 0.96 | 62 |

Bigger YOLOE is better **calibrated**, not more sensitive — more confident on
canonical classes, correctly *less* confident on odd text. It does not buy the
hard objects, and it costs 60% more time per frame.

RF-DETR is a different objection: it would improve boxes on COCO classes and
**lose open vocabulary**, which is the entire premise. That trades the
differentiator for accuracy on categories nobody will be impressed by.

## Why adjectives go in two different places

SELECTORS.md Rule 4, and it is the least obvious rule in the system.

Adjectives in `detect` **identify**: they change what the detector looks for.
Adjectives in `include`/`exclude` **discriminate**: they filter candidates the
detector already found. Measured: "duck" finds nothing in frames where "yellow
duck" finds it every time — 12 of 12 — so the adjective is doing identification
work there and belongs in `detect`.

Rule 5 says always pair `include` with `exclude`, and the measurement behind it
is the sharpest number in the project: a man in a **cream coat** scored **0.78**
for "a person wearing a dark jacket", while the actual dark jackets scored 0.99.
The *ranking* was right; the absolute number was meaningless. So the decision
must be "which phrase wins" rather than "does this phrase clear a bar" — which
is why `min_score` defaults to 0.75 and why pairing matters more than tuning.

## Why clothing is scored on the upper body

`ATTR_TIGHTEN` crops to the upper body before asking CLIP about clothing. Rule 6
exists because "a person with red shoes" scored **0.00** — the shoes are not in
the crop being scored. Use `relate: {contains: "shoe"}` instead, which is a
geometric containment check, not a CLIP question.

## Why the box filter is a pixel side, not an area fraction

`MIN_BOX_PX = 12` replaced a `MIN_BOX_FRAC = 0.0015` that was wrong twice over.
It scaled quadratically with resolution, so a sharper camera — which sees *more*
detail — got a *higher* cutoff: at 1080p it demanded a 56×56 box, throwing away
any face past about 3.5m. And it never caught the junk it was added for, because
open-vocabulary texture detections are tall narrow strips with plenty of area.

## Why refusals are a feature

An agent that says "I only have `hand_raised`, `index_finger_raised`,
`open_palm` and `fist`" is more trustworthy than one that bluffs — provided it
names the alternative. Honest capability reporting is a design goal, and the
silent no-op guard was the only version of it that was a bug.

---

# Part V — The journey

## The bug that looked like ten bugs

Early testing produced a list of failures that looked unrelated: a dark
Raspberry Pi box missed while a paper box was found; "watch" tracking a
wristband; an uploaded badge photo never matching; a keyboard spec refused; an
index-finger guard arming and never firing.

They were closer to **three** root causes.

### Root cause 1 — the threshold that was not in the detector

The leading theory was that a global confidence threshold was eating weak
detections. The theory was right; the location was not.

The gate was in `supervision.ByteTrack`. It sets
`det_thresh = track_activation_threshold + 0.1` and skips anything under it in
`update_with_tensors`, so the stock 0.25 refuses to **create a track** below
0.35. Detections were real, were logged, and were discarded one stage later.

Then a second gate: ByteTrack fuses IoU with score (`cost = 1 - IoU * score`),
so even at perfect overlap a track only re-matches itself when
`score >= 1 - MIN_MATCHING_THRESHOLD`. At stock 0.8 that is a floor of 0.20.
Measured: a detection at 0.199 creates a track on one frame and is abandoned
forever; 0.21 tracks perfectly. Everything in the 0.15–0.20 band flickered —
detected every frame, tracked for one — which reads to a human as "it only works
up close".

And a third, which cannot be configured at all: ByteTrack confirms a new track
by re-matching on the next frame with `thresh=0.7` written inline. Fused with
score, that demands `score >= 0.30`. Below it a track is born, fails to confirm,
is removed, and is born again — which looks like a mask flashing, not like a
threshold.

**The fix is the part worth presenting.** ByteTrack's constants assume a
COCO-style detector. Rather than fork the dependency or nudge a magic number, we
map confidences **monotonically** into the range ByteTrack was designed for on
the way in and restore them on the way out. Nothing downstream sees a different
ordering, and it is version-independent. `TRACKER_CONF_FLOOR = 0.35` is that
map. A regression test asserts `gate <= CONF_THRESHOLD` so it cannot drift back.

### Root cause 2 — the capability manifest

The finger guard failed for a smaller reason than "finger tracking is hard".
`skills/pose.py` read, in full:

```python
GESTURES = {"hand_raised": hand_raised}
```

One gesture. And `/models` exposed roles, and `/behaviors` exposed kinds, but
**nothing exposed `GESTURES`**. The agent knew `pose_trigger` existed, had to
fill in a `gesture` param, and the only implemented value was `hand_raised`. So
"report a raised index finger" silently became `hand_raised`, validated cleanly,
armed, and watched wrists forever.

Had the agent invented a name, validation would have rejected it loudly. **The
silence is the proof it picked the real one.** The refuse-to-arm machinery was
never missing; the vocabulary just never reached the compiler.

Now `/models` reports `gestures`, narrowed to loaded models, and the per-turn
situation states it every turn in the same shape as the fixed-vocabulary
warning: *these and no others, and do not install the nearest match and report
it as done.*

### Root cause 3 — references re-rank, they do not detect

This one is still open, and the diagnosis is sharper than "the model is bad at
it".

`attributes/references.py` works well: it CLIP-embeds an uploaded photo and
matches it against tracks with a cosine threshold **plus a 0.05 margin over the
runner-up**, so an ambiguous frame matches nobody rather than the wrong body.
That margin rule is good design.

But the signature gives it away:

```python
def best_match(sims: dict[int, float], threshold: float, margin: float = MARGIN)
```

`sims` is keyed by **track id**. References rank tracks that already exist. They
are a re-ranker over detections, not a detector.

So the badge path was: agent describes the photo as "purple digital device" →
YOLOE searches for that text → finds nothing → `sims` is empty → no match. The
CLIP embedding was computed correctly and had nothing to rank. Every pixel of
the badge was discarded at the text bottleneck before the reference system was
consulted.

## The adapter that gated half the product

For a while the console could reach exactly two of eight behaviour kinds. A
legacy adapter mapped every instruction to `track` (later also `highlight`), so
`watch`, `count_line`, `privacy`, `pan_to`, `pose_trigger` and `describe` — six
demos — were unreachable from the UI despite being fully built and tested in the
pipeline.

Deleting that adapter and compiling straight to `BehaviorSpec` was the single
highest-value change in the project: eight demo items sitting behind one file.
A test now pins that the old routes stay gone.

## The model that refused to hold tools

The orchestrator booted, health was green, and every single turn failed:

```
400 — Function tools with reasoning_effort are not supported for gpt-6-astra
in /v1/chat/completions. To use function tools, use /v1/responses.
```

Every turn attaches tools, so this failed 100% of the time. The offered
alternative — turning reasoning off — would trade away exactly the thinking that
turns a vague sentence into a correct selector. Switching to
`use_responses_api=True` keeps both.

## The camera that was quietly sabotaging everything

Late in the build, detections got worse across the board and nobody could say
why. It turned out to be three separate faults stacked, and all three were in
configuration, not in models.

**The backend.** OpenCV's default on Windows is MSMF. Measured, opening a C920
by index:

| backend | time |
|---|---|
| DirectShow | **2.35s** |
| MSMF | 32.90s |
| ANY | 55.64s |

MSMF also orders devices differently from the DirectShow names we enumerate
with, so enumerating with one and opening with the other silently selects a
different camera.

**The pixel format ordering.** The code set FOURCC then resolution, with a
comment confidently asserting that order was load-bearing. It was, and it was
backwards:

```
FOURCC then size  ->  YUY2  1280x720  10.0 fps
size then FOURCC  ->  MJPG  1280x720  30.5 fps
```

Setting the resolution **resets the pixel format** on this driver, so the MJPG
request had no effect at all — and YUY2 at 720p is ~55 MB/s, past what USB 2.0
carries, so the driver silently delivered 10fps. The fix sets size first, then
format, then **verifies the format actually took** and retries the other order
if not, because neither order is right on every driver. The negotiated format is
now reported by `GET /cameras`, so a refused format can never again hide as an
unexplained frame rate.

**The resolution.** 640×480 *crops* a webcam's sensor rather than downscaling
it. Measured on the same scene, 480p → 720p:

| | 480p | 720p |
|---|---|---|
| `person` | 0.841 | **0.882** (at a smaller subject area) |
| `person's face` | 0.072 | **0.152** |
| chairs tracked | peak 3 | **peak 5** |

`person's face` at 0.072 was *below* its own recorded band; at 0.152 it is
inside it.

## Concurrency bugs found by writing tests for them

**Releasing a `VideoCapture` from the wrong thread.** The first camera-switch
implementation released the handle from the API thread while the capture thread
sat inside `read()` on it — undefined behaviour in OpenCV, and the crash would
have landed mid-demo. Only the capture thread touches the handle now; `switch()`
posts a request and waits on an Event.

**Two switches at once.** Two callers shared one pending slot and one completion
Event, so the second overwrote the first's request and **both read the same
result** — one of them reporting success for a camera it never selected. Proven,
not theorised: with the serialising lock removed, the caller asking for camera
`2` gets back `(True, '3')`. On stage that is the console saying "C920" while
the pipeline reads the laptop.

**A cursor named `step`.** The keyboard's cursor attribute shadowed
`Behavior.step`, the base class's per-frame entry point, so every frame raised
`'int' object is not callable`.

**Config attributes that never existed.** `skills/hands.py` referenced three
`CFG` attributes that were never added. The suite was green because the only
caller is `load()`, which never runs while `mediapipe` is absent — so it would
have been an `AttributeError` on the GPU host the moment the dependency arrived,
and nowhere before. The general shape is worth internalising: **code behind an
optional-dependency guard is untested code.**

## A UI that hid the best part

The console's agent trace — the panel that makes the whole architecture legible
— was rendering as a 50px sliver at the bottom of the screen. The instruction
panel above it was `flex: 0 0 auto` (take your natural height, never shrink),
and its natural height was ~820px of a ~1050px column.

The first fix was wrong in an instructive way: capping the panel's height. But
the composer cannot shrink — the mic button, textarea, chips and SEND row are
all fixed-height — and `.panel` clips its overflow, so the cap did not squeeze
the panel, it **cut the buttons off the bottom**. The correct fix is the other
direction: leave the panel content-sized so clipping is structurally impossible,
and shrink the *content* — cap the transcript, which is the only part that grows
without bound.

A second bug was hiding underneath: both the trace and the transcript pinned to
their newest row with `scrollIntoView`, which scrolls **every** scrollable
ancestor. Once the accordion was tall enough to scroll, each arriving trace
entry would have dragged the Events and Behaviours sections out of view several
times a second.

## The deadline we lost to a file we never wrote

The final push was rejected by GitHub:

```
File perception/mobileclip_blt.ts is 571.98 MB; this exceeds the 100 MB limit
```

That file is the MobileCLIP text encoder YOLOE downloads silently on first use.
`detectors/base.py` even documents the hazard — *"YOLOE silently pulls a ~570MB
MobileCLIP text encoder on first use, so without this it lands wherever the
service happened to be started from"* — and the fix (`use_weights_dir()`) pins
it to `weights/` for exactly this reason. But it had already landed in
`perception/` root on an earlier run, and `.gitignore` only covered
`perception/weights/*.ts`, deliberately scoped narrowly to avoid ignoring
frontend TypeScript source.

So a 572MB artefact nobody wrote, in a directory the ignore rule missed, sat in
every local commit and blocked the push past the deadline.

The lesson is not "add a gitignore rule". It is that **auto-downloading
dependencies write files you did not choose, in directories you did not pick**,
and the time to notice is before the file is in five commits.

---

# Part VI — Limitations, honestly

## The 0.06 margin is the real ceiling

A present target scores 0.18–0.27 and an absent one 0.10–0.14. Anything that
lands inside that band is a coin flip. Dark, small, matte, or non-canonical
objects land inside it routinely. "Box" embeds near *cardboard carton*, so a
matte black PCB enclosure is nowhere near it.

Honest expectation for a random object handed over by a stranger: roughly **two
in three**, much better if held close and large in frame, considerably worse for
small dark technical objects.

The mitigations that help are query expansion across *axes* (material, function,
form — not more adjectives on one noun), locking camera exposure so scores stop
drifting across the band frame to frame, and the agent's probe-and-rephrase loop.
Lowering thresholds is not one of them; that headroom is spent.

## A photo does not teach it a new object

References re-rank detections; they do not produce them. If the detector finds
nothing at the object's location, the reference system is never consulted. The
real fix is YOLOE's native visual-prompt path (`get_visual_pe` with
`YOLOEVPSegPredictor`), which makes an image query a *detector input*. It is
genuinely different from the text path and was not attempted blind.

The cheaper and arguably better demo is "show it to me and I'll remember it" —
grab the reference crop from our own camera, same sensor, same lighting, no
domain gap between a product shot on white and a hand-held object under room
light.

## Things that are built but unproven

Three aux models had never produced a single frame of output until the last
hours of the build, because the packages were not installed. Their geometry is
thoroughly tested; their output is not. The keyboard behaviour — RANSAC,
homography, template fit, 263 lines of tests — has still never seen a real
keyboard.

## Counting is the behaviour that forces a deferred decision

`CONF_THRESHOLD` is tuned for `track`, where missing the subject is total
failure. `count_line` wants it **higher**: uncertain texture boxes inflate
counts, and the box filter catches boxes that are *small*, not boxes that are
*uncertain*. One global number cannot serve both, and the per-behaviour
threshold has been flagged and deferred twice.

## Frame budget

Aux models are not free. Three gesture behaviours at once took frame rate from
30 to **15.9** — MediaPipe and RTMPose run on CPU. One at a time stays near 30.

## Privacy and offline behaviour

Per-frame perception is local. But the agent can receive snapshots through its
model provider, so "the output is blurred" does not mean "no raw image ever
leaves the machine". Installed behaviours keep running without network; new
instructions and event narration do not.

---

# Part VII — What we would do next

Ranked by value per hour, which is not the same as by ambition.

1. **Query expansion across axes.** One prompt change, no perception work,
   directly targets the objects that fail. The union mechanic already exists —
   detection runs once on the union, so extra phrasings are free.
2. **Lock exposure and gain.** Four `cap.set` calls in a method that already
   runs on every camera open. Webcams hunt continuously, and a hunting exposure
   moves scores across a 0.06 band.
3. **"Show it to me and I'll remember it."** Fixes the badge failure, needs no
   GPU work, and removes the domain gap rather than trying to cross it.
4. **Point-to-command.** Point at an object; the camera tracks it. Hand
   landmarks now exist, so this is wrist → index tip, extended, intersected with
   current detections. It composes with retargeting, which already works. This
   is the most distinctive idea on the list and the one we ran out of time for.
5. **Name the backend on the enriched frame.** A label, not a feature, but
   visible model switching reads as sophistication.

What we would **not** chase: general complex-query handling. Right instinct,
wrong week. A legible capability boundary plus three demos that land cleanly
beats asymptotic generality.

---

# Appendix — Stack

**Perception:** Python 3.14, PyTorch 2.14 (CUDA 12.6), Ultralytics 8.4 (YOLOE
`yoloe-11s-seg`), supervision 0.30 (ByteTrack), OpenCV 5.0, CLIP, MediaPipe 1.0,
rtmlib + ONNX Runtime 1.30 (RTMPose COCO-WholeBody), EasyOCR 1.7, FastAPI,
uvicorn, pydantic v2, pygrabber (DirectShow enumeration).

**Orchestrator:** LangGraph 1.2, langchain-core 1.6, langchain-openai 1.6,
GPT-6-Astra via the Responses API, LanceDB 0.38 + tantivy (hybrid retrieval),
httpx, websockets, FastAPI.

**Console:** Vite 7, React 19, TypeScript 5.9, MJPEG, three WebSocket channels,
Web Speech API for dictation and TTS.

**Contract:** `linker/schemas.py` — pydantic v2, `extra="forbid"`, mirrored into
TypeScript.

**Tests:** 452 + 10 skipped (perception), 149 (orchestrator). No test opens a
camera, loads real weights, or reaches the network.
