# perception — the reflex layer

One camera becomes many devices. This service runs every frame: it detects,
tracks, checks appearance, evaluates the standing instructions, draws the
result onto the frame, and reports what happened.

It does not decide what the camera should be. That is the agent's job, in the
orchestrator. This layer decides what is true right now, thirty times a second,
and tells the agent when something changes.

```
frame ─▶ ops ─▶ DETECT ─▶ TRACK ─▶ ATTRIBUTES ─▶ BEHAVIOURS ─▶ ACTUATOR ─▶ RENDER ─▶ /video
                                                      │
                                                      └─▶ events ─▶ /ws/events
```

Detection runs **once** per frame on the union of every behaviour's classes,
tracking is shared, and appearance is scored per track and cached. Three
behaviours watching people cost one forward pass, not three.

## The three ideas

**Selector — what counts as a subject.** One shape covers the whole demo list,
which is the point: breadth comes from combining primitives, not from new code.

```jsonc
{"detect": ["person"]}                                     // everyone
{"detect": ["yellow duck"]}                                // the duck
{"detect": ["person"], "include": ["a person in a red hoodie"],
                       "exclude": ["a person in a dark jacket"]}
{"detect": ["person"], "ref_id": "r1", "pick": "ref"}      // the one in a photo
{"detect": ["person"], "relate": {"contains": "shoe"}}     // wearing shoes
```

**Behaviour — what to do about it.** Concurrent and independent: a privacy
blur, a line counter and a follow-cam all run on the same frame, and starting
one never disturbs the others. The server assigns the id.

```jsonc
{"kind": "track", "subject": {...}, "params": {...},
 "render": {"color": "#FFD400", "mask": true, "trail": true, "label": "the duck"},
 "notify": true}
```

**Event — what happened.** Edge-triggered, for the agent. System events
(`model_switched`, `camera_lost`, `camera_ok`) carry a null behaviour id.

## Behaviour kinds

| kind | states | events | built |
|---|---|---|---|
| `highlight` | ACTIVE | `count_changed` | ✅ |
| `track` | ACQUIRING → TRACKING ⇄ EDGE → LOST → SEARCHING | `acquired` `lost` `reacquired` | ✅ |
| `watch` | ARMING → ARMED → FIRED → COOLDOWN | `armed` `missing` `moved` `near` `appeared` | ✅ |
| `count_line` | ACTIVE | `crossed` | ✅ |
| `privacy` | ACTIVE | — | ✅ |
| `pan_to` | GUIDING → REACHED | `reached` | ✅ |
| `pose_trigger` | ACTIVE | `hand_raised` | ✅ |
| `keyboard` | SEARCHING ⇄ LOCKED | `keyboard_locked` `step` | needs an OCR model |

Params per kind: `watch` takes `triggers` and `cooldown_s`; `count_line` a
normalized `line`; `privacy` `keep_ref` and `mode`; `pan_to` `deg` and
`hfov_deg`; `pose_trigger` a `gesture`. Malformed params are refused
synchronously with a reason, not accepted and failed later.

Any kind can also be `PAUSED`, for one of two reasons: the active detector has
no word for its subject, or a model role it needs is unfilled. Either way it is
held with a readable reason and resumes by itself.

## The model zoo

Every model lives in one registry, typed by **role**. Exactly one model per
role is active, so swapping the pose model is the same operation as swapping
the detector.

| role | what it does | without it |
|---|---|---|
| `detector` | finds and names things; the loop's main cost | nothing works |
| `embedder` | crops and phrases to vectors: attributes, references, re-ID | attribute checks never match |
| `pose` | body keypoints | `pose_trigger` is PAUSED |
| `ocr` | reads text in frame | `keyboard` is PAUSED |

Adding a model is an entry in `models.yaml` plus, at most, a class in the
folder its role belongs to. A second pose model is five lines of YAML and
nothing else. Anything that cannot run here — no CUDA, no weights, package not
installed — reports `available: false` with a reason and the service still
boots, which is why the laptop and the GPU host can both run this.

## describe()

"What's on the table right now?" is a different shape from every other query:
it names no subject, and an open-vocabulary detector only finds what you name
it. `POST /describe` closes that gap with two things, neither of which is a
language model.

A **sweep** asks a *spare* detector about a broad everyday vocabulary in one
pass, turning "what is there" into "which of these is there". It never uses the
live detector: re-pointing the one the camera is tracking with would break
every running behaviour for the sake of a question, and the active model
changes under us anyway, so the spare is chosen per request.

**Spatial language** turns geometry into words — *"a laptop, left, large"*
rather than `cx=-0.28, area=0.19` — so the spoken answer and the overlay agree
about the scene.

It returns the objects, a one-sentence `summary` that answers simple questions
with no model at all, and a `prompt` carrying the grounding for the agent's
vision model. **Perception never calls an LLM**; that boundary has a test.

Unbuilt kinds are registered and refuse with a message naming what they need,
so the orchestrator can be written against the whole contract today. Any kind
can also be `PAUSED`: the active model cannot see its subject, so it is held
with a reason and resumes by itself when a capable model returns.

## Endpoints (8001)

| | |
|---|---|
| `POST /behaviors` | start one → `{id}`; 422 with a reason the agent can act on |
| `GET /behaviors` · `DELETE /behaviors/{id}` · `DELETE /behaviors` | list, stop one, stop all |
| `POST /model` · `GET /models` | swap the detector; what this machine can do |
| `POST /references` · `GET /references` · `GET /references/{id}.jpg` | register an appearance from an upload or from the frame |
| `GET /health` · `GET /state` · `POST /hud` | status; full frame state; on-screen instruction |
| `POST /query/count` · `POST /query/look` | median count over a window; what is visible now |
| `POST /describe` | sweep the scene and hand back grounding plus a vision-model prompt |
| `GET /video` · `GET /frame.jpg` · `GET /snapshot` | enriched MJPEG; one enriched frame; one **raw** frame |
| `GET /snapshots/{event_id}.jpg` | the crop an event fired on |
| `WS /ws/state` · `WS /ws/events` | per-frame state; things that happened |

`/snapshot` is deliberately un-annotated: the agent's vision model should see
the world, not our drawings of it.

## Run

From inside `perception/`.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# GPU host
VIDEO_SOURCE=http://localhost:8003/stream .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8001

# Mac (logic only; YOLOE runs on CPU here, see below)
DEVICE=cpu IMGSZ=640 VIDEO_SOURCE=clips/duck.mp4 .venv/bin/uvicorn main:app --port 8001
```

`http://localhost:8001/` is a dev page: the feed, live state, and a box to post
behaviours. The operator UI belongs to the orchestrator.

## Testing against real clips

```bash
pytest                                        # 354 tests, no weights or GPU needed
python scripts/clip_test.py scenarios/*.json --dump out/ --record footage/
python scripts/calibrate.py clips/duck.mp4 "a yellow duck" "a brown duck" --detect duck
python scripts/validate_spec.py plan.json     # would perception accept the agent's output?
```

`--dump` writes an `index.html` with every crop CLIP scored and the score it
gave, because a passing assertion is not the same as a correct answer.
`--record` writes the annotated video, which doubles as backup footage.

## Things learned the hard way

- **Phrasing beats vocabulary.** On a real clip YOLOE found the rubber duck for
  `"yellow duck"` in 12/12 frames and for `"duck"` in **0**. Send the phrase a
  person would say, and send several — the union is free. A selector that
  matches nothing for a few seconds now says so on its own status, so the agent
  can rephrase instead of sitting on a dead behaviour.
- **Phrasing does not transfer between scenes.** The same duck in another room:
  every phrase that scored 12/12 in the first clip scored **0/12** in the
  second, and `"yellow rubber duck"` was the one that worked. Treat a known-good
  phrase as a first guess. `../SELECTORS.md` is the full guide, written for the
  agent to retrieve.
- **Adjectives belong in `detect` when they identify and in `include`/`exclude`
  when they discriminate.** One duck described goes in `detect`; one person
  singled out from several goes in the attribute pair. Getting it backwards
  fails both ways.
- **Pair `include` with `exclude`.** Absolute attribute scores drift with
  lighting; a comparison does not. A cream coat scored 0.78 for "wearing a dark
  jacket" — over any sane bar — but 1.00 for "wearing a cream coat". Attributes
  are decided by which phrase wins.
- **Clothing is scored on the upper body**, not the whole box. A full-body crop
  is mostly trousers, shoes and background.
- **`IMGSZ` matters more on portrait video.** A 1080×1920 phone clip letterboxes
  badly: at 320 the person was found in 8/13 sampled frames, at 640 in 12/13.
- **Re-identification needs appearance.** Leaving LOST requires the *same*
  instance, not another object of the same class, or a passerby cancels the
  guidance arrow and the demo looks broken.
- **`cv2.phaseCorrelate` mutates its inputs** when given a window argument.
  Keeping a reference frame and passing it in every call re-windows it each
  time, which manufactured 3.4° of drift from a completely stationary camera.
- **A behaviour's clock must be the loop's clock.** They are built on the
  worker against wall time and first run against an injected one, so every
  duration-based transition is wrong until they rebase.

## Config

`VIDEO_SOURCE` (URL | webcam index | file, loops), `DEVICE` (auto→cuda→mps→cpu),
`IMGSZ` (640), `PORT` (8001), `CONF_THRESHOLD`, `MIN_BOX_FRAC` (drops texture
detections), `ATTR_TIGHTEN`, `VERIFY_TTL_S`, `STREAM_FPS`, `LOG_LEVEL`.

## Notes for the GPU host

- Weights (~1.2 GB) download into `weights/` on first boot, including a 572 MB
  MobileCLIP text encoder YOLOE pulls silently. Do that once on good WiFi.
- `ultralytics` and `supervision` are pinned; the predict signature has changed
  between versions.
- On a Mac YOLOE is forced to CPU: its text encoder needs float64 and MPS has
  none. CUDA is unaffected.
- A model that cannot run here reports `available: false` and the service still
  boots.
