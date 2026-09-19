# CLAUDE.md — perception (owner: Qinkai)

You are Qinkai's coding agent. Work only inside `perception/`. Read this whole file before writing code. `../STACK_BRIEF.md` has the full system diagram and the demo list.

## Project

Hack the North 2026, ~22 hours left. Pitch: **one camera, many devices.** An agent layer (the orchestrator, owned by a teammate) turns sentences into behaviors. This service executes them every frame, draws the result on the frame, and emits events. Breadth of reliable capabilities wins the demo. Every feature must be a combination of the primitives below; bespoke code is allowed only for skills.

## Ownership

| Folder | Owner | Port |
|---|---|---|
| `perception/` | **Qinkai (you)** | 8001 |
| `orchestrator/` (includes frontend) | teammate | 8000 |
| `camera/` | teammate | 8003 |
| `linker/` | shared contracts; do not edit without Qinkai's explicit instruction | — |

## Hardware reality

- **You run on a MacBook**: CPU or MPS only; no CUDA, no TensorRT.
- **Inference runs on an RTX 4070 (12 GB, CUDA)** on a friend's PC, likely under WSL2. Qinkai deploys with `git pull` and a restart.
- **No motor for now.** A webcam feeds the camera service. The "motor" is a virtual actuator that draws guidance arrows for a human to pan the camera. A stepper via Arduino may arrive later behind the same `Actuator` interface.
- Device selection always goes through `config.DEVICE` (auto → cuda → mps → cpu). Never hardcode `"cuda"`. GPU-only paths are guarded; missing backends mark registry entries `available: false`, and the process still boots.
- Mac settings: `PYTORCH_ENABLE_MPS_FALLBACK=1`, `IMGSZ=320`. All logic must be testable on the Mac with looping clips in `clips/`.
- `.pt`, `.engine`, and `.pth` files are gitignored. Use `pathlib` everywhere.

## Architecture rules

1. **Single writer.** Only the frame loop mutates behaviors, the active model, trackers, and state. The API enqueues ops; the loop drains the ops queue at the top of each frame.
2. **Never block the loop.** Model loading, OCR, and reference embedding run in a worker pool. Results come back through a queue and apply on a later frame.
3. **Latest frame only.** The capture thread overwrites a single slot.
4. **Validate before enqueue.** A bad spec returns 422 with a human-readable reason. The agent reads that reason and retries, so make it specific ("label 'duck' is not in rfdetr vocabulary").
5. **Declarative rendering.** Behaviors emit render layers; one renderer composes them. No drawing code outside `render/`.

## Frame loop

```
1. grab latest frame (camera service stream or webcam index)
2. drain ops queue: add/remove behaviors, switch model, apply finished worker results
3. DETECT with active model
     open vocab (yoloe): prompts = union of all active selectors' detect lists (re-set only when the union changes)
     fixed vocab (rfdetr): full vocab, filtered to requested labels
4. TRACK: supervision.ByteTrack over all detections
5. ATTRIBUTES: CLIP embedding per new track, refreshed at 1 Hz; score include/exclude texts; ref similarity
6. BEHAVIORS: each evaluates its selector → subjects, updates its state machine → events, render layers, actuator cmds
7. SKILLS: apply latest async results (keyboard homography, pose keypoints)
8. ACTUATOR: primary track/pan_to behavior → virtual motor → arrow layer
9. RENDER: compose layers + HUD → JPEG → /video
10. publish events (/ws/events) and state (/state)
```

Target: 25–30 fps at 720p input, 640 px detector size on the 4070.

## Models (`models.yaml`, preload all available at boot)

| Name | Role | Notes |
|---|---|---|
| `yoloe` | Default detector, open vocab, masks | `YOLOE("yoloe-11s-seg.pt")`; `set_classes(names, get_text_pe(names))` in the loop, only when the prompt union changes |
| `rfdetr` | Fixed-vocab detector (COCO) — demo #16 | `rfdetr` package, e.g. `RFDETRBase()`; `predict()` takes RGB (convert from BGR) and returns `sv.Detections`; map class IDs through the package's COCO class table (verify the indexing) |
| `pose` | Aux model for `pose_trigger` — demo #11 | `yolo11n-pose.pt`; runs only while a pose behavior is active |
| `clip` | Attributes + reference embeddings | open_clip ViT-B-32; DINOv2 is the upgrade if reference matching is weak |
| `ocr` | Aux model for the keyboard skill — demo #17 | EasyOCR, English, letter allowlist; worker thread only |

**Model switch** (`POST /model`): flip at a frame boundary (preloaded). Then re-validate every behavior. A behavior whose labels are missing from the new vocab goes `PAUSED` with a reason and resumes automatically when a compatible model returns. Emit `model_switched` with the new model and fps, and show the model name in the HUD.

## Specs

```jsonc
// Selector
{
  "detect": ["person"],                       // yoloe prompts or fixed-vocab labels
  "include": ["a person wearing a red shirt"],// CLIP; all must pass
  "exclude": ["a person wearing a black jacket"], // CLIP; any pass drops the track
  "ref_id": null,                             // appearance match to a registered reference
  "pick": "all"                               // all | largest | most_centered | ref
}

// Behavior
{
  "kind": "highlight|track|watch|count_line|privacy|pan_to|pose_trigger|keyboard",
  "subject": { /* Selector */ },
  "params": { /* per kind, below */ },
  "render": {"color": "#FFD400", "mask": true, "trail": false, "label": "duck"},
  "notify": true                              // events wake the agent
}
```

**CLIP scoring:** contrastive softmax over `[text, f"a {label}"]` × 100; pass at p ≥ 0.6 (tune on clips). Cache per `track_id`.
**Reference match:** cosine ≥ threshold *and* ≥ 0.05 above the second-best candidate.

## Behavior kinds

| Kind | Params | States | Events | Demo |
|---|---|---|---|---|
| `highlight` | — | ACTIVE, PAUSED | `count_changed` | 1, 5, 6, 8, 15 |
| `track` | `guidance: true` | ACQUIRING → TRACKING ⇄ EDGE → LOST → SEARCHING (10 s); PAUSED | `acquired`, `lost`, `reacquired` | 2, 5, 6, 7, 8 |
| `watch` | `triggers[]`, `cooldown_s: 5` | ARMING (subject stable 1 s, baseline learned) → ARMED → FIRED → COOLDOWN → ARMED; PAUSED | `armed`, `missing`, `moved`, `near`, `appeared` | 3, 9, 10 |
| `count_line` | `line: [[x1,y1],[x2,y2]]` normalized (default vertical center) | ACTIVE | `crossed` (in/out counts) | 13 |
| `privacy` | `keep_ref`, `mode: blur\|pixelate` | ACTIVE | — | 12 |
| `pan_to` | `deg`, `hfov_deg: 70` | GUIDING → REACHED | `reached` | 4 |
| `pose_trigger` | `gesture: hand_raised` | ACTIVE | `hand_raised` | 11 |
| `keyboard` | `text`, `step_mode: all\|auto\|manual`, `step_s: 0.8` | SEARCHING → LOCKED ⇄ SEARCHING | `keyboard_locked`, `keyboard_lost`, `step` | 17 |

Triggers for `watch`:
`{"type":"missing","after_s":2}`, `{"type":"moved","min_shift":0.15}`, `{"type":"near","other":Selector,"margin":0.1}`, `{"type":"appeared","other":Selector}`.
Triggers evaluate per subject track. "Authorized" (#10) is expressed as `exclude` on the `other` selector, the same mechanism as #3.

**Identity** (track LOST → reacquire): persons use `ref_id` when set; otherwise use the CLIP embedding of the last good crop (cosine > 0.8). Never let a random same-class object clear LOST.

## Virtual motor (guidance arrows)

`Actuator` interface: `command(rate_deg_s: float, reason: str)`. `VirtualMotor` renders arrows; a future `StepperMotor` sends to the camera service's `/motor`.

- Only the **primary** behavior drives the actuator: the most recently started `track` or `pan_to`.
- EDGE (`|cx| > 0.7`): faint arrow toward that edge.
- LOST: bold pulsing arrow toward the exit edge (from the last velocity), labeled "pan left" or "pan right". Hold until the same identity reacquires.
- SEARCHING (lost > 10 s): banner "target lost — searching".

## `pan_to` odometry (#4)

Grayscale, downscale to 320 px, `cv2.phaseCorrelate` (or median LK optical flow) between consecutive frames → dx px. `yaw += -dx / width * hfov_deg`. The arrow shows the remaining degrees; within 5° it shows "STOP" and emits `reached`. The agent waits for `reached`, then calls `count`.

## Keyboard skill (#17)

- **OCR worker** (2–4 Hz): EasyOCR on the frame; keep single-character A–Z/0–9 results with conf ≥ 0.5; normalize to lowercase; record each character's center.
- **Layout template** (QWERTY, key units, relative): number row `1234567890` y = -1, x = -0.5…8.5; `qwertyuiop` y = 0, x = 0…9; `asdfghjkl` y = 1, x = 0.25…8.25; `zxcvbnm` y = 2, x = 0.75…6.75; space ≈ (4.5, 3).
- **Fit:** `cv2.findHomography(template_pts, image_pts, RANSAC, ~0.5 key width)`. Needs ≥ 6 inliers; with 3–5, fall back to `estimateAffinePartial2D`. EMA-smooth the projected key centers between OCR updates.
- **Render:** a numbered badge on each key in typing order. Repeated keys list all their numbers (h → `1·7·14`, space → `␣ 5·9`). Draw a path line through the keys in order; in `auto` or `manual` mode, highlight the current step (`POST /behaviors/{id}/advance` for manual).
- `keyboard_lost` if no successful fit in 2 s.

## Renderer layers (in draw order)

privacy blur → masks → boxes + labels + track IDs → trails → zones/lines → keyboard badges/path → guidance arrows → alert flash border (1 s on trigger) → HUD (model, fps, HUD text, behavior chips with state).
Colors come from the behavior's `render.color` or a fixed palette by behavior index.

## API (FastAPI, bind 0.0.0.0:8001)

| Method | Path | Body / notes |
|---|---|---|
| GET | `/health` | status, fps, model, camera ok |
| GET | `/state` | behaviors (id, kind, state, spec summary), model, refs, fps |
| GET | `/models` | name, open_vocab, classes, loaded, available |
| POST | `/model` | `{name}` |
| POST | `/behaviors` | Behavior → `{id}`; 422 with reason on failure |
| DELETE | `/behaviors/{id}` and `/behaviors` | stop one or all |
| POST | `/behaviors/{id}/advance` | keyboard manual step |
| POST | `/references` | multipart image, or `{from: "largest_person"}` → `{ref_id, thumb_url}` |
| POST | `/query/count` | `{selector, window_s: 1}` → median count |
| POST | `/query/look` | `{selector?}` → detections with labels, attributes, positions |
| GET | `/snapshot` | current raw JPEG (for the orchestrator's vision model) |
| GET | `/snapshots/{id}.jpg` | event crops |
| POST | `/hud` | `{text}` shown in the HUD |
| GET | `/video` | enriched MJPEG (~15 fps) |
| WS | `/ws/events` | events |

Event:
```json
{"id":"e42","ts":0.0,"behavior_id":"b3","type":"near","detail":"person (track 17) near duck",
 "data":{},"snapshot_url":"/snapshots/e42.jpg","notify":true}
```
System events use `behavior_id: null`: `model_switched`, `camera_lost`, `camera_ok`.

## Config

`VIDEO_SOURCE` (URL | webcam index | file, files loop), `DEVICE`, `IMGSZ`, `PORT=8001`, `MODELS_CONFIG`, `CONF_THRESHOLD=0.25`, `LOG_LEVEL`.

## Layout

```
perception/
  main.py  config.py  models.yaml  contracts.py
  runtime/    capture.py loop.py ops.py workers.py events.py
  detectors/  base.py yoloe.py rfdetr.py
  attributes/ clip_cache.py references.py
  behaviors/  base.py highlight.py track.py watch.py count_line.py privacy.py pan_to.py pose_trigger.py
  skills/     keyboard.py pose.py
  actuator/   base.py virtual.py
  render/     layers.py hud.py
  server/     api.py stream.py
  clips/  (gitignored)   tests/
```

## Dependencies

`torch`, `ultralytics`, `supervision`, `open_clip_torch`, `rfdetr`, `easyocr`, `opencv-python`, `fastapi`, `uvicorn[standard]`, `python-multipart`, `pydantic>=2`, `pyyaml`, `numpy`.

## Build order

1. Loop, capture, YOLOE, ByteTrack, renderer, `/video`, `highlight` behavior, ops queue, `/behaviors`, events → demos 1, 5, 6, 8.
2. `track` + virtual motor arrows + identity on reacquire → 2 (partial).
3. CLIP attributes, include/exclude, references → 2, 3, 7, 10, 12, 15.
4. `watch` triggers → 3, 9, 10.
5. Registry: `rfdetr`, `pose` → 16, 11.
6. `pan_to`, `count_line`, `privacy` → 4, 13, 12.
7. Keyboard skill → 17.
Demo 14 lives entirely in the orchestrator.

## Working style

- Record clips of each demo scene early and test every behavior against them on the Mac.
- Log per-stage timings every 100 frames. Log every op, state transition, and event.
- Say plainly when something can only be verified on the 4070.
- No auth, persistence, Docker, or ROS.
