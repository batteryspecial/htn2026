# Hack the North 2026

## Pitch

Physical AI is limited by how its hardware was programmed, not by the hardware itself. One webcam, driven by agents, becomes a people counter, a follow-cam, a security guard, a privacy camera, a traffic counter, or a typing guide. Every switch happens from one sentence, with no retraining.

## Components

| Component | Owner | Port | What it does |
|---|---|---|---|
| Camera service | Teammate | 8003 | Reads the webcam and serves raw frames. Later drives the stepper via Arduino |
| Perception (reflex layer) | Qinkai | 8001 | Runs every frame: detect, track, check attributes, run behaviors, draw overlays, emit events |
| Orchestrator (agent layer) | Teammate | 8000 | LLM agent with tools. Turns instructions into behaviors and queries; reacts to events |
| Frontend | Teammate | served by 8000 | Enriched live feed, chat with image upload, agent trace, events, behaviors, switch timer |

## Two speeds

- **Agents think in seconds.** They start and stop behaviors, ask questions (count, look, describe), and react to events.
- **The pipeline acts every frame.** Tracking, guidance arrows, trigger checks, and drawing never wait for an agent.
- Agents configure the pipeline; they never sit inside its loop.

## Full design

```
 ┌───────────────────────────────── FRONTEND (single page, browser) ─────────────────────────────────┐
 │  ┌────────────────────────────┐  ┌──────────────────────┐  ┌──────────────┐  ┌─────────────────┐  │
 │  │ ENRICHED LIVE FEED  <img>  │  │ CHAT + IMAGE UPLOAD  │  │ AGENT TRACE  │  │ EVENTS / ALERTS │  │
 │  │ masks·boxes·arrows·HUD     │  │ switch timer         │  │ tool calls   │  │ behaviors panel │  │
 │  └─────────────▲──────────────┘  └──────────┬───────────┘  └──────▲───────┘  └────────▲────────┘  │
 └────────────────┼────────────────────────────┼─────────────────────┼───────────────────┼───────────┘
     GET /video   │ (MJPEG)       POST /chat   │ (text + images)     │  WS /ws/trace     │
                  │                            ▼                     │ (trace, say, timer, events)
                  │     ┌──────────────── ORCHESTRATOR :8000 — agent layer (seconds) ─────────────────┐
                  │     │                                                                             │
                  │     │  user turn ──▶ ┌──────────────────────────────┐ ──▶ reply / say()           │
                  │     │                │ AGENT LOOP                   │                             │
                  │     │  event turn ─▶ │ LLM ⇄ tools (max 10 steps)   │ ──▶ describe() ──▶ vision   │
                  │     │   ▲            └──────────────┬───────────────┘       model on snapshot     │
                  │     │   │ rate-limited queue        │ tool calls = HTTP                           │
                  │     │   │ (1 per behavior / 5 s)    │ attachments store (uploads)                 │
                  │     └───┼───────────────────────────┼─────────────────────────────────────────────┘
                  │         │ WS /ws/events             │ /behaviors /references /query /model /snapshot
                  │         │                           ▼
 ┌────────────────┴─────────┴──────────── PERCEPTION :8001 — reflex layer (every frame) ────────────────┐
 │                                                                                                      │
 │  frame ─▶ apply ops queue ─▶ DETECT ─────────▶ TRACK ─────▶ ATTRIBUTES ────────▶ BEHAVIORS          │
 │  (latest)  (single writer)    active model:     ByteTrack    CLIP cache per       highlight · track  │
 │                               yoloe (open)                   track: include /     watch · count_line │
 │                               rfdetr (COCO)                  exclude / ref match  privacy · pan_to   │
 │                                                                                   pose_trigger       │
 │                                  SKILLS (async aux models) ─▶ keyboard (OCR + homography) · pose      │
 │                                                                                                      │
 │  BEHAVIORS ─▶ ACTUATOR: virtual motor = guidance arrows   (later: stepper via camera service)        │
 │            ─▶ RENDERER: blur · masks · boxes · trails · zones · key badges · arrows · alerts · HUD ──▶ /video
 │            ─▶ EVENTS: acquired · lost · missing · moved · near · crossed · hand_raised · reached ──▶ /ws/events
 └──────────────────────────────────────────────▲───────────────────────────────────────────────────────┘
                                                │ VIDEO_SOURCE = http://localhost:8003/stream (or webcam index)
                         ┌──────────── CAMERA SERVICE :8003 ─────────────┐
                         │ webcam ─▶ latest frame ─▶ /stream  /frame     │
                         │ (later) POST /motor ─▶ serial ─▶ Arduino ─▶ stepper
                         └───────────────────────────────────────────────┘
```

## How one instruction flows

"Watch over this yellow duck but ignore anybody wearing a black jacket":

1. The frontend sends the text to the orchestrator and starts the switch timer.
2. The agent calls `start_behavior` with kind `watch`, subject "yellow duck", and a `near` trigger for any person who does *not* match "a person wearing a black jacket".
3. Perception validates the spec, applies it between two frames, learns the duck's position (`armed`), and draws its mask.
4. A person without a black jacket approaches. Perception fires a `near` event with a snapshot crop.
5. The orchestrator starts an event turn. The agent looks at the snapshot and calls `say("Someone is approaching the duck")`. The frontend shows an alert and speaks it.

## Demo list (source of truth)

Perception column: 🎬 proven on recorded footage · ✅ built and unit-tested,
waiting on a clip · ⏳ not built.

| # | Instruction | Device it becomes | Perception | What the agent must do |
|---|---|---|---|---|
| 1 | Detect all humans | People counter | 🎬 | one `highlight` |
| 2 | Upload photo + "track that human" | Follow-cam (guidance arrows) | ✅ | `POST /references`, then `track` with `pick: ref` |
| 3 | Watch the yellow duck, ignore black jackets | Sentinel | 🎬 | `watch` + a `near` trigger with `exclude` |
| 4 | Turn 45° left and count people | Surveyor (arrow guides the pan, odometry measures it) | ✅ | `pan_to`, wait for `reached`, then `/query/count` |
| 5 | Track the pencil → now the eraser | Instant retarget | 🎬 | replace the behaviour |
| 6 | Track the pencil and the eraser | Multi-target tracker | 🎬 | two behaviours; colours are assigned apart |
| 7 | Follow the person in red shoes | Attribute + relation reasoning | ✅ | `relate` + attribute pair, phrased per SELECTORS.md |
| 8 | Now follow the dog | New class, no retraining | 🎬 | replace the behaviour |
| 9 | Guard the table: laptop, phone, wallet | Incident logger | ✅ | three `watch` behaviours, `missing` triggers |
| 10 | Anyone in a red hoodie is authorized | Access control | ✅ | `exclude` on the trigger's own selector |
| 11 | Tell me when someone raises their hand | Gesture trigger (stretch) | ✅ | `pose_trigger`; the pose model downloads on first use |
| 12 | Blur everyone's face except mine | Privacy camera | ✅ | `POST /references`, then `privacy` with `keep_ref` |
| 13 | Count people crossing this line | Traffic counter | 🎬 | `count_line` with a normalized line |
| 14 | What's on the table right now? | Scene Q&A | ✅ | `POST /describe`, then its own vision model |
| 15 | Highlight anything red | Attribute search | ✅ | `highlight` + attribute pair |
| 16 | Switch your model to RF-DETR | Model swap | ✅ | `POST /model`; works today with `coco` |
| 17 | Hover over a keyboard, "type hack the north" | Typing guide (showpiece) | ⏳ | needs OCR + a homography fit |

**Fifteen of the sixteen perception items are built.** Seven are proven on
recorded clips; the rest are unit-tested and waiting on footage. Only #17
remains, and it is the one item that is a genuine skill rather than a
combination of existing primitives.

The model zoo landed, so adding a model is now an entry in
`perception/models.yaml` rather than a code change. #16 already demonstrates
model swapping with the COCO detector; RF-DETR would be one more entry if it
installs cleanly.

## Backups

- **4070 unavailable:** RunPod pod plus the webcam streamed from a laptop (more lag), or the Mac on MPS at 320 px.
- **An item is flaky in rehearsal:** cut it. Twelve reliable items beat seventeen shaky ones.
- **CLIP attributes weak:** put attribute words directly into YOLOE prompts ("black jacket").
- **OCR too slow for #17:** run it at 1 Hz and hold the camera still.
- **RF-DETR won't install:** swap in a YOLO11 COCO model for #16.
- **Offline venue:** pre-download every weight file. Record a clean full run as a backup video.