# CLAUDE.md — perception (owner: Qinkai)

You are Qinkai's coding agent. You work only inside `perception/`. Read this whole file before writing code.

## Project

Hack the North 2026, 36-hour hackathon, team of 3. The goal is an impressive live demo, not a production system. Prefer the simplest thing that works on stage.

**Product:** natural-language retasking of a robot car's perception. An operator types "follow the person in red shoes." An LLM compiles the instruction into a JSON `TaskSpec`. This pipeline hot-swaps to that spec without restarting, and the car follows the new target. The headline metric is retask time: instruction received → target acquired.

## Repo layout and ownership

| Folder | Owner | Purpose |
|---|---|---|
| `perception/` | **Qinkai (you)** | Detection, tracking, target selection, hot-swap state machine. Port 8001 |
| `orchestrator/` | Danny | Instruction → LLM → TaskSpec → `POST /spec` here; relays status to UI. Port 8000 |
| `controller/` | Kevin | Reads `TargetState` from here, drives the car. Port 8002 |
| `frontend/` | Kevin | Operator UI: chat, `/video` feed, status board, E-STOP |
| `linker/` | all three | Shared Pydantic contracts |

Never edit files outside `perception/`. Never change `linker/` without an explicit instruction from Qinkai, because the change needs team agreement. If a needed contract field isn't in `linker/` yet, mirror it in `perception/contracts.py` and flag it in your reply.

## Hardware reality (important)

- **You run on a MacBook.** No CUDA, no TensorRT. You can run code on CPU or MPS only.
- **Inference runs on a different machine:** a friend's PC with an **RTX 4070 (12 GB, Ada, CUDA)**, probably in WSL2 Ubuntu. Qinkai deploys there with `git pull` and a restart.
- Therefore:
  - All device selection goes through `config.DEVICE` (`auto` → cuda → mps → cpu). Never hardcode `"cuda"`.
  - GPU-only paths (TensorRT engines, `half=True`) must be guarded and must degrade gracefully. Registry entries whose backend is unavailable are marked `available: false`, and the process still boots.
  - `.engine` files are specific to one GPU architecture and TensorRT version. They are built on the 4070 by `scripts/build_engines.py`. Never build or commit them from the Mac. `.engine` and `.pt` are gitignored.
  - On the Mac, set `PYTORCH_ENABLE_MPS_FALLBACK=1` and use `IMGSZ=320` for speed.
  - Use `pathlib` everywhere. The host may be Windows or WSL.
- Every piece of logic must be testable on the Mac against a recorded clip (`VIDEO_SOURCE=clips/x.mp4`, looping).

## Critical design decision

YOLOE exported to TensorRT bakes in its prompt vocabulary. That kills open-vocab swapping. So:
- **YOLOE runs in PyTorch** (fp16 on CUDA). Open vocab, swapped with `set_classes`.
- **Fixed-vocab models** (YOLO11-COCO, RF-DETR) may run as TensorRT engines.

## Architecture

```
capture thread ──latest frame only──▶ inference loop ──▶ /ws/target (TargetState) + /video (MJPEG)
                                          ▲
loader worker (1 thread) ──PreparedTask──▶ pending slot ──swapped at top of next frame
FastAPI (asyncio) ── /spec, /models, /health, /ws/events
```

Rules:
1. **Single writer.** Only the inference loop mutates the active task, detector state, and phase. Other threads communicate through the pending slot and a thread-safe event queue.
2. **Never block the inference loop.** Model loading, text embedding, and CLIP text encoding happen in the loader worker. Exception: YOLOE `set_classes(names, text_pe)` mutates the model, so it runs in the loop at swap time. It's cheap because `text_pe` was precomputed by the worker via `model.get_text_pe(names)`.
3. **Latest frame only.** The capture thread overwrites a single slot. Set `CAP_PROP_BUFFERSIZE=1` on network streams.
4. **Atomic swap.** The old task stays active until a `PreparedTask` is complete. A failed prepare is a no-op: emit `failed`, revert the phase, keep running the old task.
5. **Newest spec wins.** A new spec during prepare replaces the pending job. The superseded instruction gets `failed` with detail `superseded`.
6. **Events vs phase.** Events are edge-triggered and meant for humans (via the orchestrator). Phase is level-triggered and meant for the controller (on every `TargetState`).

## Per-frame stages

1. **Detect.** The active detector returns `supervision.Detections` with class names in `data["class_name"]`. With two targets, detect on the union of both prompt lists.
2. **Track.** `supervision.ByteTrack`, a fresh instance per `PreparedTask`.
3. **Verify.** open_clip ViT-B-32. Contrastive: softmax over [`verify.text`, f"a {cls}"] × 100; keep if p ≥ `verify.min_score` (default 0.6 in practice). Cache per `track_id`, re-verify every 1 s. Batch crops, cap at 16.
4. **Relate.** Keep tracks of `relate.keep` that contain a (verified) `if_contains` box whose center lies in the keep-box's lower 40%.
5. **Select.** `largest` | `most_centered` | `highest_conf` | `locked`. `locked` picks via `largest` first, then stores a CLIP image embedding. It prefers the same `track_id`; after ID loss, it adopts a candidate with cosine > 0.8. The embedding updates by EMA (α = 0.1) only when conf > 0.6.
6. **Arbitrate.** An active-target pointer flips every `arbitration.alternate_s`. It flips early if the active target has been missing > 1 s while the other is visible.
7. **Publish.** One `TargetState` per frame. Annotated JPEG (supervision annotators, overlay shows phase + spec summary) to `/video` at ~15 fps.

## Detector interface

```python
class Detector(Protocol):
    name: str
    open_vocab: bool
    classes: list[str] | None          # None for open-vocab
    def prepare(self, prompts: list[str]) -> Any: ...   # worker thread; raises on unknown class
    def apply(self, prepared: Any) -> None: ...         # loop thread; cheap
    def infer(self, frame: np.ndarray) -> sv.Detections: ...
    def warmup(self) -> None: ...
```

Implementations:
- `yoloe`: `YOLOE("yoloe-11s-seg.pt")`. `prepare` → `get_text_pe(names)`; `apply` → `set_classes(names, pe)`. Ignore masks.
- `ultralytics_fixed`: any `YOLO(...)` `.pt` or `.engine` with a fixed class list. `prepare` → map names to IDs (raise on unknown); filter at inference.
- `rfdetr` (stretch): `rfdetr` package, returns `sv.Detections` natively.

The registry is configured in `models.yaml` (name, type, weights, open_vocab, preload, requires_cuda). Preload everything available at boot. Cold load (load + 3 warmups in the worker) is a fallback path.

## State machine

Phases: `BOOTING, IDLE, SWITCHING, LOADING_MODEL, ACQUIRING, TRACKING, LOST, NO_TARGET, FAULT`.
The controller drives **only** in `TRACKING` with `visible=true`.

| From | Trigger | To | Event |
|---|---|---|---|
| BOOTING | registry ready | IDLE | — |
| BOOTING | yoloe fails to load | FAULT | — |
| any running | valid spec received | SWITCHING | — |
| SWITCHING | spec model not loaded | LOADING_MODEL | `model_loading` |
| LOADING_MODEL | load + warmup ok | SWITCHING | `model_loaded` |
| SWITCHING / LOADING_MODEL | prepare/load error | previous phase | `failed(reason)` |
| SWITCHING | worker done | SWITCHING | `prepared` |
| SWITCHING | loop swaps pending task | ACQUIRING | `applied` |
| ACQUIRING / NO_TARGET | target seen 3 consecutive frames | TRACKING | `active` (once per spec) |
| ACQUIRING | 3 s timeout | NO_TARGET | `no_target` |
| TRACKING | target missing 5 frames | LOST | — |
| LOST | reacquired | TRACKING | — |
| any | camera dead > 2 s or 3 consecutive inference errors | FAULT | `failed(detail)` |
| FAULT | recovered | ACQUIRING (spec) / IDLE | — |
| any running | `DELETE /spec` | IDLE | — |

In FAULT, publish heartbeat `TargetState` at 5 Hz with `visible=false`. Implement the machine as a plain enum plus one `transition(new, event=None)` method that logs every transition. No state-machine library.

## Contracts (from `linker/`; fields marked NEW are proposed, additive)

```python
class TaskSpec:        # spec_id, targets[1..2], mode "follow"|"center", arbitration, model: str (NEW: was Literal["yoloe"])
class Target:          # ref, detect[1..6], verify?{class,text,min_score}, relate?{keep,if_contains}, select
class TargetState:     # ts, spec_id, mode "follow"|"center"|"idle", visible, cx, cy, area, conf, label, track_id
                       # NEW: phase, model, target_ref
class StatusEvent:     # instruction_id, spec_id, stage, ts, detail, data
                       # stages from perception: prepared, applied, active, no_target, failed
                       # NEW: model_loading, model_loaded
```

`cx`, `cy` are in [-1, 1] as offsets from frame center (right and down positive). `area` is bbox area / frame area. `ts` is `time.time()`.

## Endpoints (FastAPI, bind 0.0.0.0:8001)

- `POST /spec` `{instruction_id, spec}` → 202; rejects with 422 + `failed` event on schema or vocabulary error
- `DELETE /spec` → IDLE
- `GET /models` → registry: name, open_vocab, classes, loaded, available (NEW; the orchestrator builds its manifest from it)
- `GET /health` → phase, fps, model, spec_id, last_frame_age_ms
- `WS /ws/target` → TargetState every frame
- `WS /ws/events` → StatusEvent stream
- `GET /video` → annotated MJPEG

## Config (env vars, `config.py`)

`VIDEO_SOURCE` (webcam index | file path, loops | URL), `DEVICE` (auto), `IMGSZ` (640; 320 on Mac), `PORT` (8001), `MODELS_CONFIG` (models.yaml), `ACQUIRE_TIMEOUT_S` (3), `CONF_THRESHOLD` (0.25), `LOG_LEVEL`.

## Suggested layout

```
perception/
  main.py            # uvicorn entry, wires threads
  config.py  contracts.py  models.yaml
  runtime/   capture.py  loop.py  loader.py  state.py  events.py  task.py
  detectors/ base.py  yoloe.py  ultralytics_fixed.py  rfdetr.py
  stages/    verify.py  relate.py  select.py  arbitrate.py
  server/    api.py  stream.py
  scripts/   build_engines.py  smoke_test.py  send_spec.sh
  clips/     (gitignored test videos)
```

## Dependencies

`torch` (CUDA 12.x build on host, default build on Mac), `ultralytics`, `supervision`, `open_clip_torch`, `fastapi`, `uvicorn[standard]`, `opencv-python`, `pydantic>=2`, `pyyaml`, `numpy`. Host only: `tensorrt`. Optional: `rfdetr`.

## Working style

- Build the thinnest end-to-end path first: capture → YOLOE → ByteTrack → `largest` → `/ws/target` + `/video`. Then hot-swap, then registry, then verify/relate, then `locked`.
- Log per-stage timings every 100 frames. Log every phase transition and every event.
- Qinkai is vibe coding under time pressure. Keep modules small, avoid abstractions beyond the Detector protocol, and say plainly when something can only be verified on the 4070.
- Don't add auth, persistence, Docker, or ROS.
- Build tests for the most critical parts, since correctness is critical.

## Stretch: Cutie for `locked`
Only after baseline `locked` (ByteTrack + CLIP) fails the two-people-crossing test.
Seed Cutie with the YOLOE-seg mask of the selected track; Cutie mask → bbox → TargetState.
YOLOE stays on as a check (IoU > 0.3 with a same-class detection); mask empty 5 frames → LOST;
re-seed via CLIP re-ID. Fresh processor per PreparedTask. Cap long-term memory. 4070 only.
