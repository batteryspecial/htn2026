# perception

Real-time perception service for the robot car. It reads a video stream, finds the target described by the active `TaskSpec`, and publishes the target's position on every frame. New specs and new models are hot-swapped without restarting.

## What it does

1. Pulls the latest frame from `VIDEO_SOURCE` (car camera, phone, webcam, or a looping file).
2. Detects objects: YOLOE for open-vocabulary text prompts, or a fixed-vocab model (YOLO11, RF-DETR, TensorRT builds) from the registry.
3. Tracks with ByteTrack, filters by attribute (CLIP) and containment ("shoe inside person"), and selects one target.
4. Publishes a `TargetState` (target offset, size, and pipeline `phase`) for the controller, plus an annotated MJPEG for the UI.
5. Accepts new specs on `POST /spec`, prepares them in the background, and swaps them in between frames. Progress goes out as status events.

The controller drives only when `phase == "TRACKING"`.

## Endpoints (port 8001)

| Method | Path | Purpose |
|---|---|---|
| POST | `/spec` | Apply a new TaskSpec `{instruction_id, spec}` |
| DELETE | `/spec` | Stop tracking (IDLE) |
| GET | `/models` | Model registry and vocabularies |
| GET | `/health` | Phase, fps, active model and spec |
| WS | `/ws/target` | TargetState per frame |
| WS | `/ws/events` | Status events |
| GET | `/video` | Annotated MJPEG |

## Run

Run everything from inside `perception/`.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# GPU host
VIDEO_SOURCE=http://<car-ip>/stream .venv/bin/uvicorn main:app --host 0.0.0.0 --port 8001

# Mac (logic testing only; YOLOE runs on CPU here, see below)
DEVICE=cpu IMGSZ=320 VIDEO_SOURCE=clips/demo.mp4 .venv/bin/uvicorn main:app --port 8001
```

Open `http://localhost:8001/` for a dev page with the feed, live target state and
a box to post specs. The operator UI is Kevin's, in `frontend/`.

```bash
pytest                                   # 141 tests, no weights or GPU needed
PERCEPTION_TEST_WEIGHTS=1 DEVICE=cpu IMGSZ=320 pytest tests/test_detectors_real.py
python scripts/smoke_test.py             # whole pipeline against a generated clip
python scripts/e2e_check.py              # drive a running service over HTTP + WS
```

Try a spec:

```bash
curl -X POST localhost:8001/spec -H 'content-type: application/json' -d '{
  "instruction_id": "test1",
  "spec": {"spec_id": "s1", "model": "yoloe", "mode": "center",
           "targets": [{"ref": "t1", "detect": ["pencil"], "select": "largest"}]}}'
```

## Config

`VIDEO_SOURCE`, `DEVICE` (auto/cuda/mps/cpu), `IMGSZ`, `PORT`, `MODELS_CONFIG`, `ACQUIRE_TIMEOUT_S`, `CONF_THRESHOLD`, `LOG_LEVEL`. Models are listed in `models.yaml`. TensorRT `.engine` files are built on the GPU host and never committed.

## Notes for the GPU host

- Weights (~620 MB) download into `weights/` on first boot, including a 572 MB
  MobileCLIP text encoder that YOLOE pulls silently. Do that once on good WiFi.
- `ultralytics` and `supervision` are pinned. The predict signature changed
  between versions; don't float them.
- On a Mac, YOLOE is forced to CPU: its text encoder needs float64 and MPS has
  none. CUDA is unaffected.
- A model that can't run on the current machine reports `available: false` from
  `GET /models` and the service still boots.
