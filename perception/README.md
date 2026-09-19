# perception

The reflex layer. It reads webcam frames and runs whatever behaviors the agent has installed, every frame: detect, track, check attributes, evaluate triggers. It draws the result onto the frame and streams it to the frontend. Events go out to the orchestrator.

## What it does

- **Detects** with the active model: YOLOE (open vocabulary, any text prompt) or RF-DETR (COCO). Models switch live.
- **Tracks** objects across frames (ByteTrack) and checks attributes like "red shirt" or "black jacket" with CLIP.
- **Matches references:** register a photo of a person, then pick that person out of a crowd.
- **Runs behaviors:** `highlight`, `track`, `watch`, `count_line`, `privacy`, `pan_to`, `pose_trigger`, `keyboard`.
- **Guides the operator:** when a tracked target leaves the frame, arrows tell a person which way to pan the camera.
- **Renders** masks, boxes, trails, zones, key badges, arrows, alerts, and a HUD onto an MJPEG stream.
- **Emits events** (`acquired`, `lost`, `near`, `missing`, `crossed`, `reached`, ...) with snapshot crops.

## API (port 8001)

| Method | Path | Purpose |
|---|---|---|
| POST | `/behaviors` | Start a behavior; returns `{id}` or 422 with a reason |
| DELETE | `/behaviors/{id}`, `/behaviors` | Stop one or all |
| POST | `/model` | Switch detector (`yoloe`, `rfdetr`) |
| POST | `/references` | Register a person image, returns `ref_id` |
| POST | `/query/count`, `/query/look` | One-shot questions |
| GET | `/snapshot` | Current raw frame |
| GET | `/state`, `/models`, `/health` | Introspection |
| GET | `/video` | Enriched MJPEG |
| WS | `/ws/events` | Event stream |

## Run

```bash
pip install -r requirements.txt

# GPU host (RTX 4070)
VIDEO_SOURCE=http://localhost:8003/stream uvicorn perception.main:app --host 0.0.0.0 --port 8001

# Mac (logic testing on a recorded clip)
PYTORCH_ENABLE_MPS_FALLBACK=1 IMGSZ=320 VIDEO_SOURCE=clips/duck.mp4 uvicorn perception.main:app --port 8001
```

Try it:

```bash
curl -X POST localhost:8001/behaviors -H 'content-type: application/json' \
  -d '{"kind":"highlight","subject":{"detect":["person"],"pick":"all"},"render":{"mask":true}}'
```

Then open `http://localhost:8001/video`.
