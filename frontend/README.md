# Retask Console

React + TypeScript + Vite operator console for the camera pipeline.

```powershell
cd frontend
npm.cmd install
npm.cmd run dev          # http://localhost:3000
npm.cmd run build        # frontend/dist, also served by orchestrator on :8000
```

## Live integration

- Chat sends multipart `POST /chat` to the orchestrator (:8000), with `text`,
  repeated `images` fields and a browser-generated `turn` id.
- The orchestrator registers uploaded references, runs its tool loop, and
  installs/removes behaviors through perception (:8001). The browser does
  not install a second copy of the agent's behaviors.
- `WS /ws/trace` streams tool calls/results, spoken alerts and final replies.
  Progress is matched by turn id. HTTP and socket replies are deduplicated;
  reconnect history is displayed without speaking old alerts again.
- The timer measures the agent turn. Completion is separate from acquiring a
  target; actual tracking/match state appears in the pipeline status and
  behaviors panel.
- HTTP/trace behavior receipts prove what the turn compiled. The stage strip
  marks `APPLIED` only after those ids appear in `/ws/state`, and marks `ACTIVE`
  only after their state or match count shows target acquisition. A
  conversational answer with no receipt is shown as `REPLY`.
- STOP calls `POST /stop`, cancelling active/queued agent turns before clearing
  behaviors and the HUD. A direct perception clear is attempted if the agent
  service cannot be reached, with cancellation uncertainty shown in chat.
- Video (`/video`), state (`/ws/state`), events (`/ws/events`), model selection
  and individual behavior removal communicate directly with perception.

`LIVE` uses the agent. `MOCK` uses local keyword rules and sends the resulting
program directly to perception; it still requires the local perception service.
The frontend has no model API key and makes no model API calls.

## Configuration

`frontend/.env.local` contains service URLs only:

```dotenv
VITE_ORCHESTRATOR_URL=http://localhost:8000
VITE_PERCEPTION_URL=http://localhost:8001
# VITE_VIDEO_URL=http://localhost:8001/video
```

Query overrides: `?api=...&pipeline=...`, `?live=1`, `?mock=1`.
Voice/language/mode preferences are stored in localStorage. Optional Vite proxy
paths are configured by `PROXY_ORCHESTRATOR` and `PROXY_PERCEPTION`; set the
corresponding VITE URL to `/orchestrator` or `/perception` to use them.

The orchestrator reads its own API key from `orchestrator/.env`. Its default
turn deadline is 180 seconds; the browser allows 195 seconds. If you increase
`ORCH_TURN_TIMEOUT_S`, increase `REQUEST_TIMEOUT_MS` in the frontend too.

## Structure

- `components/`: camera, chat, trace, events, behaviors, settings and history.
- `hooks/useAgentRun.ts`: live turn lifecycle, replies, progress and cancellation.
- `hooks/useRetaskRun.ts`: local mock compilation, validation and dispatch.
- `hooks/useTrace.ts`, `hooks/usePerception.ts`: live subscriptions.
- `services/`: HTTP clients and reconnecting sockets.
- `contracts/`: TypeScript mirror of `linker/schemas.py` plus mock validation.
- `config/`, `styles/`, `utils/`: deployment settings, styles and small helpers.

The available detector models and behavior kinds come from the running pipeline.
The root README describes the broader design; unsupported capabilities such as
keyboard guidance remain pipeline work, not a browser feature.
