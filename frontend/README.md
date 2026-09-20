# Retask Console

Operator front end. An instruction — spoken or typed — becomes a **program of
behaviours**, which is posted to perception and drawn on the live feed.

React + TypeScript + Vite.

```bash
cd frontend
npm install
npm run dev          # http://localhost:3000
```

## What it talks to

| Service | Port | What this app uses |
| --- | --- | --- |
| Perception | 8001 | `/behaviors` · `/models` · `/model` · `/health` · `/state` · `/hud` · `/video` · `/snapshots` · `WS /ws/state` · `WS /ws/events` |
| Orchestrator | 8000 | `/instruction` · `WS /ws/status` today; `/chat` · `WS /ws/trace` once the agent loop exists |

Nothing is hardcoded. Settings live in localStorage, and a preconfigured link
works too: `?api=http://10.0.0.5:8000`, `?pipeline=http://10.0.0.7:8001`,
`?mock=1`.

## Modes

The pill in the top bar cycles **OPENAI → LIVE → MOCK**.

| Mode | What it does |
| --- | --- |
| `OPENAI` | Compiles in the browser against `api.openai.com`. Standalone — needs no orchestrator and no teammates. |
| `LIVE` | `POST {apiBase}/instruction`, listens on `{apiBase}/ws/status`. The orchestrator. |
| `MOCK` | In-page keyword rules. No network, no internet. The fallback if the venue WiFi or a teammate's laptop dies mid-demo. |

`OPENAI` mode is a **bridge, not the design.** In `README.md` the orchestrator's
agent loop compiles, over several steps with tools, which is strictly more
capable: it can upload a reference first, wait for an event, rephrase when
nothing matches, and answer a question instead of installing anything. Four
demo items need that loop and cannot be done in one structured-output call
(#2, #4, #12, #14). Delete `src/services/compilers/` when the loop lands.

## The contract

One instruction compiles to a **program**: one or more `BehaviorSpec`s that
should be running afterwards.

```jsonc
{
  "kind": "watch",
  "subject": {"detect": ["yellow duck", "duck"], "pick": "largest"},
  "params": {"triggers": [{
    "type": "near",
    "other": {"detect": ["person"],
              "include": ["a person in light coloured clothing"],
              "exclude": ["a person wearing a black jacket"]}
  }]},
  "render": {"label": "the duck"}
}
```

Applying a program is `DELETE /behaviors` then one `POST /behaviors` per
behaviour, in order — an instruction means "this is the whole objective now".
That is `PerceptionClient.applyProgram`.

`src/contracts/behavior.ts` is a hand-maintained mirror of `linker/schemas.py`,
which is the source of truth and is owned jointly. Change it only to follow.

### Why the TaskSpec shim is gone

The previous contract had `targets[]` with `verify` / `relate` / `select`, and
perception translated it in `server/legacy.py`. That adapter mapped every
target to a `track` behaviour, which left `watch`, `count_line`, `privacy`,
`pan_to` and `pose_trigger` unreachable from the only UI — demo items 3, 4, 9,
10, 11, 12, 13 and 14, eight of seventeen, behind one adapter.

Both sides speak `BehaviorSpec` now. `POST /spec` and `WS /ws/status` were
removed from perception along with `server/legacy.py`; `tests/test_integration.py`
pins that they stay gone.

### Validation

`src/contracts/validateProgram.ts` mirrors both layers the server applies: the
pydantic contract (`extra="forbid"`, bounds) and the per-kind `params` check
`_params_error` performs by building each behaviour and throwing it away. An
invalid program is never posted, because the pipeline would 422 it.

Warnings are things the pipeline accepts but that will not do what was asked —
a `highlight` with `pick: "largest"`, a `track` with `pick: "all"`, a
`keep_ref` with no `ref_id`. They never block a post.

Available kinds come from `GET /behaviors` rather than being hardcoded, so a
kind landing later needs no frontend change.

## Structure

```
src/
  contracts/     the wire format, mirrored from linker/schemas.py
    behavior.ts      BehaviorSpec, Selector, Event, StateView, Health
    program.ts       what one instruction compiles to; summaries, key order
    validateProgram.ts
    format.ts        JSON tokenising for the viewer
  services/      everything that does I/O
    perception.ts    the :8001 client
    orchestrator.ts  the :8000 client
    socket.ts        reconnecting WebSocket
    http.ts          fetch with a timeout and readable errors
    compilers/       openai · mock · live, behind one interface
  hooks/         state and effects, one concern each
    useRetaskRun.ts  compile -> validate -> apply -> settle
    usePerception.ts live state, events, health, models, kinds
    useVideoStream.ts
    useSpeechRecognition.ts · useAudioLevel.ts · useSpeech.ts
    useSettings.tsx · useHistory.ts · useHotkeys.ts
  components/    presentation, grouped by panel
  config/        settings and fixed vocabulary
  styles/        TTC Dupont theme, one file per concern
```

## The API key

`.env.local` holds `VITE_OPENAI_API_KEY` and **is gitignored**. Copy
`.env.example` and paste the key in.

Vite inlines it at build time, so anyone who can load the page can read it out
of the bundle. On localhost that is only you. **Do not deploy this build
anywhere public.** The right long-term shape is compiling on the orchestrator,
where the key stays server-side.

Rotate the key at <https://platform.openai.com/api-keys> when the hackathon
ends.

### Model notes

Default is `gpt-6-astra`, chosen for headroom on phrasing nobody rehearsed.
`gpt-5.4-mini` is the fast fallback.

**Models disagree about request parameters.** `gpt-6-astra` and `gpt-5.6-terra`
reject `temperature: 0` outright; `gpt-5.4-mini` and `gpt-4o-mini` accept it.
The adapter sends it, catches that specific 400, retries without it, and
records the model in `localStorage`, so the discovery round trip happens once
ever rather than once per compile.

Avoid `gpt-5.6-luna`: an 11.7 s outlier when it spent 1024 reasoning tokens on
a three-way choice. `requestTimeoutMs` is 45 s for that reason.

On the older TaskSpec prompt, **every accuracy gap measured was a prompt gap,
not a model gap** — twice. Tune `src/services/compilers/prompt.ts` and re-run
before reaching for a bigger model.

## Controls

| Key | Action |
| --- | --- |
| `Enter` | compile |
| `Shift`+`Enter` | newline |
| `Ctrl`+`M` | start / stop the mic |
| `Esc` | cancel listening, close settings |
| `V` | toggle spoken confirmations |

**Spacebar is deliberately unbound.** It was reserved for E-STOP. The car is
out of scope now, so it is free — but leave it free unless something genuinely
needs a panic key, because space is what people hit when they mean "stop".

STOP calls `DELETE /behaviors`. It used to call `DELETE /spec`, which never
existed and returned 405 every time.

## What the screen shows

- **Timer** — instruction sent → the pipeline's `acquired` event. The retask
  metric. Compile time is the small line underneath. Kinds that have nothing to
  acquire (`highlight`, `privacy`, `count_line`) settle on `applied` instead of
  waiting 25 s for an event that never comes.
- **Stage strip** — `received · compiled` (ours) then `applied · active`. Only
  three pipeline events have a stage equivalent; everything else in the event
  vocabulary goes to the events panel in full rather than being flattened into
  a name that does not fit.
- **Behaviours panel** — live off `WS /ws/state`. What is actually running,
  with its state and match count. **A behaviour that is `ACTIVE` with zero
  matches is not working** — check the count, not the state.
- **Events panel** — `WS /ws/events`, with the snapshot crop each event fired
  on.
- **History** — last 8 runs, click to recall.

## The camera pane

A plain `<img>` pointed at the pipeline's MJPEG. The pipeline draws the boxes;
this app draws nothing over them.

**Why there is a host probe.** An MJPEG `<img>` dies silently: when the server
goes away mid-stream it fires no `error`, no `load`, and the last frame sits
there looking perfectly live. Measured — 14 s after killing the stream the pane
still said LIVE. A per-frame heartbeat does not help either, because Chrome
does not emit a load event per part of a multipart stream.

So every 4 s the page fetches the stream's *origin* with `mode: no-cors`.
Connection refused rejects; any HTTP answer, 404 included, resolves. Verified
across a full kill/restore cycle: offline within ~9 s, back to live
automatically. There is a RECONNECT button for the cases this misses.

## CORS, and the dev-server proxy

Perception ships `CORSMiddleware`, so the app talks to it directly.

If a service is reachable but CORS is not — an orchestrator someone is still
writing, a tunnel, a phone IP-cam — set `PROXY_PERCEPTION` or
`PROXY_ORCHESTRATOR` in `.env.local` and point the matching Settings field at
`/perception` or `/orchestrator`. The Vite dev server proxies both HTTP and
WebSockets, and nothing else changes.

## Known limits

- `SpeechRecognition` is Chrome/Edge only. Firefox and most of Safari have no
  implementation; typing still works everywhere.
- Chrome's speech recognition uploads audio to Google, so the **mic needs
  internet** even though everything else here is local. Mock mode does not.
- An HTTPS deployment cannot reach `http://localhost:8001`. The services would
  need a tunnel (`cloudflared tunnel --url http://localhost:8001`) first.
