# Orchestrator — the agent layer (:8000)

Turns a sentence into behaviours the pipeline runs every frame.

The agent decides **what** should happen. Perception decides **how**, on every
frame, without waiting for anyone. They share no code and no memory — only
`linker/schemas.py`, the JSON both sides agree on.

The old design was a translator: one sentence in, one `TaskSpec` out. This is
a loop. One instruction can register a reference, probe three wordings, start
two behaviours and set the HUD, using each answer to decide the next call.

---

## Where it sits

```
 ┌──────────────────────────── FRONTEND :3000 ─────────────────────────────┐
 │  ENRICHED FEED      CHAT + UPLOAD        AGENT TRACE      EVENTS/ALERTS │
 └────────▲──────────────────┬───────────────────▲──────────────────▲──────┘
          │ GET /video       │ POST /chat        │ WS /ws/trace     │
          │ (MJPEG, direct   │ (text + images)   │ (thought, tool   │
          │  from :8001)     ▼                   │  call, say,      │
          │       ┌──────────────────── ORCHESTRATOR :8000 ─────────┴────┐
          │       │                                                       │
          │       │   user turn ──▶┌───────────────┐──▶ reply / say()     │
          │       │                │  AGENT LOOP   │                      │
          │       │  event turn ──▶│  ≤ 10 steps   │──▶ tool calls        │
          │       │       ▲        └───────┬───────┘                      │
          │       │       │ 1 per behaviour/5s     │                      │
          │       └───────┼────────────────────────┼──────────────────────┘
          │               │ WS /ws/events          │ HTTP
          │               │ (inbound, the only     │ /behaviors /probe
          │               │  socket we consume)    │ /query /describe
          │               │                        │ /references /model /hud
 ┌────────┴───────────────┴────────────────────────▼──────────────────────┐
 │                    PERCEPTION :8001 — every frame                       │
 └─────────────────────────────────────────────────────────────────────────┘
```

**Protocols are not interchangeable.** Tool calls are HTTP request/response
(`tools/perception.py`). The event stream is a WebSocket we consume
(`events/watcher.py`). The trace is a WebSocket we serve (`app/trace.py`).
Each lives in exactly one file.

---

## The loop

```
   instruction, or an event that woke us
                 │
                 ▼
 ┌───────────────────────────────────────────────────────────────┐
 │ GROUND                                    (no model call yet) │
 │                                                               │
 │   GET /behaviors   what is already running, and its state     │
 │                    — plus "ACTIVE with 0 matches" called out, │
 │                      because that is failure wearing success  │
 │   GET /health      fps, resident model, camera ok             │
 │   GET /models      open-vocabulary, or a fixed class list?    │
 │   memory.brief()   wordings measured before, for a subject    │
 │                    like this one                              │
 └───────────────────────────────┬───────────────────────────────┘
                                 ▼
 ┌───────────────────────────────────────────────────────────────┐
 │ THINK               one model call, every tool bound          │◀─┐
 └───────────────────────────────┬───────────────────────────────┘  │
                                 │                                  │
                    tool calls?  ├──── no ───▶ FINISH ──▶ reply     │
                                 │ yes                              │
                                 ▼                                  │
 ┌───────────────────────────────────────────────────────────────┐  │
 │ ACT       run them in order, one at a time                    │──┘
 │                                                               │
 │   Sequential on purpose: "stop the old one, then start the    │
 │   new one" is a common pair, and running those concurrently   │
 │   is a race that sometimes leaves the pipeline empty.         │
 │                                                               │
 │   A tool never raises at the agent. A refusal comes back as   │
 │   its reason, so the next THINK can fix the call.             │
 └───────────────────────────────────────────────────────────────┘

   capped at 10 tool calls — a demo that thinks for twenty has already lost
```

`GROUND` is what makes this better than a one-shot compiler. The model is told
what is running and what has been measured *before* it writes anything, so it
can probe, read the result, and only then commit.

---

## A turn, end to end

> "Watch over this yellow duck but ignore anybody wearing a black jacket."

```
GROUND     nothing running · yoloe (open vocab) · 31 fps
           memory: "duck" 0.00 found nothing · "yellow duck" 0.22 12/12
                   · "rubber duck" 0.58 11/12

THINK   ──▶ probe_phrases("the yellow duck",
                          ["yellow rubber duck", "rubber duck", "duck toy"])
ACT     ◀── "yellow rubber duck" 0.44, "rubber duck" 0.31, "duck toy" 0.00
            use 'yellow rubber duck'. Sending the others alongside costs
            nothing and only helps.

THINK   ──▶ start_behavior({
              kind: "watch",
              subject: {detect: ["yellow rubber duck", "rubber duck"],
                        pick: "largest"},
              params: {triggers: [
                {type: "near",
                 other: {detect: ["person"],
                         include: ["a person in light coloured clothing"],
                         exclude: ["a person wearing a black jacket"]}}]},
              render: {label: "the duck", color: "#FFD400"}})
ACT     ◀── Started b3.

THINK   ──▶ set_hud("watching the duck")
ACT     ◀── HUD now reads: watching the duck

FINISH  ──▶ "Watching the duck. I'll tell you if anyone but a black jacket
             gets close."
```

Then, seconds later, with no model call in between — the pipeline is running
b3 on every frame by itself:

```
WS /ws/events  {type: "near", behavior_id: "b3", notify: true,
                detail: "person (track 17) near duck",
                snapshot_url: "/snapshots/e42.jpg"}
                 │
                 │  notify? yes.  cooldown on b3? clear.
                 ▼
EVENT TURN  ──▶ agent sees the crop, calls say("Someone in a grey hoodie is
                reaching for the duck")
                 │
                 ▼
WS /ws/trace  {kind: "say", label: "Someone in a grey hoodie is …"}
                 → the frontend shows an alert and speaks it
```

Authorisation is not a separate mechanism. "Anyone in a red hoodie is allowed"
is an `exclude` on the trigger's own selector — the same field, used the same
way.

---

## Tools

| Tool | Calls | For |
|---|---|---|
| `start_behavior` | `POST /behaviors` | every camera capability. Argument schema **is** `BehaviorSpec` |
| `stop_behavior` | `DELETE /behaviors/{id}` | drop one, leave the rest |
| `clear_behaviors` | `DELETE /behaviors` | "stop", and before a replacement objective |
| `list_behaviors` | `GET /behaviors` | re-check after a change (GROUND already supplies it) |
| `probe_phrases` | `POST /probe` | **does this wording find anything, in this room, right now** |
| `count_objects` | `POST /query/count` | "how many…" — median over a window, installs nothing |
| `look` | `POST /query/look` | what is tracked now, with attribute scores |
| `describe_scene` | `POST /describe` | "what's on the table?" — sweeps a broad vocabulary |
| `add_reference` | `POST /references` | "follow **him**" with no photo |
| `set_model` | `POST /model` | demo 16. Behaviours the new model cannot serve pause and resume themselves |
| `set_hud` | `POST /hud` | the label burned onto the projected frame |
| `say` | — | one spoken sentence, for alerts |

The argument schemas are the shared pydantic contracts, imported from
`linker/schemas.py`. The model is shown the same field descriptions the
pipeline validates against, so there is no second copy to drift.

### `probe_phrases` is the one that matters

Phrasing is the single most common cause of a behaviour that starts cleanly
and then matches nothing — and it does not transfer between scenes. The same
duck, two rooms:

| `detect` | white table | classroom |
|---|---|---|
| `"yellow duck"` | 12 / 12 | **0 / 12** |
| `"rubber duck"` | 11 / 12 | **0 / 12** |
| `"yellow rubber duck"` | — | 8 / 12 |

So the agent measures instead of guessing. One round trip, on a spare
detector, before it commits.

---

## Memory (LanceDB)

Not the rules — those are ~5k tokens and live in the system prompt, where they
always apply. Chunking them is worse than not retrieving at all, because Rules
4 and 5 are coupled: fetching "put the adjective in `include`" without "always
pair `include` with `exclude`" produces a weaker spec than either alone.

What the table holds is **evidence**: which wording scored what, where, and
whether the behaviour built on it ever acquired.

```
  ┌────────── seeded from the numbers already measured ──────────┐
  │  "duck" 0.00 found nothing      · "yellow duck" 0.22 12/12   │
  │  "face" 0.00 found nothing      · "person's face" 0.48       │
  │  "person" 0.88 — the only class that clears every threshold  │
  │  cream coat scored 0.78 for "dark jacket" — pair the exclude │
  └──────────────────────────────────────────────────────────────┘
                              │
             GROUND ──────────┤ read: a prior, best evidence first
                              │
        probe / outcome ──────┘ write: every measurement, hits and zeros
```

The zeros matter as much as the hits: *"duck scored 0.00 here"* is what stops
the agent trying it again.

Vector search when `ORCH_EMBED_MODEL` is set, full-text otherwise. Full-text
needs no network, which is the right mode for an offline venue. Losing the
store entirely costs a prior, not the ability to work — the agent still
probes.

---

## Event turns

```
  /ws/events ──▶ notify? ──▶ cooldown, 1 per behaviour / 5 s ──▶ agent turn
        │           │                    │
        │           │ no                 │ suppressed
        └───────────┴────────────────────┴──▶ trace only, agent stays quiet
```

Also never woken by `count_changed`, `crossed`, `step` or `camera_ok` — these
fire continuously by nature and carry no decision.

A person standing near the duck fires `near` at frame rate. Without the
cooldown the agent is called thirty times a second and the demo stalls on rate
limits. One alert that arrives is worth more than a hundred that get dropped.

Turns are serialised behind a lock. A user turn and an event turn landing
together would otherwise both read "what is running", both decide to replace
it, and race to an empty pipeline.

---

## API

| Method | Path | Notes |
|---|---|---|
| POST | `/chat` | `text` + optional `images[]`, multipart. The turn. Uploads are registered as references before the agent thinks |
| WS | `/ws/trace` | thought · tool_call · tool_result · say · reply · event · timer · error |
| POST | `/instruction` | `{text}`. Kept as an alias so the console's LIVE mode never breaks |
| WS | `/ws/status` | the stage strip, derived from the trace |
| GET | `/health` | ours, plus whether perception is reachable |

---

## Run

```bash
cd orchestrator
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Windows
cp .env.example .env            # then paste the OpenAI key
python main.py                  # :8000
```

The key lives here, not in the browser. `frontend/.env.local` holds service
URLs only.

`orchestrator/.gitignore` currently excludes `/documents`, which is where
`SELECTORS.md` and `BEHAVIORS.md` live — and the system prompt is built from
those two files. A fresh clone has no prompt, and `agent/prompts.py` raises
saying so rather than quietly compiling worse specs. Un-ignore the folder
before anyone else clones this.

## Layout

```
orchestrator/
  main.py  config.py  contracts.py        <- contracts re-exports linker/schemas.py
  documents/   SELECTORS.md BEHAVIORS.md  <- the prompt, as documents
  agent/       graph.py runner.py prompts.py llm.py
  tools/       perception.py registry.py
  memory/      store.py seed.py
  events/      watcher.py
  app/         api.py trace.py
```
