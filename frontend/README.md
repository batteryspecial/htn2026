# Retask Console (operator front end)

Voice or text instruction in, **TaskSpec JSON** out. Pure HTML/CSS/JS — no build,
no npm, no dependencies.

## Run

```bash
cd frontend
python -m http.server 5173
```

Open <http://localhost:5173> in **Chrome or Edge**.

Do not open `index.html` by double-clicking it. `file://` breaks the microphone
and the fetch to the orchestrator. `http://localhost` is a secure context, so
everything works there.

## Modes

The pill in the top bar cycles **OPENAI → LIVE → MOCK**. Same choice lives in
**SETTINGS**.

| Mode | What it does |
| --- | --- |
| `OPENAI` | Compiles in the browser against `api.openai.com` using the key in `config.local.js`. Standalone — needs no orchestrator and no teammates. ~2–3 s per compile on the default frontier model. |
| `LIVE` | `POST {apiBase}/instruction {text}`, listens on `{apiBase}/ws/status`. Danny's orchestrator. |
| `MOCK` | In-page keyword rules. No network, no internet. The fallback if the venue WiFi or a teammate's laptop dies mid-demo. ~0.5 s. |

Nothing is hardcoded. Share a preconfigured link with `?api=http://10.0.0.5:8000`
or force the offline path with `?mock=1`.

## The API key

`config.local.js` holds the OpenAI key and **is gitignored**. This repo is
public, so a committed key gets scraped and auto-revoked within minutes. Never
move the key into `index.html`, `api.js`, or `localStorage`.

New machine:

```bash
cp config.local.example.js config.local.js   # then paste the key in
```

Two things to understand about this path:

- Anyone who can load the page can read the key from view-source. On localhost
  that is only you. **Do not deploy this to Vercel with the key in it** — that
  publishes the key to every visitor.
- The right long-term shape is compiling on Danny's server, where the key stays
  server-side. `OPENAI` mode is the standalone demo path, not the final one.

Rotate the key at <https://platform.openai.com/api-keys> when the hackathon ends.

### How the compile works

`api.js` sends the instruction with a strict `json_schema` response format, so
the model cannot return prose or an off-contract shape. Two deliberate details:

- OpenAI strict mode rejects `minItems` / `maxItems`, so the "1–2 targets,
  1–6 detect prompts" bounds are stated in the prompt and enforced by
  `taskspec.js`. That is what the VALID/INVALID badge is checking.
- `spec_id`, `model: "yoloe"`, and `arbitration` on two-target specs are filled
  client-side. The model dropped `arbitration` about half the time; the value is
  fixed by the contract, so there is no reason to let it be a variable.

### Choosing a model

Set it in SETTINGS, or as `openaiModel` in `config.local.js`. Default is
`gpt-6-astra` — chosen deliberately for headroom on phrasing nobody rehearsed,
accepting ~2–3 s per compile. `gpt-5.4-mini` is the fast fallback at ~0.8 s if
the displayed retask time turns out to matter more on the day.

**Models disagree about request parameters.** `gpt-6-astra` and `gpt-5.6-terra`
reject `temperature: 0` outright ("only the default (1) is supported"), while
`gpt-5.4-mini` and `gpt-4o-mini` accept it. `api.js` sends `temperature: 0`,
catches that specific 400, retries without it, and records the model in
`localStorage` so the discovery round trip happens once ever rather than once
per compile. Swap models freely; this is handled.

`requestTimeoutMs` is 45 s. Frontier models stall occasionally — `gpt-5.6-luna`
took 11.7 s on one case — and the old 15 s limit would have shown that as a
network error.

Six models were benchmarked on the demo script and on off-script phrasing a
judge might use. On the demo script every model scored 6/6 in 740–870 ms —
the task is small enough that capability is not the constraint. Off-script,
before the prompt was tightened:

| model | off-script | latency spread |
| --- | --- | --- |
| `gpt-5.4-mini` | 4/5 | 623–870 ms |
| `gpt-4.1` | 3/5 | 450–1282 ms |
| `gpt-4o-mini` | 3/5 | 552–1477 ms |
| `gpt-4.1-nano` | 2/5 | 544–1616 ms |

**The prompt was the bottleneck, not the model.** Three rules added to
`SYSTEM_PROMPT` — several of one class is still one target, slang resolves to a
concrete noun, `detect` is never empty — took both `gpt-5.4-mini` and
`gpt-4o-mini` to 5/5 with no latency change. Tune the prompt before reaching for
a bigger model.

### Frontier models were tested too

`gpt-5.5`, `gpt-5.6-luna`, `gpt-5.6-sol`, `gpt-5.6-terra` and `gpt-6-astra` were
run against an adversarial set — instructions the contract cannot express
(negation, conditionals, spatial relations, three targets, a bare pronoun).

They were genuinely better than `gpt-5.4-mini` at first: `gpt-5.4-mini` emitted a
three-target spec (invalid) and used `"woman"` as a detection class, while every
frontier model degraded gracefully. **Two more prompt rules closed the whole
gap** — cap at two targets, every human is `"person"`.

After the fix:

| model | median | worst | adversarial |
| --- | --- | --- | --- |
| `gpt-5.4-mini` | 780 ms | 803 ms | 5/6 |
| `gpt-5.6-terra` | 1234 ms | 4116 ms | 5/6 |

On the *tested* set that is equal quality for 1.6× the latency. The deliberate
call was to run frontier anyway: the benched cases are the ones we thought of,
and the frontier models were already correct on all of them *before* the prompt
was patched. That margin is insurance against the phrasing a judge invents on
the spot, and a retask that takes three seconds and works beats one that takes
one second and locks onto the wrong object.

Avoid `gpt-5.6-luna` specifically: an 11.7 s outlier when it spent 1024 reasoning
tokens deciding which of three pens to drop.

The pattern still held twice, though: **every accuracy gap so far was a prompt
gap, not a model gap.** When something breaks, fix `SYSTEM_PROMPT` and re-run the
benches before blaming the model.

Neither model handles "the mug to the left of the laptop" correctly — both claim
`relate(mug <- laptop)`, but `relate` means containment, not direction. No model
fixes that; the contract has no spatial operator.

### A gap in the frozen contract

"keep following what you're already on" should compile to `select: "locked"`,
but the contract requires `detect` to hold 1–6 entries, and a lock-on retask has
no new class to detect. Every model either emitted an empty `detect` (invalid)
or invented a class name. The prompt now forces a best guess, but Danny and
Qinkai should decide whether `detect` ought to be optional when
`select == "locked"`.

## Controls

| Key | Action |
| --- | --- |
| `Enter` | compile |
| `Shift`+`Enter` | newline |
| `Ctrl`+`M` | start / stop the mic |
| `Esc` | cancel listening, close settings |
| `V` | toggle spoken confirmations |

**Spacebar is deliberately unbound.** It was reserved for E-STOP. The car is out
of scope now, so it is free — but leave it free unless something genuinely needs
a panic key, because space is what people hit when they mean "stop".

## Camera

The annotated feed from Qinkai's pipeline is the projected centrepiece. It is a
plain `<img>` pointed at an MJPEG endpoint — the pipeline draws the boxes, the
UI draws nothing. The objective compiled from your last instruction is overlaid
across the bottom as a platform-sign band.

Set the URL in SETTINGS or with `?video=http://10.0.0.7:8080/video`. Point it
straight at a phone IP-cam app to test the pane before the pipeline exists.

**Why there is a host probe.** An MJPEG `<img>` dies silently: when the server
goes away mid-stream it fires no `error`, no `load`, and the last frame sits
there looking perfectly live. Measured — 14 s after killing the stream the pane
still said LIVE. A per-frame heartbeat does not help either, because Chrome does
not emit a load event per part of a multipart stream.

So every 4 s the page fetches the stream's *origin* with `mode: no-cors`.
Connection refused rejects; any HTTP answer, 404 included, resolves. On failure
the pane flips to OFFLINE and retries with backoff until frames return. Verified
across a full kill/restore cycle: offline within ~9 s, back to live
automatically. There is a RECONNECT button for the cases this misses.

## What the screen shows

- **Timer** — milliseconds from send to spec received, counting live. This is the
  seed of the retask metric; the full `received → active` number lands with the
  status board.
- **VALID / INVALID badge** — the TaskSpec is checked in the browser against the
  frozen contract (1–2 targets, 1–6 detect prompts, `mode` in `follow|center`,
  `select` enum, `model == "yoloe"`). A hallucinated field shows up here in red
  instead of in Qinkai's pipeline.
- **Stage strip** — `received · compiled · sent · prepared · applied · active`,
  lighting up from `/ws/status`.
- **History** — last 8 instructions and their compile times, click to recall.

## Wiring up the real orchestrator

Set the API base URL in settings and flip to `LIVE`. The response parser is
deliberately tolerant — it accepts the TaskSpec at the top level, or nested under
`spec` / `taskspec` / `data` / `result`, or an `instruction_id` alone with the
spec arriving later as a `compiled` status event.

Two things are needed from the orchestrator:

1. **CORS.** The page is served from `:5173` and the API is on `:8000`, so that
   is a cross-origin request. Without `CORSMiddleware` the fetch fails with a
   bare "Failed to fetch". The health dot goes red and its tooltip says so.
2. **Confirm the response shape** — synchronous spec, or `instruction_id` plus a
   `compiled` event on `/ws/status`. Both already work; this is just so we know
   which one we are demoing.

## Known limits

- `SpeechRecognition` is Chrome/Edge only. Firefox and most of Safari have no
  implementation; typing still works everywhere.
- Chrome's speech recognition uploads audio to Google, so the **mic needs
  internet** even though everything else here is local. Mock mode does not.
- Hosting this on Vercel later is one settings change, but an HTTPS page cannot
  reach `http://localhost:8000` — the orchestrator would need a tunnel
  (`cloudflared tunnel --url http://localhost:8000`) to get an HTTPS URL first.

## Files

| File | Contents |
| --- | --- |
| `index.html` | markup |
| `styles.css` | TTC Dupont theme, projector-scale |
| `app.js` | speech, timer, stages, history, rendering |
| `api.js` | openai + orchestrator + mock adapters, status socket, settings |
| `taskspec.js` | contract mirror, validator, JSON highlighter |
| `config.local.js` | the API key. **gitignored** |
| `config.local.example.js` | template, committed |
