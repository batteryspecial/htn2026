# Camera / model connection audit and implementation handoff

## Implementation result — 2026-09-19

The listed defects were addressed and the updated services were exercised
end-to-end. The final `watch the walnut` turn completed in 8.289 seconds with
`ok: true`, `outcome: applied`, behavior `b2`; `/behaviors/b2/status` reported
`installed`; and `/state` continued to report the same `b2` after the agent
turn ended. It remains in `ARMING` with zero
matches and a visible objective/HUD, so the persistent camera loop is running
until STOP even though the current frame has not acquired the target.

Validation: orchestrator 145 passed; perception 386 passed with 10 optional
weight-dependent skips; frontend production build passed. The detailed sections
below are retained as the failure analysis and acceptance criteria used for the
implementation.

## Immediate failure: resolved at runtime

The orchestrator was still running inside the restricted execution sandbox after
an earlier restart. Its outgoing model connection failed. Perception had already
been restarted outside that sandbox, explaining why video worked while chat did
not.

Evidence from this investigation:

- The configured SDK's read-only models request failed inside the sandbox with
  `APIConnectionError -> ConnectError: All connection attempts failed`.
- The identical request outside the sandbox authenticated successfully and listed
  the configured model. Neither the API key nor the model name needed changing.
- Restarted only the orchestrator outside the sandbox. A real `POST /chat`
  returned HTTP 200, `ok: true`, and `Connection restored.` in 4.043 seconds.
- Perception reported `camera_ok: true`, approximately 31 fps, and zero behaviors.
  The diagnostic chat deliberately requested no behavior changes.

This proves camera capture and the app's model-response path work. It does not
prove detection/acquisition of an object named Walnut. A name alone can still
require a category, reference image, or clarification.

The old `camera_lost: no frame for infs` entries are historical startup events;
they are not evidence that the currently healthy camera is disconnected.

For error classification, OpenAI distinguishes transport connection failures from
authentication failures in its [error guide](https://developers.openai.com/api/docs/guides/error-codes).

## Scope and constraints for the implementing model

Work from the current files, not historical README implementation claims. Follow
the request through frontend -> orchestrator -> shared schemas -> perception.
Do not edit virtual environments or installed dependencies. Leave all git actions
to the human. This audit changes no application source.

The findings below distinguish local reproductions from source-review findings.
The isolated reproductions used fake detectors/graphs, not the running camera
pipeline. Existing automated suites were not rerun in this diagnostic pass.

## Fix first

### 1. Readiness conceals a broken model connection

Files: `orchestrator/app/api.py:184`, `orchestrator/agent/graph.py:137`,
`frontend/src/services/orchestrator.ts:71`.

Observed: `/health` returns `status: ok`; the frontend reports Ready from an API
key's presence and perception reachability. Both remained true during the outage.
The trace exposes only the SDK's generic `Connection error.`

Edit: distinguish process reachability, perception health, and model connection
state. Cache last model success/failure with timestamps and an initial unknown
state. Provide an explicit diagnostic check if needed; do not send billed model
requests on every health poll. Classify transport/authentication/rate-limit/model
errors; log the exception chain and expose a safe, actionable summary without
credentials. Show degraded model status in the frontend.

Check: mock a connection failure with a key present and perception reachable;
the UI must not say model Ready. A subsequent successful turn must recover it.

### 2. A queued behavior is reported as successfully applied

Files: `perception/server/api.py:113`, `perception/runtime/workers.py`,
`perception/runtime/loop.py:123`, `perception/runtime/health.py:89`,
`orchestrator/tools/registry.py:116`, `orchestrator/agent/graph.py:204`.

Reproduced: force a worker loading failure. POST returns 201 with `b1`, but `b1`
never appears in installed behaviors. The failure event has top-level
`behavior_id: null`; its ID is nested in `data.behavior_id`. The tool says Started
and the graph derives `outcome: applied` from the receipt alone.

Edit: keep acceptance asynchronous, but distinguish accepted, installed, and
failed. Correlate worker and frame-installation outcomes to the operation and
behavior ID, including top-level failure-event IDs. Have the agent observe the
outcome with a bounded wait or explicit operation-status query. Do not report
applied until the frame loop confirms installation. Account for cold model load
times; an arbitrary two-second absence is not proof of removal.

Check: successful install, cold install, worker rejection, and detector-apply
failure each produce the correct correlated outcome and frontend state.

### 3. Count/look queries ignore most of the selector

Files: `perception/server/api.py:201`, `perception/server/api.py:404`,
`linker/schemas.py:49`, `orchestrator/tools/registry.py`.

Reproduced with two tracked people: ordinary selection, `pick: largest`, and
`pick: ref` with a nonexistent reference all returned both people and count 2.
The endpoints only filter by detector label; include/exclude, reference, relation,
and picking semantics are not applied.

Source finding: these endpoints also read only existing tracks. Before a detector
world exists, the loop publishes empty tracks even with a healthy camera. A query
for a category absent from running detector prompts cannot establish absence.

Edit: run queries through the shared selector evaluation/picking rules. Arrange
appropriate detection and attribute preparation for query-only requests, using
an isolated query path so standing behaviors are preserved. Reject unknown
references and unsupported filters explicitly. Return unavailable/not evaluated
when there is no usable frame or detection coverage, rather than a confident zero.

Check: idle pipeline; category outside active prompts; largest; include/exclude;
valid/invalid reference; relations; disconnected camera.

### 4. Probe failures look like successful measurements

Files: `perception/server/api.py:218`, `orchestrator/tools/perception.py`,
`orchestrator/tools/registry.py:175`.

Reproduced: without a frame, `/probe` returns HTTP 200 with empty results and
`detail: no frame to look at`. Detector exceptions take the same response path.
The orchestrator ignores that detail and reports `Measured right now:`.

Edit: establish an explicit probe success/failure contract, preferably appropriate
non-2xx responses for unavailable frames or failed inference. Propagate the reason
through the tool and trace. Preserve the distinction between a valid zero-match
measurement and an execution failure. Do not store failed execution as evidence.

Check: no frame, detector exception, valid empty detection, and successful probe.

### 5. Time limits exclude queued/preparation work; timeout loses receipts

Files: `orchestrator/agent/runner.py:101`, `orchestrator/agent/runner.py:169`,
`orchestrator/agent/graph.py`, `frontend/src/services/orchestrator.ts`.

Reproduced: with a 50 ms turn timeout, a task still waited for the runner lock
after 120 ms, then completed successfully. Another simulated turn installed one
behavior before timing out, but its result returned an empty behaviors list.
Source finding: reference registration and snapshot capture also precede the
timeout; the camera snapshot is taken before acquiring the lock.

Edit: establish one monotonic deadline at submission covering preparation, queue
wait, and graph work. Capture the live frame when the turn is ready to execute.
Keep a turn-owned receipt ledger outside graph-local return state, and reconcile
it on failure/cancellation. Prevent an expired queued turn from later modifying
the pipeline. Preserve timeout/error status alongside any partial installation.

Check: occupied lock, slow snapshot/upload, install then timeout, and cancellation
while queued. Report installed work accurately without claiming turn success.

### 6. Pipeline updates can overwrite a failed turn's ERROR badge

Files: `frontend/src/hooks/useAgentRun.ts:85`,
`frontend/src/contracts/liveRun.ts:32`.

Source finding: finish sets ERROR, but a subsequent pipeline update unconditionally
sets APPLIED/ACTIVE when any returned installation IDs satisfy the checks. If IDs
disappear or state disconnects, the effect returns without clearing stale success.
Merely existing in state also includes PAUSED/FAILED behaviors.

Edit: model turn outcome separately from current behavior lifecycle. Keep failed
or partially failed turns visible; show behavior failure/pause/removal and unknown
state on disconnect. Preserve historical reached stages without treating them as
current readiness. Do not let state updates erase a terminal turn error.

Check: partial install followed by model failure; behavior FAILED/PAUSED/removed;
state socket disconnect and reconnect; successful target acquisition.

## Then fix

### 7. Embedding fallback changes vector dimensions

File: `orchestrator/memory/store.py:129`, `orchestrator/memory/store.py:184`.

Reproduced with a simulated 1,536-dimensional embedder: success returns 1,536
values; a network failure returns eight zeros. This conflicts with a fixed-width
vector table. An offline-created eight-dimensional table also needs handling when
embeddings become available later. Failed persistent writes fall back in-process.

Edit: preserve explicit schema/model dimensions and use a real full-text fallback
without submitting an incompatible vector. Detect incompatible existing tables
and migrate/re-embed deliberately. Bound embedding timeout/retries so optional
memory does not stall a turn.

Check: online -> offline -> online and offline-created -> online tables; memory
remains writable/searchable without schema errors or indefinite network waits.

### 8. Green camera indicator measures the stream, not camera health

Files: `frontend/src/App.tsx:171`, `frontend/src/hooks/useVideoStream.ts`,
`frontend/src/components/layout/TopBar.tsx:51`.

Source finding: a valid MJPEG stream showing NO SIGNAL still sets `video.live`.
The camera indicator uses only that value, matching the earlier contradictory
green CAMERA / red NO_CAMERA screenshot.

Edit: combine stream reachability with fresh perception `camera_ok` for the
pipeline-backed camera indicator. Treat missing/stale pipeline state as unknown.
Preserve separate reachability semantics for a custom external video URL.

Check: server remains up while camera disconnects; synthetic no-signal frames
must not produce a green camera-health indicator.

### 9. Legacy status endpoint marks any reply active

File: `orchestrator/app/api.py:39`, `orchestrator/app/api.py:161`.

Source finding: `/ws/status` maps any successful reply to active and accepted
start-tool output to applied. A clarification question therefore looks active.
The current live console uses `/ws/trace`, so this did not cause this screenshot.

Edit: align the compatibility endpoint with confirmed lifecycle state, or clearly
deprecate it if no supported client uses it. A reply alone must not imply execution.

Check: clarification-only, rejected behavior, accepted-only, and acquired turns.

### 10. Optional image argument can crash a helper

File: `orchestrator/agent/graph.py:241`.

Reproduced: `opening_messages('test', 'user', live_image_url='...')` raises
`TypeError: 'NoneType' object is not iterable` because it iterates `image_urls`
despite its default being None. The current runner passes a list, so normal chat
does not hit this branch.

Edit: normalize the optional list before iteration. Check live-image-only,
reference-only, both, and neither.

### 11. Ignore rules can omit required source/prompt files

Files: `.gitignore:23`, `orchestrator/.gitignore:1`,
`orchestrator/agent/prompts.py:112`.

Source finding: root `*.ts` matches TypeScript source as well as model artifacts.
`/documents` excludes required SELECTORS.md and BEHAVIORS.md prompt files. These
rules affect untracked files; this audit did not inspect tracked status or run git.
Also `.venv/` does not cover the existing perception `.venv-light/` directory.

Edit: scope model ignores to model locations, allow required prompt files, and
ignore the actual virtual-environment directory. The human handles staging and
verifying repository contents. Add startup validation for required prompt assets.

Check: the human confirms a clean checkout contains frontend TypeScript and both
prompt documents, but no local environment/dependency trees.

## Documentation and validation boundaries

- Keep the existing shared `linker/schemas.py` contract; avoid new duplicate schemas.
- Do not implement every root README idea to fix this outage. Keyboard guidance
  and physical hardware control need separate scope; consult implemented behavior
  kinds and the model manifest. Local webcam operation does not require port 8003.
- The prior tracker-threshold diagnosis describes an already implemented fix;
  do not reapply historical patches without reproducing a regression.
- Stage labels received/compiled/applied/active are always rendered. Their presence
  in copied text does not mean all four stages completed.
- Implement items 1-6 first, then 7-11. Add focused regression coverage for each
  changed failure path. Run the relevant Python suites and frontend build afterward.
- Finish with one real visible-target instruction and verify the same behavior ID
  across HTTP, trace, pipeline state, and target acquisition. Keep this separate
  from the already-passed model-connectivity check.
