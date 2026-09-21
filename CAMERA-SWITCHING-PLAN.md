# Map: switching cameras from Settings

Plan only. No code yet. Target hardware: a USB Logitech webcam alongside the
laptop's built-in camera, on Windows.

---

## 1. The one decision that shapes everything

**The switch happens in perception (:8001), not in the browser.**

It is tempting to reach for `navigator.mediaDevices.enumerateDevices()`, because
that is how a web app usually picks a camera. It is the wrong layer here. The
browser never touches the camera — it receives an MJPEG stream from
`GET /video`, and the frames that stream shows have already been through YOLOE,
ByteTrack and the behaviour layer. The camera is opened by OpenCV in a Python
thread on the machine running perception. So:

- enumeration must be server-side (what OpenCV can open),
- the switch is an HTTP call to perception,
- the frontend only ever picks from a list perception gave it.

The browser's device list would name cameras perception cannot open and index
them differently. Do not use it.

**This is the same shape as the detector swap that already works:**
`GET /models` → a `SelectField` in the drawer → `POST /model` on save. Camera
switching should be that pattern again, not a new one.

---

## 2. What exists today

| Piece | Where | What it does |
|---|---|---|
| `Capture` | `perception/runtime/capture.py` | One thread, one slot, newest frame wins. Handles a webcam index, a file that loops, or a URL. Already reconnects on read failure with backoff. |
| Construction | `perception/main.py:52` | `capture = Capture()` — once, at boot, from `CFG.VIDEO_SOURCE`. |
| Ownership | `Service.capture` (`server/api.py:65`) and `InferenceLoop.capture` (`runtime/loop.py:81`) | **Two references to the same object.** |
| Reads | `runtime/loop.py:206` | `self.capture.read()` each tick; `capture.age()` drives `camera_ok`. |
| Config | `perception/config.py:34` | `VIDEO_SOURCE` env, default `"0"`. |
| Deferred mutation | `Builder` (`runtime/workers.py`) | The established way to change pipeline state without blocking the frame loop — `Op("set_model", ...)`. |
| Frontend prefs | `frontend/src/config/settings.ts` | `Preferences` in localStorage; `detector` is stored here and applied with `POST /model`. |
| Frontend UI | `components/settings/SettingsDrawer.tsx` | The drawer, with the detector `SelectField` to copy. |

Two facts from that table matter later:

1. `Capture` **already** has a reopen path — `_run()` does
   `if self._cap is None and not self._open()`. A switch is mostly "make
   `_cap` None and change `source`", not new machinery.
2. Two objects hold the `Capture` reference, so **mutate it in place; do not
   replace it.** Building a new `Capture` means updating both references under
   a lock while the frame loop is reading, which is a race for no benefit.

---

## 3. Enumeration on Windows — the hard part

OpenCV cannot list cameras. `cv2.VideoCapture(i)` either opens or does not, and
there is no name attached. Three options:

### 3a. `pygrabber` — recommended

```
FilterGraph().get_input_devices()  ->  ["HD Pro Webcam C920", "Integrated Camera"]
```

Pure Python over DirectShow (`comtypes`), tiny, no build step. The list index is
the DirectShow device index.

**The trap, and it is the whole reason this needs care:** that index matches
`cv2.VideoCapture(i, cv2.CAP_DSHOW)`. It does **not** reliably match
`cv2.CAP_MSMF`, which is what OpenCV picks by default on Windows — today's
`_open()` calls `cv2.VideoCapture(target)` with no backend. So enumerating with
DirectShow while opening with MSMF can hand you the wrong camera.

If you take this route, `_open()` must pass `cv2.CAP_DSHOW` explicitly on
Windows. Enumerate and open through the same backend or the indices are
meaningless.

### 3b. Probe indices 0..7 — the cross-platform fallback

Open each index, grab one frame, record whether it worked and at what
resolution. Needs no dependency and works on Linux/macOS.

Costs: 200–500ms per index (worse on MSMF, which can take seconds to fail), no
device names — only "Camera 0", "Camera 1" — and **it is disruptive**: probing
an index that is already open can fail or steal the device. Any probe must skip
the index the live `Capture` currently holds and report that one from the live
object instead.

Cache the result; do not re-probe on every `GET /cameras`.

### 3c. PowerShell / WMI — names only, for display

`Get-CimInstance Win32_PnPEntity` filtered to `PNPClass -eq 'Camera'` gives real
device names but **no mapping to an OpenCV index**. Useful only to prettify a
list obtained another way. Not a primary source.

**Recommendation:** 3a on Windows, 3b elsewhere and as the fallback when
`pygrabber` is missing. Put it behind one function —
`perception/runtime/cameras.py: list_cameras() -> list[CameraInfo]` — so the
platform mess is in one file and can be monkeypatched in tests.

---

## 4. Logitech-specific gotchas

These are the difference between "it switches" and "it switches and then runs at
5 fps", and they are worth building in from the start rather than debugging on
stage.

- **Pixel format decides your frame rate.** Most Logitech webcams (C920, C922,
  Brio) expose both YUY2 (uncompressed) and MJPG. OpenCV commonly negotiates
  YUY2, which at 1080p over USB 2.0 has the bandwidth for about **5 fps**. Ask
  for MJPG and the same camera does 30:
  `cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))`.
- **Order matters.** Set FOURCC *first*, then width, then height. Setting
  resolution before format silently resets the format on some drivers.
- **`CAP_DSHOW` opens faster and more reliably than `CAP_MSMF`** on Windows for
  these devices. MSMF can take 2–5 seconds to open a Logitech and occasionally
  fails outright.
- **Verify what you got.** `cap.set(...)` returns True having done nothing on
  plenty of drivers. Read `CAP_PROP_FRAME_WIDTH` / `HEIGHT` / `FPS` back after
  opening and report the *actual* values in `GET /cameras`, not the requested
  ones.
- **Indices shuffle on replug.** Unplug the Logitech and index 1 may become the
  laptop camera. See §7 — binding by name avoids quietly switching to the wrong
  camera.
- **Two USB cameras at once can exceed hub bandwidth.** This constrains the
  failure-handling design in §6: you may not be able to hold the old camera open
  while testing the new one.

---

## 5. The swap itself

Add to `Capture`:

```
switch(source: str) -> None       # request a new source; the thread does the work
pending / current source          # guarded by the existing self._lock
```

The capture thread notices a pending source at the top of its loop, releases
`self._cap`, sets it to `None`, and falls into the reopen path that already
exists. `Capture.source` changes; the object identity does not, so
`Service.capture` and `InferenceLoop.capture` both stay valid and nothing else
needs touching.

**Route it through `Builder`, not straight from the HTTP handler.** `Op("set_camera", source=...)` follows the same discipline as `set_model`: the API
thread never reaches into the frame loop's world. (`Capture.switch` is itself
thread-safe, so a direct call would *work* — but every other pipeline mutation
goes through the queue, and one exception is how ordering bugs start.)

### What breaks during the ~0.3–2s gap, and what must not

| Thing | What happens | Action needed |
|---|---|---|
| `camera_ok` | Goes false once frame age exceeds `CAMERA_DEAD_S` (2s) | None — this is honest, and the CAMERA dot in the top bar correctly goes red |
| Installed behaviours | Keep running, match nothing briefly | **None. Do not clear them.** A camera change is not a change of objective |
| ByteTrack track ids | Meaningless across a camera change — a `track` behaviour locked to track #7 will never see it again | **Reset the tracker on switch.** The `LOST → SEARCHING` machine would eventually recover, but a reset is immediate and correct |
| References (CLIP vectors) | Still valid — appearance is camera-independent | None |
| MJPEG stream | `Streamer` reads `loop.view`, so it serves the last frame then a placeholder | Frontend should auto-reconnect after a successful switch (§8) |
| Resolution change | `IMGSZ` letterboxes, so inference is fine | Audit anything caching frame dimensions — the normalized coordinate maths assumes it reads them per frame |
| HUD / objective | Unaffected | None |

---

## 6. Failure handling — decide this before writing anything

**The pipeline must never end up blind because a switch failed.**

The obvious design — hold the old camera open, open the new one, keep whichever
works — is unsafe here: two USB cameras on one hub may not both open (§4), so
the test itself can fail for a reason that has nothing to do with the new
camera.

Recommended sequence:

1. Release the old camera.
2. Try to open the new one, with a bounded timeout (~3s; MSMF can hang).
3. **On failure, reopen the old source** and return `422` naming what happened.
4. On success, report the actual resolution and fps that came back.

The revert step is the part that is easy to leave out and expensive to discover
live. There should be a test for exactly that path: switch to a bad index, and
assert the pipeline is still serving frames from the original camera afterwards.

---

## 7. Stable identity — worth doing

An index is a poor name for a camera: replug the Logitech and index 1 might be
the laptop. Prefer accepting **either** in `POST /camera`:

- `{"source": "1"}` — an index, what works today
- `{"name": "HD Pro Webcam C920"}` — resolved to an index at open time

Storing the *name* in the frontend preference means a reconnected webcam is
still the webcam after a replug, rather than silently becoming the built-in
camera. If the named device is gone, fail with a clear 422 rather than falling
back to index 0 — falling back to the wrong camera is worse than not switching.

---

## 8. API surface

Mirror `/models` and `/model` exactly.

```
GET /cameras
  -> { "cameras": [ { "index": 0, "name": "Integrated Camera",
                      "backend": "dshow", "active": false, "available": true,
                      "width": 1280, "height": 720, "fps": 30.0,
                      "detail": null } ],
       "source": "1" }

POST /camera   { "source": "1" }  |  { "name": "HD Pro Webcam C920" }
  -> 202 { "accepted": true, "source": "1", "name": "...",
           "width": 1920, "height": 1080, "fps": 30.0 }
  -> 422 { "detail": "could not open camera 3: device not found" }
```

Notes:

- `GET /cameras` should serve a cached enumeration and take `?refresh=1` to
  force a re-scan, because probing is disruptive (§3b).
- Keep the 422 sentence specific, like every other refusal in this service —
  the agent reads these.
- **Contract question to settle:** adding a `camera` field to `Health` in
  `linker/schemas.py` would let the status strip show which camera is live. That
  file is `extra="forbid"` and explicitly shared ("change this only by
  agreement"), so it is a coordination cost, not a free win. The alternative is
  leaving camera identity in `GET /cameras` only. **Ask before touching the
  shared schema.**
- An agent tool (`set_camera`) is deliberately **not** proposed. "Switch to the
  webcam" is an operator action, not something the agent should decide mid-turn.

---

## 9. Frontend changes

| File | Change |
|---|---|
| `services/perception.ts` | `listCameras()`, `setCamera(source)` — beside the existing `setModel` |
| `hooks/usePerception.ts` | `useCameras(perception, base)`, a copy of `useModels` including its `refresh` |
| `config/settings.ts` | `Preferences.camera?: string`. Add to `DEFAULT_PREFERENCES` and to the destructure in `saveSettings` — **that function lists keys explicitly, so a new key is silently dropped if you forget it** |
| `components/settings/SettingsDrawer.tsx` | A `SelectField` "Camera" next to "Detector model". On save, `if (draft.camera !== settings.camera) onSelectCamera(draft.camera)` — exactly the detector's shape |
| `App.tsx` | Wire `useCameras`, pass `cameras` + `onSelectCamera` to the drawer, and add `refreshCameras()` to the existing `onOpenSettings` handler beside `refreshModels()` |
| `hooks/useVideoStream.ts` | After a successful switch, call `reconnect()`. The MJPEG `<img>` will not recover on its own from a stream that stopped mid-frame |

Label the select with the real resolution and fps, as the detector select does
with its class count — `HD Pro Webcam C920 · 1920×1080 · 30fps`. Mark
unavailable devices `disabled`, again as the detector select does.

---

## 10. Config additions

In `perception/config.py`, beside `VIDEO_SOURCE`:

- `CAMERA_BACKEND` — `auto | dshow | msmf | v4l2 | avfoundation`, default `auto`
  (which should mean **dshow on Windows**, given §4)
- `CAPTURE_WIDTH` / `CAPTURE_HEIGHT` — `0` meaning "whatever the driver gives"
- `CAMERA_FOURCC` — default `MJPG`, the fix for the 5 fps problem
- `CAMERA_OPEN_TIMEOUT_S` — default `3.0`

`VIDEO_SOURCE` keeps its current job: the camera at boot.

---

## 11. Testing

Perception's suite must stay runnable with no camera attached, so:

- `list_cameras()` lives alone in `runtime/cameras.py` and is monkeypatched in
  every test. No test may call the real enumerator.
- `Capture.switch()` is testable against a file source — switch between two
  clips and assert frames keep arriving and `source` changed.
- `POST /camera` with a bad source → 422, **and the original source is still
  live afterwards** (§6). This is the test that matters most.
- Behaviours survive a switch: install one, switch, assert it is still in
  `GET /behaviors` and was not cleared.
- Frontend: `saveSettings` round-trips the new `camera` key (the explicit-keys
  trap in §9).

---

## 12. Suggested order

1. `runtime/cameras.py` — enumeration + `CameraInfo`, pygrabber with probe
   fallback. Verify by hand that the listed index opens the camera you expect.
2. `Capture` — backend selection, FOURCC/resolution, and `switch()`.
3. `Builder` op + `GET /cameras` / `POST /camera`, including the revert-on-failure
   path and its test.
4. Frontend: service → hook → drawer field → auto-reconnect.
5. Optional: name-based binding (§7), and the `Health.camera` contract question
   (§8) if the team agrees to it.

Steps 1–3 are independently testable with `curl` before any UI exists, and that
is the right place to find out whether the Logitech runs at 30 fps or 5.

---

## 13. Open questions for the operator

1. **Should a camera switch clear running behaviours?** Recommendation: no — it
   is a change of sensor, not of objective. Worth confirming, because the
   opposite is defensible for a demo.
2. **Is `Health.camera` worth a shared-contract change?** (§8)
3. **Is one USB webcam plus the built-in the whole scope**, or should this handle
   an IP camera / a video file chosen from the same dropdown? `Capture` already
   supports all three sources, so the dropdown could offer a file for offline
   rehearsal — cheap to include if wanted, scope creep if not.
