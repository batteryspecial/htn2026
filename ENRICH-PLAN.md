# Plan: implementing what ENRICH.md leaves open

Plan only, no code. Written from `origin/main:ENRICH.md` plus the state of this
working tree on 2026-09-20.

Two things in here are **not** in ENRICH.md, because Qinkai could not have
known them when he wrote it. Both change the order of the work, and §0 is the
one that blocks everything else.

---

## 0. The merge, before anything else

**Qinkai's Part I is not in this working tree.** Everything ENRICH.md
describes as shipped — the gesture vocabulary, `hands`, `wholebody`, `ocr`,
the keyboard behaviour — is on `origin/main` and absent locally.

```
git rev-list --left-right --count HEAD...origin/main
1       8          # 1 commit local-only, 8 commits we do not have
```

`perception/skills/` here contains `pose.py` and nothing else. No
`gestures.py`, no `hands.py`, no `wholebody.py`, no `ocr.py`, no
`behaviors/keyboard.py`. Any work started against this tree would be written
against a pipeline two feature-sets out of date, and several Part II items
(§2, §3.1) touch files that only exist over there.

### What the merge looks like

The good news, measured rather than assumed:

```
git merge-tree --write-tree HEAD origin/main   ->  a clean tree, no conflicts
```

So the *commit-level* merge is clean. The risk is the uncommitted work sitting
on top of it. Nine files are modified locally **and** changed by
`origin/main`:

| file | what origin/main does | what is uncommitted here |
|---|---|---|
| `perception/server/api.py` | ~203 lines rewritten | camera routes (`/cameras`, `/camera`) |
| `perception/runtime/loop.py` | 72 lines removed | `request_reset` / `_pending_reset` |
| `perception/config.py` | +27 lines (OCR, wholebody, hands knobs) | +camera knobs, **same class body** |
| `perception/requirements.txt` | +11 lines | +pygrabber |
| `frontend/src/{App.tsx, config/settings.ts, contracts/behavior.ts, hooks/usePerception.ts, services/perception.ts}` | various | camera dropdown |

`config.py` and `requirements.txt` are near-certain textual conflicts — both
sides append to the same region. The rest should merge but need reading.

### Suggested order

1. **Commit the camera work first**, on its own branch. It is finished and
   tested (18 camera tests, 404 in the suite) and merging it as a unit is far
   easier than reconciling a dirty tree against 8 commits. *You said not to
   commit yet, so this is the first thing to approve, not something to do.*
2. Merge `origin/main`.
3. Re-run both suites. Expect roughly **423 + 18 ≈ 441** perception tests.
   Anything below that is a merge that dropped something.
4. Re-verify the camera switch by hand — `origin/main` rewrote `api.py`
   heavily and `loop.py` lost 72 lines, so the two hooks the camera work
   relies on (`Service.capture`, `loop.request_reset`) both want eyes on them.

### One thing to notice in that merge

`origin/main` **deletes `perception/mobileclip_blt.ts` — a 600 MB binary**
that is still in this tree. Take the deletion. It is still in git history
either way, which is worth a separate conversation after the hackathon.

---

## 0b. Two items on Qinkai's open list are already closed

Do not re-do these.

- **`orchestrator/documents/` is fixed.** ENRICH.md calls it "a demo-day
  blocker and nobody's current task". In this tree both `SELECTORS.md` and
  `BEHAVIORS.md` are tracked by git and `orchestrator/.gitignore` no longer
  excludes the folder — it now carries a comment saying why it must not. A
  fresh clone gets a working agent prompt. **Confirm this survives the merge**,
  since the fix is local-only and `.gitignore` is not in the overlap table
  above.
- **Query expansion is partly done.** See §1.1 — the prompt already tells the
  agent to keep every probed phrasing in `detect`. The remaining delta is
  narrower than ENRICH.md implies.

---

## 1. Root cause 1 — the 0.06 margin

ENRICH.md's own ranking, cheapest first. This is the dark Pi box and the
wristband: a present target scores 0.18–0.27, an absent one 0.10–0.14, and a
matte black enclosure lands inside that band.

**Do not lower thresholds.** That trade is spent: `CONF_THRESHOLD = 0.15`,
real tracker gate 0.12, with a regression test pinning `gate <=
CONF_THRESHOLD`.

### 1.1 Query expansion — smaller than it looks

`orchestrator/agent/prompts.py`, the `WORKFLOW` block, **already** says:

> Send two to four candidate phrasings in one call... Use the `best` it
> returns, and keep the others in `detect` alongside it — detection runs once
> on the union, so extra phrasings are free.

So the union mechanic is in place. What is missing is *what kind* of phrasings
to generate for the objects that actually fail. "box", "small box", "brown
box" are three ways of saying the same prototype; the Pi box needs the axis
changed, not the adjective:

- **material** — "matte black plastic case"
- **function** — "electronics enclosure", "circuit board case"
- **form** — "small dark rectangular object"

Changes, all in the orchestrator, no perception work:

1. Add a short rule to `WORKFLOW`: when the subject is dark, small,
   non-canonical or an object a detector is unlikely to have a prototype for,
   vary phrasings **across material / function / form**, not by stacking
   adjectives on one noun. Name the failure — "box" is near *cardboard
   carton* in embedding space — because the model reasons better from the
   mechanism than from the instruction.
2. Seed the phrase memory (`orchestrator/memory/seed.py`) with the Pi-box and
   wristband measurements from testing. The seeds are the project's own
   numbers and this is exactly the shape they take: a wording, a score, a
   scene. The retrieval path (`brief()` → `ground()`) already puts them in
   front of the model every turn.
3. Nothing in perception changes. Worth stating in the PR so nobody looks.

**Verification:** put the Pi box in frame, `POST /probe` with `["box"]` versus
the three-axis expansion, and record both. If the expansion does not clear
0.15, this item did not work and §1.2 is the next lever, not more phrasings.

### 1.2 Lock exposure and gain — now nearly free

ENRICH.md calls this "closer to the lighting intuition than anything else
here". It got cheaper this week: the camera work added
`Capture._configure(cap)`, which already runs on **every** open, including
after a camera switch. Auto-exposure lock is four more `cap.set` calls in a
method that exists.

- `CAP_PROP_AUTO_EXPOSURE`, `CAP_PROP_EXPOSURE`, `CAP_PROP_GAIN`,
  `CAP_PROP_AUTO_WB`, behind config (`CAMERA_EXPOSURE`, `CAMERA_GAIN`,
  `CAMERA_AUTO_WB`), unset meaning "leave the driver alone".
- **The magic numbers need measuring, not guessing.** On DirectShow,
  `CAP_PROP_AUTO_EXPOSURE` is commonly `0.25` for manual and `0.75` for auto,
  and the exposure value itself is a log2 scale on some drivers and
  milliseconds on others. Read every property back after setting it, the way
  `_read_back()` already does for resolution, and report the actual values in
  `GET /cameras` — `cap.set` returns True having done nothing on plenty of
  drivers.
- Applying this needs `CAMERA_FOURCC`/resolution ordering respected: format,
  then size, then exposure. Exposure set before format is reset by the format
  change on some drivers.

**Verification is the point of the item:** probe one phrase against a static
scene ~20 times with auto-exposure on, then off, and compare the *spread* of
`max_conf`, not its mean. The claim is that a hunting exposure moves scores
across a 0.06 band. Either the spread shrinks or this item is folklore, and
both answers are worth the ten minutes.

### 1.3 Dump the frame the detector receives

The post-resize, post-colour-conversion array, not the preview.

`GET /debug/frame.jpg` on perception, encoding whatever is handed to
`detector.infer()`. Cheapest of the three and it settles the lighting argument
by evidence. Keep it out of `/snapshot`, which has a defined meaning already
(the un-annotated frame for a vision model).

---

## 2. Root cause 2 — references re-rank, they do not detect

The badge. The diagnosis in ENRICH.md is exact and worth restating because it
determines which fix to build: `best_match(sims: dict[int, float], ...)` is
keyed by **track id**, so references rank tracks that already exist. The badge
was never detected, so `sims` was empty and the CLIP embedding had nothing to
rank. Every pixel was discarded at the text bottleneck.

There are two fixes and ENRICH.md prefers the second. **So do I, and I would
build only it this week.**

### 2a. "Show it to me and I'll remember it" — build this

Grab the reference crop from our own camera instead of an uploaded photo.
Same sensor, same lighting, no domain gap.

- `references.add_vector()` already exists for exactly this, and
  `POST /references {"from": "largest_person"}` already does the
  frame-grab-and-embed round trip. The work is a second source — centre crop,
  or largest detection, or a box the operator points at.
- No GPU host required. No ultralytics internals. Testable against a fake
  encoder like the rest of `attributes/`.
- It is demo #2 on Qinkai's own ranked list and it fixes the badge.

**The real caveat:** this still only re-ranks. It works when *something* is
detected at the badge's location — so it wants a broad `detect` (`["object",
"device", "card"]`) plus the reference as the discriminator. That is a
selector-construction rule for the agent prompt, and it belongs in the same
change. Without it the demo fails the same way for the same reason.

### 2b. YOLOE visual prompts — do not start blind

`get_visual_pe` with a box/mask prompt, making an image query a detector input
rather than a post-hoc filter. This is the correct long-term fix.

ENRICH.md's warning is the whole story and should be honoured: the visual
prompt path is `YOLOEVPSegPredictor` with a different `predict` signature, not
a parallel of `get_text_pe`. **It needs the GPU host to develop against, and
writing it unrun is the trap.** If nobody is at the GPU host, this is not this
week's work.

---

## 3. Smaller open items

### 3.1 `needs_roles` — AND to ANY

Loading `wholebody` does not satisfy a `pose` or `hands` gesture even though
it sees everything both do. `world.revalidate` treats `needs_roles` as AND
over `registry.has_role(r)`. Making one model substitute for another means
`needs_roles` becoming a list of *alternatives*.

Qinkai left a test pinning current behaviour, so that test changes with the
semantics — which is correct and should be called out in the PR rather than
looking like a regression. Only worth doing if the frame budget gets tight;
three small models on a 4090 is not tight.

### 3.2 Name the backend on the enriched frame

"A label, not a feature", in ENRICH.md's words, and it reads as sophistication
on a projector. The manifest that closed root cause 3 already knows which
backend answered. Cheap, visible, low risk.

### 3.3 Per-behaviour confidence threshold

`CONF_THRESHOLD = 0.15` is tuned for `track`, where missing the subject is
total failure. `count_line` wants it **higher** — uncertain texture boxes
inflate counts, and `MIN_BOX_FRAC` catches boxes that are small, not boxes
that are uncertain.

This has been flagged and deferred twice (there is a comment about it in
`config.py` today). Counting is the behaviour that forces it, and "how many
people do you see" is a question judges actually ask. Smallest honest version:
an optional `min_conf` in a behaviour's `params`, defaulting to `CONF_THRESHOLD`.

---

## 4. Verify before demo — the real risk column

ENRICH.md ranks these by how badly they hurt on the day, and its ranking is
right. This is the part I would do **first after the merge**, because it is
discovery, not construction, and it changes what is worth building.

1. **Install `mediapipe`, `rtmlib`, `onnxruntime`, `easyocr` on the GPU host.**
   None are on the Mac, so `hands`, `wholebody` and `ocr` report
   `available: false` and boot cleanly — meaning **nothing those three models
   produce has ever run**. The geometry over their output is well tested; the
   output is not. Qinkai's own `CFG.MAX_HANDS` bug is the precedent: code
   behind an optional-dependency guard is untested code, and it failed only
   when the dependency arrived.
2. **Can EasyOCR read keys off a real keyboard?** Camera-only, and it is the
   whole keyboard demo. `OCR_HZ` and `OCR_MIN_CONF` are the knobs. Failing to
   fit gives `SEARCHING` rather than a wrong answer, which is the right
   failure — but find out now.
3. **Counting against a real crowded frame.** See §3.3, plus track-ID churn
   under occlusion (`LOST_TRACK_BUFFER`).
4. **The guessed constants**: `MOUTH_OPEN_RATIO = 0.35`,
   `HAND_DETECTION_CONF = 0.4`, `WHOLEBODY_MODE = "balanced"`.

---

## 5. What I would actually do, in order

Assuming limited hours and a demo on Saturday.

| # | item | why here |
|---|---|---|
| 1 | §0 merge + re-verify | nothing else is real until the tree has Part I |
| 2 | §4.1 install the optional packages, look at the enriched frame | pure discovery, changes everything downstream, and three models have never run |
| 3 | §1.2 exposure lock + its measurement | lands in a method that already exists; either it shrinks the spread or it is folklore |
| 4 | §1.1 query expansion prompt + seeds | one file, no perception work, directly targets the Pi box |
| 5 | §2a "show it to me" | fixes the badge, no GPU dependency, demo #2 on the list |
| 6 | §4.2 OCR on a real keyboard | the most deterministic demo *once it locks*; needs rehearsal time |
| 7 | §3.2 backend label | 30 minutes, visible |
| 8 | §3.3 per-behaviour threshold | only if counting is going in the script |

Not this week: §2b visual prompts (needs the GPU host and a clear day),
§3.1 role substitution (no pressure on the frame budget yet).

And ENRICH.md's closing advice is worth keeping in front of whoever picks this
up: **skip general complex-query handling**, and lean into the refusals. An
agent that says "I have `hand_raised`, `index_finger_raised`, `open_palm` and
`fist`" reads better than one that bluffs, as long as it names the
alternative. The silent no-op guard was the only version of that which was a
bug, and it is closed.

---

## 6. Open questions

1. **Is anyone at the GPU host before Saturday?** It gates §4.1 and §2b, and
   §4.1 is the highest-information item on the page.
2. **Does counting go in the demo script?** If not, §3.3 drops off entirely.
3. **Who merges?** The camera work is uncommitted here and the merge wants a
   clean tree. If Qinkai is out, someone needs to own §0 — it is an hour of
   care, not a formality, and doing it wrong loses either his four models or
   the camera switching.
