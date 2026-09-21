# The behaviour catalogue

Reference for the agent that compiles instructions into perception specs.
Companion to `SELECTORS.md`: a **selector** says what counts as a subject,
a **behaviour** says what to do about it.

Everything here is validated on arrival. `POST /behaviors` builds the
behaviour and throws it away before accepting it, so a malformed `params` is a
422 with a reason, not a 201 followed by silence.

```jsonc
{
  "kind":    "watch",                    // one of the eight below
  "subject": {"detect": ["yellow duck"]},// a Selector — see SELECTORS.md
  "params":  {"triggers": [...]},        // kind-specific, see below
  "render":  {"color": "#FFD400",        // optional, all fields optional
              "label": "the duck",
              "mask": true, "boxes": true, "trail": false},
  "notify":  true                        // false = show it, don't wake the agent
}
```

Only `kind` and `subject` are required.

Breadth comes from combining a selector with a behaviour, **not** from asking
for a kind that does not exist. If an instruction does not fit one of these
eight, it is a query (`/query/count`, `/query/look`, `/describe`) or it is
several behaviours.

---

## One instruction can be several behaviours

"Guard the table: laptop, phone, wallet" is three `watch` behaviours, not one
with three subjects. A behaviour has exactly one subject; running three is how
you guard three things, and each fires its own event with its own label.

Post them in one turn. They are installed between the same two frames.

---

## The kinds

### `highlight` — draw every match, report the count

No params. The default for "detect all X", "show me every X", "highlight X".

```json
{"kind": "highlight", "subject": {"detect": ["person"], "pick": "all"}}
```

Use `pick: "all"`. A highlight with `pick: "largest"` draws one box and is
almost never what was asked for.

---

### `track` — follow one thing, drive the actuator

No params. Emits `acquired`, `lost`, `reacquired`. Drives the guidance arrow.

```json
{"kind": "track", "subject": {"detect": ["yellow duck", "rubber duck"], "pick": "largest"}}
```

`pick` decides the behaviour here more than anything else:

- `pick: "largest"` — "track the duck". Re-picks the biggest match each frame,
  so it will swap between two ducks.
- `pick: "ref"` — "follow **him**", "track **that** one". Latches onto one
  instance and holds it across frames, including out of shot and back. Use it
  for any instruction with *that*, *him*, *her*, *this one*, or an uploaded
  photo (pair with `ref_id`).

States: `ACQUIRING → TRACKING ↔ EDGE → LOST → SEARCHING`.

---

### `watch` — guard a thing, fire when something happens to it

The widest-reach kind on the board. "Watch the duck and tell me if anyone
approaches", "guard the table", "anyone in a red hoodie is authorised" are all
this with different triggers.

| param | type | default | meaning |
|---|---|---|---|
| `triggers` | list, non-empty | `[{"type": "missing", "after_s": 2.0}]` | what fires the alert |
| `cooldown_s` | number | `5.0` | quiet period after firing |

States: `ARMING → ARMED → FIRED → COOLDOWN → ARMED`.

**`ARMING` matters.** The behaviour learns where the subject *is* before it
will judge anything, so "it moved" means moved from where it was when you
pointed at it. It needs the subject steadily visible for ~1 s before it arms,
and stays quiet until then. An `armed` event says it is ready.

#### Trigger types

| `type` | extra fields | fires when |
|---|---|---|
| `missing` | `after_s` (default `2.0`) | the subject has been gone that long |
| `moved` | `min_shift` (default `0.15`) | the subject's centre shifted that far, in normalized units |
| `near` | **`other`** (a Selector, required), `margin` (default `0.1`) | something matching `other` comes within `margin` of the subject |
| `appeared` | **`other`** (a Selector, required) | something matching `other` shows up at all |

`other` is a full selector and obeys every rule in `SELECTORS.md`. This is
where authorisation lives — it is not a separate mechanism:

```jsonc
// "watch the duck, tell me if anyone approaches, but people in red hoodies are allowed"
{
  "kind": "watch",
  "subject": {"detect": ["yellow rubber duck", "rubber duck"], "pick": "largest"},
  "params": {"triggers": [{
    "type": "near",
    "other": {
      "detect":  ["person"],
      "include": ["a person in dark clothing"],
      "exclude": ["a person in a red hoodie"]
    }
  }]}
}
```

An `exclude` on the trigger's own selector is how "X is authorised" is said.

---

### `count_line` — count crossings of a line

| param | type | default | meaning |
|---|---|---|---|
| `line` | `[[x1,y1],[x2,y2]]` | `[[0.5,0.0],[0.5,1.0]]` | normalized 0..1, must be two different points |

Emits `crossed`, and keeps `in` / `out` tallies. Crossings come from each
track's path, not from which side of the line it is on now.

```json
{"kind": "count_line",
 "subject": {"detect": ["person"], "pick": "all"},
 "params": {"line": [[0.5, 0.0], [0.5, 1.0]]}}
```

The default vertical centre line is right for "count people crossing this
line" when no line was indicated. Use `pick: "all"` — counting one thing is
not counting.

---

### `privacy` — blur matches, or everything except a match

| param | type | default | meaning |
|---|---|---|---|
| `mode` | `"blur"` \| `"pixelate"` | `"blur"` | how |
| `keep_ref` | a `ref_id`, or null | `null` | the one match to spare |

```json
// "blur everyone's face"
{"kind": "privacy", "subject": {"detect": ["face", "person's face"], "pick": "all"}}

// "blur everyone's face except mine"  (after POST /references -> r1)
{"kind": "privacy",
 "subject": {"detect": ["face", "person's face"], "pick": "all"},
 "params": {"keep_ref": "r1"}}
```

`keep_ref` fails closed: a face whose appearance has not been scored yet is
blurred, not shown. That is deliberate and is the whole point of the feature.

---

### `pan_to` — guide the view until it has turned far enough

| param | type | default | meaning |
|---|---|---|---|
| `deg` | number | `0.0` | how far to turn; positive pans right |
| `hfov_deg` | number | `70.0` | camera horizontal field of view |
| `tolerance_deg` | number | see config | how close counts as arrived |

States `GUIDING → REACHED`, emits `reached`. Draws an arrow; odometry measures
how far the view has actually moved.

```json
{"kind": "pan_to", "subject": {"detect": ["person"], "pick": "all"}, "params": {"deg": -45}}
```

**This kind is half of a two-step instruction.** "Turn 45° left and count
people" is `pan_to`, then wait for the `reached` event, then
`POST /query/count`. Do not post a `count_line` and hope.

---

### `pose_trigger` — fire on a body pose

| param | type | default | meaning |
|---|---|---|---|
| `gesture` | string | `"hand_raised"` | which pose |
| `cooldown_s` | number | see config | quiet period after firing |
| `hold_frames` | int | see config | frames the pose must hold |

Emits `hand_raised`. The pose model downloads on first use, so the first
instance takes a few seconds longer to start.

```json
{"kind": "pose_trigger", "subject": {"detect": ["person"], "pick": "all"}}
```

---

### `keyboard` — find a keyboard and guide typing

**Not built.** `GET /behaviors` reports it under `kinds.planned`. Posting it is
a 422. Do not emit it.

---

## Not behaviours

Some instructions are a question, not a standing order. These are one HTTP
call and a reply, with nothing installed:

| Instruction | Call |
|---|---|
| "how many people are there?" | `POST /query/count` with a selector |
| "what can you see?" | `POST /query/look` |
| "what's on the table right now?" | `POST /describe` — its own vision model |
| "switch to the COCO model" | `POST /model` |
| "stop" / "clear everything" | `DELETE /behaviors` |
| "stop watching the duck" | `DELETE /behaviors/{id}` |

Posting a `highlight` to answer "how many people are there?" is wrong: it
installs a standing behaviour to answer a one-off question, and the count it
reports is one frame rather than a median over a window.

---

## `render`

The agent picks colours and which layers to turn on. It never writes drawing
code — generated drawing code is the fastest way to crash on stage.

| field | default | notes |
|---|---|---|
| `color` | a palette slot | `"#FFD400"`. Leave unset unless the operator named a colour |
| `label` | none | a short human-readable name; shown on the HUD and in the trace |
| `mask` | `true` | segmentation mask |
| `boxes` | `true` | bounding boxes |
| `trail` | `false` | path history; useful for `track`, noise elsewhere |

Always set `label` to something the operator would recognise — it is what the
behaviours panel shows, and "green water bottle" reads better than `b8`.

With two behaviours running, leave `color` unset: the palette assigns distinct
slots automatically, and two hand-picked colours are more likely to collide.

---

## Checklist before posting

1. Is this actually a behaviour, or is it a query? (see **Not behaviours**)
2. Is `kind` one of the seven built ones — never `keyboard`?
3. Does the subject follow `SELECTORS.md`? Two to four `detect` phrasings,
   `include` paired with `exclude`, relate classes also in `detect`.
4. Is `pick` right? `all` for counting / highlighting / blurring, `ref` for
   *that one*, `largest` otherwise.
5. Does every `near` / `appeared` trigger carry an `other` selector?
6. Is `render.label` set to something a person would recognise?
7. Several things named → several behaviours, posted together.
