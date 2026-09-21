# How to phrase a selector

Reference for the agent that compiles instructions into perception specs.
Every claim here was measured on real footage from this project; the numbers
are included because they are the argument.

A **selector** says what counts as a subject. It is the only place an
instruction turns into something the camera can find, and almost every failure
in this system is a badly phrased selector rather than a broken pipeline.

```jsonc
{
  "detect":  ["yellow rubber duck"],   // what the detector hunts for
  "include": ["a duck that is yellow"],// CLIP: must look like this
  "exclude": ["a brown duck"],         // CLIP: must not look like this
  "relate":  {"contains": "shoe"},     // must physically contain this
  "ref_id":  "r1",                     // must look like this uploaded photo
  "pick":    "largest"                 // which one, when several match
}
```

Only `detect` is required.

---

## Rule 1 — Phrase `detect` the way a person would say it, not as a dataset label

The detector is open-vocabulary. It matches your words against the image; it
does not look up a class in a table. Plain nouns are often the *worst* choice
because they are ambiguous across the whole visual world.

Measured on a clip of a rubber duck on a table:

| `detect` | frames found |
|---|---|
| `"duck"` | **0 / 12** |
| `"bird"` | **0 / 12** |
| `"rubber duck"` | 11 / 12 |
| `"yellow duck"` | **12 / 12** |
| `"toy"` | 12 / 12 |

The obvious noun found nothing. The descriptive phrase found it in every
frame. This is the single most common cause of a behaviour that starts
cleanly and then never matches anything.

**Do:** `["yellow rubber duck"]`, `["blue water bottle"]`, `["black backpack"]`
**Don't:** `["duck"]`, `["bottle"]`, `["object"]`, `["it"]`

---

## Rule 2 — Send several phrasings at once, because the union is free

Detection runs **once per frame on the union of every phrase**, so three
phrasings cost exactly what one costs. Sending several converts a silent miss
into a hit.

Measured on the same duck clip:

| `detect` | frames found | confidence |
|---|---|---|
| `["rubber duck"]` | 11 / 12 | 0.58 |
| `["duck", "rubber duck", "toy"]` | **12 / 12** | **0.62** |

The combined list beat every phrase alone, on both count and confidence.

**Send two to four phrasings** spanning specific to general:

```json
{"detect": ["yellow rubber duck", "rubber duck", "duck toy", "toy"]}
```

---

## Rule 3 — Phrasing is scene-dependent, so never assume a phrase transfers

The same physical duck, filmed in two rooms:

| `detect` | clip A (on a white table) | clip B (in a classroom) |
|---|---|---|
| `"yellow duck"` | **12 / 12** | **0 / 12** |
| `"rubber duck"` | 11 / 12 | **0 / 12** |
| `"toy"` | 12 / 12 | **0 / 12** |
| `"yellow rubber duck"` | — | **8 / 12** |

Every phrase that worked perfectly in one room found nothing in the other.
Lighting, distance and background all move the match.

**Consequence:** a phrase that worked before is a good first guess and nothing
more. Always send a spread (Rule 2), and always be ready to rephrase (Rule 7).

---

## Rule 4 — Adjectives go in `detect` to *identify*, in `include`/`exclude` to *discriminate*

This is the distinction that decides whether a spec works.

**One object, described** → put the description in `detect`. The adjective
helps the detector find it at all.

```json
{"detect": ["yellow rubber duck", "rubber duck"]}
```

**Several of the same kind, one singled out** → detect the plain class, then
discriminate with CLIP. A long phrase in `detect` will either find nothing or
find all of them.

```json
{
  "detect":  ["person"],
  "include": ["a person wearing a red hoodie"],
  "exclude": ["a person wearing a dark jacket"]
}
```

Getting this backwards is the classic failure:

- `{"detect": ["person in a red hoodie"]}` — the detector is being asked to do
  a job CLIP does better, and usually returns nothing.
- `{"detect": ["duck"], "include": ["a yellow duck"]}` — the attribute check
  never runs, because `detect` found nothing to check.

---

## Rule 5 — Always pair `include` with `exclude`

CLIP's raw scores drift with lighting, so an absolute threshold is fragile.
A comparison between two phrases is not, because the drift moves both.

Measured on a street clip, scoring two people against two phrases:

| person | "wearing a dark jacket" | "wearing a cream coat" |
|---|---|---|
| man in a **cream coat** | **0.78** | **1.00** |
| man in a **dark jacket** | **0.99** | 0.01 |

The cream-coat man scored 0.78 for *dark jacket* — over any sane threshold,
and wrong. But he scored 1.00 for *cream coat*. **The ranking was correct even
where the absolute number was not.**

So when both are present the decision is *which phrase wins*, which is far
steadier than any cutoff. A lone `include` or a lone `exclude` falls back to an
absolute bar and is materially less reliable.

```json
// weak — absolute threshold
{"detect": ["person"], "exclude": ["a person in a dark jacket"]}

// strong — a competition
{"detect": ["person"],
 "include": ["a person in light coloured clothing"],
 "exclude": ["a person in a dark jacket"]}
```

Write the `exclude` as the natural opposite of the `include`. If you cannot
think of an opposite, the attribute probably belongs in `detect` instead
(Rule 4).

---

## Rule 6 — Describe the upper body for clothing

Clothing attributes are scored on the upper portion of a person, not the whole
body, because a full-body crop is mostly trousers, shoes and background. Phrase
accordingly: *"a person in a red hoodie"* works; *"a person with red shoes"*
does not, because the shoes are not in the crop being scored.

For anything below the waist, use `relate` instead:

```json
{"detect": ["person", "shoe"], "relate": {"contains": "shoe"}}
```

`relate` keeps a subject only when the named thing is found low inside it. It
is geometric, not semantic, so it is far steadier than asking CLIP about a
whole body. Every class named in `relate` must also appear in `detect`.

---

## Rule 7 — When nothing matches, rephrase; do not wait

A selector that matches nothing for about four seconds reports it on its own
status:

```
"nothing matches 'duck' — try a more specific phrase"
```

Treat this as an instruction to you, not a pipeline error. It is the expected
outcome of Rule 3, and the recovery is mechanical:

1. **Widen the spread.** Add a more specific phrase and a more general one:
   `["duck"]` → `["yellow rubber duck", "rubber duck", "duck toy", "toy"]`
2. **Move the adjective.** If the attribute is in `include`, try it in
   `detect` instead — the check cannot run if nothing was detected (Rule 4).
3. **Drop to a category.** `"toy"`, `"bottle"`, `"bag"` often land when a
   specific phrase does not.
4. **Ask the operator** only after the above. "I cannot find the duck — is it
   in shot?" is a reasonable third move, not a first one.

A behaviour that is `ACTIVE` with zero matches is *not* working. Check the
match count, not just the state.

---

## Rule 8 — Check the vocabulary before choosing a fixed model

`GET /models` reports, per model, whether it is open-vocabulary and what
classes it knows.

- **Open-vocabulary** (the default): any phrase is legal. Rules 1–3 apply.
- **Fixed vocabulary** (COCO-style): only its own class names are legal, and
  an unknown one is refused with the list. Descriptive phrasing does not help
  here — `"yellow rubber duck"` is simply not in the vocabulary.

Switching to a fixed model **pauses** any behaviour whose classes it cannot
see, with a reason, and resumes it when a capable model returns. A paused
behaviour is not a failure; do not restart it.

---

## Rule 9 — `pick` decides which one, and `ref` means *that* one

| `pick` | meaning | use when |
|---|---|---|
| `all` | every match | counting, highlighting, blurring |
| `largest` | biggest box | "the duck", "the nearest person" |
| `most_centered` | closest to frame centre | "the one in the middle" |
| `ref` | latch onto one instance and keep it | "follow **him**", "track **that** one" |

`ref` is the only one with memory. It holds a single instance across frames and
will not swap to a different object of the same class — which is what makes
"follow that person" survive them walking out of frame and back. Use it for any
instruction with *that*, *him*, *her*, *this one*, or a reference photo.

Pair it with a `ref_id` from `POST /references` when the operator uploaded a
photo.

---

## Worked examples

| Instruction | Selector |
|---|---|
| "detect all humans" | `{"detect": ["person"], "pick": "all"}` |
| "watch the yellow duck" | `{"detect": ["yellow rubber duck", "rubber duck", "duck toy"], "pick": "largest"}` |
| "follow him" | `{"detect": ["person"], "pick": "ref"}` |
| "track that one" (photo uploaded) | `{"detect": ["person"], "ref_id": "r1", "pick": "ref"}` |
| "the person in the red hoodie" | `{"detect": ["person"], "include": ["a person in a red hoodie"], "exclude": ["a person in a dark jacket"], "pick": "largest"}` |
| "anyone not wearing a black jacket" | `{"detect": ["person"], "include": ["a person in light coloured clothing"], "exclude": ["a person in a black jacket"], "pick": "all"}` |
| "the person in red shoes" | `{"detect": ["person", "red shoe", "shoe"], "relate": {"contains": "shoe"}, "pick": "largest"}` |
| "highlight anything red" | `{"detect": ["red object", "red item"], "include": ["a red object"], "exclude": ["a blue object"], "pick": "all"}` |
| "count people crossing the line" | `{"detect": ["person"], "pick": "all"}` |
| "blur everyone's face except mine" | `{"detect": ["face", "person's face"], "ref_id": "r1", "pick": "all"}` |
| "now follow the dog" | `{"detect": ["dog", "small dog"], "pick": "ref"}` |

---

## Checklist before sending

1. Is every `detect` entry a phrase a person would say, not a bare dataset noun?
2. Are there **two to four** phrasings, specific through general?
3. If an attribute is present: is it in `detect` (identifying) or
   `include`+`exclude` (discriminating)? Rule 4.
4. If `include` is set, is there a matching `exclude`? Rule 5.
5. Is every class named in `relate` also in `detect`?
6. Does the instruction say *that one* / *him* / *her*? Then `pick: "ref"`.
7. If a fixed-vocabulary model is active, is every class in its list?

---

## Things that are never worth putting in `detect`

- Pronouns: `"it"`, `"him"`, `"them"` — resolve to the concrete noun.
- Slang: `"guy"`, `"dude"`, `"kid"` — every human is `"person"`.
- Bare abstractions: `"object"`, `"thing"`, `"item"` on their own.
- Whole sentences: `"the person who just walked in wearing a red hoodie"`.
  Split it: class into `detect`, appearance into `include`/`exclude`, and the
  rest is not a selector at all.
- Counts: `"two people"` — a selector describes *a kind*; `pick` and the
  behaviour decide how many.
