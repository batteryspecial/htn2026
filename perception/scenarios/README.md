# Scenarios

One JSON file per real-clip test. Record the clip, write the file, run:

```bash
python scripts/clip_test.py scenarios/*.json --record out/
```

Exits non-zero if reality disagrees with `expect`, so this is a pre-demo
checklist rather than something to read.

```jsonc
{
  "name": "follow the person in red",
  "clip": "clips/red_shoes.mp4",   // path, relative to perception/
  "model": "yoloe",                 // open vocabulary unless you need speed
  "duration_s": 20,                 // how long to run; the clip loops
  "steps": [
    {"at": 1.0, "add": {"kind": "highlight",
                        "subject": {"detect": ["person"]}}},
    {"at": 6.0, "add": {"kind": "track",
                        "subject": {"detect": ["person"], "pick": "ref",
                                    "include": ["a person wearing red shoes"],
                                    "exclude": ["a person wearing dark shoes"]}}},
    {"at": 12.0, "remove": "b1"},
    {"at": 14.0, "model": "coco"}
  ],
  "expect": {
    "b1": {"reaches": "ACTIVE", "min_matches": 1, "acquire_under_ms": 1500},
    "b2": {"reaches": "TRACKING", "max_matches": 1, "events": ["acquired"]}
  }
}
```

Ids are assigned by the server in order, so the first `add` is `b1`, the
second `b2`, and `expect` and `remove` refer to those.

`expect` keys: `reaches`, `min_matches`, `max_matches`, `acquire_under_ms`,
`events`. Leave `expect` out to just watch what happens.

## Attributes

Pair `include` with `exclude` wherever you can. A phrase judged against a
competitor is far steadier than one judged against an absolute bar, because
lighting moves every score together and a comparison cancels that out.

Before writing a scenario for a new clip, find the numbers:

```bash
python scripts/calibrate.py clips/duck.mp4 \
    "a yellow duck" "a brown duck" --detect duck --dump out/
```

It prints each track's scores, suggests a `min_score`, and saves the crops so
you can check the ranking by eye. If the tracks do not split into two clear
groups, the phrase is not discriminating: put the attribute straight into the
detector prompt instead, e.g. `detect: ["yellow duck"]`.
