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
    {"at": 1.0, "add": {"behavior_id": "count", "kind": "highlight",
                        "subject": {"detect": ["person"]}}},
    {"at": 6.0, "add": {"behavior_id": "red", "kind": "track",
                        "subject": {"detect": ["person"],
                                    "include": ["a person wearing red shoes"]}}},
    {"at": 12.0, "remove": "count"},
    {"at": 14.0, "model": "coco"}
  ],
  "expect": {
    "count": {"reaches": "active", "min_matches": 1, "acquire_under_ms": 1500},
    "red":   {"reaches": "active", "max_matches": 1, "events": ["acquired"]}
  }
}
```

`expect` keys: `reaches`, `min_matches`, `max_matches`, `acquire_under_ms`,
`events`. Leave `expect` out to just watch what happens.
