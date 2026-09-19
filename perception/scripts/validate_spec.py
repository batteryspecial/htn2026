"""Check whether what the orchestrator produced is something perception accepts.

    python scripts/validate_spec.py plan.json
    cat plan.json | python scripts/validate_spec.py
    python scripts/validate_spec.py plan.json --live http://gpu-box:8001

Takes one behaviour, a list of them, or a full {instruction_id, behavior} body,
and answers four questions without starting anything:

1. Does it parse against the shared contract in linker/schemas.py?
2. Is the behaviour kind one this build actually implements?
3. Does the model exist and can this machine run it?
4. If the model has a fixed vocabulary, can it produce these classes?

Use it to close the loop on the agent: feed it the LLM's output and see
whether the camera would have understood. With --live it asks a running
service instead of the local registry, which is what to do before a demo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import ValidationError  # noqa: E402

from behaviors.kinds import BUILT, KINDS  # noqa: E402
from contracts import Behavior  # noqa: E402

OK, BAD, WARN = "  ok  ", " FAIL ", " warn "


def load(argv: list[str]) -> list[dict]:
    """Accept a file or stdin, and one behaviour or many."""
    paths = [a for a in argv[1:] if not a.startswith("-")]
    raw = Path(paths[0]).read_text() if paths else sys.stdin.read()
    data = json.loads(raw)
    if isinstance(data, dict):
        data = [data]
    # Unwrap {instruction_id, behavior} bodies so either form works.
    return [d.get("behavior", d) if isinstance(d, dict) else d for d in data]


def local_models() -> dict:
    from detectors.registry import Registry

    reg = Registry.from_yaml()
    return {m["name"]: m for m in reg.manifest()}


def live_models(base: str) -> dict:
    import httpx

    d = httpx.get(f"{base.rstrip('/')}/models", timeout=10).json()
    return {m["name"]: m for m in d["models"]}


def check(raw: dict, models: dict) -> list[tuple[str, str]]:
    findings = []
    try:
        b = Behavior(**raw)
    except ValidationError as e:
        for err in e.errors():
            loc = ".".join(str(x) for x in err["loc"]) or "(root)"
            findings.append((BAD, f"{loc}: {err['msg']}"))
        return findings
    findings.append((OK, f"parses as a Behavior: {b.behavior_id!r} ({b.kind})"))

    if b.kind in BUILT:
        findings.append((OK, f"kind {b.kind!r} is implemented"))
    elif b.kind in KINDS:
        findings.append((BAD, f"kind {b.kind!r} is in the contract but not built yet; "
                              f"available: {sorted(BUILT)}"))
    else:
        findings.append((BAD, f"kind {b.kind!r} is not in the contract"))

    # Model choice is not part of a Behavior; it is set separately. Only the
    # vocabulary of whatever model is loaded can be checked here.
    fixed = {n: m for n, m in models.items() if m.get("classes")}
    if not fixed:
        findings.append((OK, "no fixed-vocabulary model loaded, any class name is fine"))
    for name, m in fixed.items():
        missing = [c for c in b.subject.prompts() if c not in m["classes"]]
        if missing:
            findings.append((WARN, f"{name} cannot produce {missing} "
                                   f"(fine unless you switch to it)"))
        else:
            findings.append((OK, f"{name} can produce {b.subject.prompts()}"))

    if b.subject.needs_attributes:
        findings.append((OK, f"attributes: include={b.subject.include} "
                             f"exclude={b.subject.exclude} (needs CLIP loaded)"))
    if b.subject.reference_ids:
        findings.append((WARN, "reference_ids set; image references are not built yet"))
    return findings


def main() -> int:
    live = None
    if "--live" in sys.argv:
        live = sys.argv[sys.argv.index("--live") + 1]
    try:
        behaviors = load(sys.argv)
    except Exception as exc:
        print(f"{BAD} could not read input: {exc}")
        return 2

    models = live_models(live) if live else local_models()
    print(f"checked against {'live ' + live if live else 'local models.yaml'}: "
          f"{sorted(models)}\n")

    failed = 0
    for i, raw in enumerate(behaviors):
        print(f"[{i}] {json.dumps(raw)[:100]}")
        for mark, line in check(raw, models):
            print(f" {mark} {line}")
            failed += mark == BAD
        print()
    print("VERDICT:", "perception would accept all of these" if not failed
          else f"{failed} problem(s); perception would refuse")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
