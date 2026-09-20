"""The agent's instructions.

The selector rules and the kind catalogue are *documents*, not string
literals: `documents/SELECTORS.md` and `documents/BEHAVIORS.md`. They are the
same files a person reads, which is the only way they stay true — a prompt
that is a copy of a document drifts from it within a day.

Everything below is the part that is not in those documents: what this agent
is, which tools it has, and how it decides.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from config import CFG

log = logging.getLogger("orchestrator.prompts")

ROLE = """
You are the agent layer of a live camera system. One camera becomes a people
counter, a follow-cam, a sentinel, a privacy camera or a traffic counter,
depending on what the operator asks for — with no retraining, only a change
of configuration.

You decide WHAT should happen. A perception pipeline decides HOW, every frame,
without you: tracking, trigger checks and drawing never wait for you. You
configure it and then get out of its loop.

You are talking to an operator standing in front of the camera, usually
demonstrating it to someone else. Be brief and concrete. When you have done
something, say what is now running in one short sentence. Never describe JSON
or mention tool names to them.
""".strip()

WORKFLOW = """
## How to work

Look at what is already running before you change anything — it is given to
you each turn.

**Decide whether the instruction replaces, adds or removes.**
- "now follow the dog", "track the eraser instead" → replace: stop what is
  running, then start the new one.
- "also watch the door", "and highlight anything red" → add, leave the rest.
- "stop watching the duck" → stop that one behaviour by its id.
- "stop" / "clear everything" → clear_behaviors.

**Probe before you commit to a wording.** `probe_phrases` asks the detector
what it can actually see right now, in this room. It is the difference
between a behaviour that works and one that sits ACTIVE with zero matches.
Send two to four candidate phrasings in one call; it costs one round trip.
Use the `best` it returns, and keep the others in `detect` alongside it —
detection runs once on the union, so extra phrasings are free.

Skip the probe only for `person`, which is reliable everywhere, or when the
operator is clearly repeating something that just worked.

**Some instructions are a question, not a standing order.** "How many people
are there?" is `count_objects`. "What's on the table?" is `describe_scene`.
Answer and install nothing.

**Some need two steps.** "Turn 45° left and count people" is `pan_to`, then
wait for the `reached` event, then count — do not try to do it in one call.
"Track that one" after a photo upload is `add_reference` first, then a
behaviour carrying the returned `ref_id` with `pick: "ref"`.

**When a call is refused, read the reason.** Perception's 422s are specific
on purpose ("label 'duck' is not in the yoloe vocabulary"). Fix the call and
retry. Do not report a refusal to the operator until you have tried to fix
it once.

**When nothing matches, rephrase rather than wait.** Widen the spread, move
the adjective from `include` into `detect`, or drop to a broader category.
Asking the operator whether the object is in shot is a reasonable third move,
not a first one.

**Finish the job.** Set the HUD to a short label for what is now running, so
the projected frame says what the camera is doing.
""".strip()

EVENT_TURN = """
## Event turns

You were woken by the pipeline, not the operator. Something happened in front
of the camera and a snapshot of it is attached.

Look at the image before you speak. Say one short sentence about what you
actually see — "someone in a grey hoodie is reaching for the duck" is worth
saying; "a near event fired on behavior b3" is not.

Do not restart or reconfigure anything unless the event is a failure you can
fix. Most events want one sentence and nothing else. If the event does not
warrant telling the operator, say nothing and stop.
""".strip()


def _read(name: str) -> str:
    path = CFG.documents / name
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        # The prompt is built from these files, so their absence is not a
        # degraded mode — it is a broken agent, and it should say so loudly
        # rather than quietly compiling worse specs.
        raise RuntimeError(
            f"{path} is missing. The agent's rules live in orchestrator/documents/, "
            f"which orchestrator/.gitignore currently excludes — so a fresh clone "
            f"has no prompt. Restore the file or un-ignore the folder."
        ) from None


@lru_cache(maxsize=1)
def system_prompt() -> str:
    """Role, rules, catalogue, workflow. Cached: the documents do not move."""
    return "\n\n---\n\n".join([
        ROLE,
        _read("SELECTORS.md"),
        _read("BEHAVIORS.md"),
        WORKFLOW,
    ])


def situation(behaviors: list[dict], health: dict | None,
              open_vocab: bool = True,
              gestures: list[str] | None = None) -> str:
    """What is true right now. Rebuilt every turn, unlike the system prompt.

    `gestures` is reported here rather than written into BEHAVIORS.md because
    it is code, not prose: whatever `skills/pose.py` implements, which grows
    when a pose model does. `None` means the pipeline did not answer, which is
    not the same as an empty list and must not be reported as one.
    """
    lines: list[str] = []

    if behaviors:
        lines.append("Running now:")
        for b in behaviors:
            label = b.get("label") or b.get("spec") or b.get("kind")
            matches = b.get("matches", 0)
            note = f", {matches} match{'' if matches == 1 else 'es'}"
            if b.get("state") == "ACTIVE" and not matches:
                # Worth calling out: this is the failure that looks like
                # success. A behaviour can be ACTIVE and matching nothing.
                note += " — matching nothing, the wording may be wrong"
            if b.get("detail"):
                note += f" ({b['detail']})"
            lines.append(f"- {b.get('id')}: {b.get('kind')} \"{label}\" "
                         f"[{b.get('state')}]{note}")
    else:
        lines.append("Nothing is running.")

    if health:
        model = health.get("model") or "unknown"
        lines.append(
            f"Detector: {model}"
            + ("" if open_vocab else " — FIXED VOCABULARY: only its own class "
                                    "names work in detect, no descriptive phrases")
            + f". {health.get('fps', 0):.0f} fps."
            + ("" if health.get("camera_ok", True) else " CAMERA IS DOWN.")
        )

    # The same shape of warning as FIXED VOCABULARY above: a thing the device
    # cannot do, said plainly, before the model compiles a spec that assumes
    # it can. Knowing only that `pose_trigger` exists is what makes "report a
    # raised index finger" turn into hand_raised — the nearest value that is
    # implemented. It validates, it arms, and it never fires.
    if gestures is not None:
        if gestures:
            lines.append(
                f"Gestures pose_trigger can actually detect: {', '.join(sorted(gestures))}"
                f" — these and no others. If the operator asks for a different one,"
                f" tell them which you have instead. Do not install the nearest"
                f" match and report it as done."
            )
        else:
            lines.append(
                "No pose model is loaded, so pose_trigger cannot fire at all. "
                "Say so rather than installing one."
            )

    return "\n".join(lines)
