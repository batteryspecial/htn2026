"""The measurements this project already made, as the memory's starting state.

Every number here was measured on real footage and is recorded in
`documents/SELECTORS.md`, `PIPELINE-DIAGNOSIS.md` or `FOR-QINKAI.md`. Seeding
them means the agent starts with the team's hard-won knowledge instead of
rediscovering that "duck" finds nothing.

Two independent measurements agreed on the duck, from opposite ends of the
system — the detector side (12/12 vs 0/12) and the compile side (0.22 vs
0.00). That is the strongest evidence in the corpus, and the clearest
illustration of the rule: the obvious noun is often the worst choice.
"""

from __future__ import annotations

from memory.store import PhraseRecord


def _seed(phrase: str, subject: str, max_conf: float, found: int,
          scene: str, note: str) -> PhraseRecord:
    return PhraseRecord(phrase=phrase, subject=subject, max_conf=max_conf,
                        found=found, outcome="seed", scene=scene, note=note)


SEED: list[PhraseRecord] = [
    # --- the duck, measured twice from opposite ends of the system --------
    _seed("duck", "yellow rubber duck", 0.00, 0,
          "white table, yoloe-11s",
          "the obvious noun found nothing in 12 frames; measured at the "
          "detector and again at the compiler"),
    _seed("bird", "yellow rubber duck", 0.00, 0,
          "white table, yoloe-11s", "a category above the object; no match"),
    _seed("rubber duck", "yellow rubber duck", 0.58, 11,
          "white table, yoloe-11s", "11 of 12 frames"),
    _seed("yellow duck", "yellow rubber duck", 0.22, 12,
          "white table, yoloe-11s",
          "12 of 12 frames; the adjective is doing the work"),
    _seed("toy", "yellow rubber duck", 0.50, 12,
          "white table, yoloe-11s", "the general category also landed"),
    _seed("yellow rubber duck", "yellow rubber duck", 0.40, 8,
          "classroom, yoloe-11s",
          "8 of 12 in a room where every shorter phrase found nothing — "
          "phrasing does not transfer between scenes"),
    _seed("duck, rubber duck, toy", "yellow rubber duck", 0.62, 12,
          "white table, yoloe-11s",
          "the union of three phrasings beat every one alone, on both count "
          "and confidence; detection runs once on the union, so this is free"),

    # --- faces ------------------------------------------------------------
    _seed("face", "a person's face", 0.00, 0,
          "yoloe-11s", "bare noun, no match"),
    _seed("person's face", "a person's face", 0.48, 10,
          "yoloe-11s",
          "the possessive is load-bearing; scored 0.12-0.28 on another run, "
          "which is under ByteTrack's stock activation gate"),

    # --- the reliable one -------------------------------------------------
    _seed("person", "a person", 0.88, 10,
          "yoloe-11s",
          "0.85-0.88, the only class that clears every threshold comfortably; "
          "testing with a person hides threshold bugs"),

    # --- absent-object noise floor ---------------------------------------
    _seed("yellow duck", "yellow duck that is not in shot", 0.14, 0,
          "yoloe-11s, object absent",
          "0.10-0.14 with nothing there against 0.18-0.27 with the duck "
          "present; about 0.06 between signal and noise, which is why "
          "dropping thresholds further buys false locks on wall texture"),

    # --- clothing attributes, from the street clip -----------------------
    _seed("a person wearing a dark jacket", "a person in a dark jacket", 0.99, 1,
          "street clip, CLIP",
          "correct on the man actually wearing one; the man in a cream coat "
          "also scored 0.78 for this, which is over any sane threshold and "
          "wrong — pair it with an exclude so the decision is which phrase "
          "wins rather than whether one clears a bar"),
    _seed("a person wearing a cream coat", "a person in a light coat", 1.00, 1,
          "street clip, CLIP",
          "1.00 against 0.01 for the dark-jacket man; the ranking was right "
          "where the absolute number was not"),
    _seed("a person with red shoes", "a person in red shoes", 0.00, 0,
          "CLIP, upper-body crop",
          "clothing is scored on the upper body, so shoes are not in the crop "
          "being scored; use relate {contains: shoe} instead"),
]
