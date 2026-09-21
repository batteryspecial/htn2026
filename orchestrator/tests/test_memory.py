"""The phrase memory: the RAG step, and the misfire it exists to avoid.

The corpus is short, repetitive and full of near-synonyms that mean opposite
things — "a person wearing a dark jacket" and "a person wearing a cream coat"
are neighbours in embedding space. Pure vector search pulls coats back for a
question about ducks, and pure full-text cannot match "duck" to "rubber duck
toy". So retrieval is hybrid, and these tests hold the line on what comes
back first.

They run in full-text mode, because `conftest.py` blanks the embedding model:
that is the offline path, it needs no network, and it is the one that has to
work at a venue. The hybrid path is the same ranking code with a second
retriever fused in.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

from memory.store import PhraseMemory, PhraseRecord


# 1. Setup ------------------------------------------------------------------

def test_it_starts_with_what_the_team_already_measured(memory):
    """A fresh store is not empty: the project's own findings seed it."""
    assert memory.mode == "full-text"

    known = {r.phrase for r in memory.suggest("yellow rubber duck", limit=20)}
    assert "yellow duck" in known
    assert "duck" in known


def test_the_mode_says_how_lookups_are_answered(tmp_path):
    """Shown in the log and by GET /memory, so a silent downgrade is visible."""
    offline = PhraseMemory(uri=str(tmp_path / "a"), embed_model="")

    assert offline.mode == "full-text"


# 2. Ranking ----------------------------------------------------------------

def test_a_question_about_a_duck_does_not_return_coats(memory):
    """The misfire this ranking was rewritten to stop.

    The coat entries carry the strongest evidence in the corpus (1.00, and
    an outcome that worked). Ranking by evidence alone put them at the top of
    a duck query — which is worse than returning nothing, because the agent
    would have tried one.
    """
    phrases = [r.phrase for r in memory.suggest("the yellow duck on the table")]

    assert phrases, "the duck should match something"
    assert not any("coat" in p or "jacket" in p for p in phrases[:3])


def test_the_wording_that_actually_worked_is_offered_first(memory):
    suggestions = memory.suggest("yellow rubber duck", limit=6)

    assert suggestions[0].max_conf > 0.0
    assert suggestions[0].found > 0


def test_something_that_never_acquired_ranks_below_something_that_did(memory):
    memory.remember(PhraseRecord(phrase="a blue mug", subject="the mug",
                                 outcome="never_acquired"))
    memory.remember(PhraseRecord(phrase="a ceramic mug", subject="the mug",
                                 outcome="acquired"))

    phrases = [r.phrase for r in memory.suggest("the mug", limit=6)]

    assert phrases.index("a ceramic mug") < phrases.index("a blue mug")


def test_one_entry_per_phrase_keeping_the_strongest_evidence(memory):
    memory.remember(PhraseRecord(phrase="a steel kettle", subject="the kettle",
                                 max_conf=0.2, found=1))
    memory.remember(PhraseRecord(phrase="a steel kettle", subject="the kettle",
                                 outcome="acquired"))

    kettle = [r for r in memory.suggest("the kettle", limit=6)
              if r.phrase == "a steel kettle"]

    assert len(kettle) == 1
    assert kettle[0].outcome == "acquired"


def test_a_subject_nothing_was_ever_measured_for_returns_nothing(memory):
    assert memory.suggest("a xylophone in a snowstorm") == []


# 3. Writing ----------------------------------------------------------------

def test_a_probe_is_filed_including_the_wordings_that_failed(memory):
    """The zeros are half the value: they stop the next run repeating them."""
    memory.remember_probe("the stapler", [
        {"phrase": "stapler", "max_conf": 0.0, "found": 0},
        {"phrase": "a red stapler", "max_conf": 0.31, "found": 9},
    ])

    filed = {r.phrase: r for r in memory.suggest("the stapler", limit=6)}
    assert filed["stapler"].found == 0
    assert filed["a red stapler"].max_conf == pytest.approx(0.31)
    assert filed["a red stapler"].outcome == "probe"


def test_an_outcome_is_filed_against_every_phrasing_that_was_used(memory):
    memory.remember_outcome("the bicycle", ["bicycle", "a bike"],
                            acquired=True, note="installed as track")

    filed = {r.phrase: r for r in memory.suggest("the bicycle", limit=6)}
    assert filed["bicycle"].outcome == "acquired"
    assert filed["a bike"].note == "installed as track"


def test_a_row_written_now_is_findable_now(memory):
    """The full-text index does not include rows added after it was built.

    Without a rebuild the memory silently stops returning its newest
    measurements, which is the failure mode where it looks like it is working.
    """
    memory.remember(PhraseRecord(phrase="a green watering can",
                                 subject="the watering can", max_conf=0.44,
                                 found=6))

    assert [r.phrase for r in memory.suggest("the watering can")] == \
        ["a green watering can"]


# 4. The prompt fragment ----------------------------------------------------

def test_the_brief_is_labelled_as_a_prior_not_a_fact(memory):
    """Phrasing does not transfer between scenes; the agent must still probe."""
    brief = memory.brief("yellow rubber duck")

    assert "A prior, not a fact" in brief
    assert "probe before" in brief


def test_the_brief_spells_out_the_verdict_for_each_wording(memory):
    memory.remember(PhraseRecord(phrase="a tape measure", subject="the tape measure",
                                 outcome="never_acquired"))
    memory.remember(PhraseRecord(phrase="a yellow tape measure",
                                 subject="the tape measure", max_conf=0.37, found=4))

    brief = memory.brief("the tape measure")

    assert '- "a tape measure" — never acquired' in brief
    assert '- "a yellow tape measure" — found, best score 0.37' in brief


def test_the_brief_names_the_scene_a_number_came_from(memory):
    brief = memory.brief("yellow rubber duck")

    assert "[white table, yoloe-11s]" in brief


def test_nothing_known_means_no_brief_at_all(memory):
    """An empty prior must not become an empty heading in the prompt."""
    assert memory.brief("a xylophone in a snowstorm") == ""


# 5. Degrading --------------------------------------------------------------

def test_without_lancedb_it_still_remembers_within_the_process(memory):
    """Losing the store costs a prior, not the ability to work."""
    memory._table = None

    memory.remember(PhraseRecord(phrase="a paper cup", subject="the cup",
                                 max_conf=0.5, found=3))

    assert memory.mode == "in-process"
    assert [r.phrase for r in memory.suggest("the cup")] == ["a paper cup"]


def test_a_search_that_blows_up_returns_nothing_rather_than_raising(memory):
    """A broken memory must not take the turn down with it."""
    class Exploding:
        def search(self, *_args, **_kw):
            raise RuntimeError("index is corrupt")

    memory._table = Exploding()

    assert memory.suggest("anything at all") == []
    assert memory.brief("anything at all") == ""


def test_unrelated_vector_neighbors_do_not_become_a_prior(memory, monkeypatch):
    """Hybrid/vector rankings also return rows for completely unknown names."""
    neighbors = [
        asdict(PhraseRecord(phrase="rubber duck", subject="yellow rubber duck",
                            max_conf=0.58, found=11)),
        asdict(PhraseRecord(phrase="person", subject="a person",
                            max_conf=0.88, found=10)),
    ]
    monkeypatch.setattr(memory, "_search", lambda *_args: neighbors)

    assert memory.brief("please follow my Walnut") == ""
    assert memory.brief("The pipeline reported: no frame for infs. Event type: camera_lost.") == ""
    assert "rubber duck" in memory.brief("please follow the yellow duck")
    assert '"person"' not in memory.brief("please follow the yellow duck")


def test_generic_words_and_measurement_notes_are_not_subject_matches(memory, monkeypatch):
    monkeypatch.setattr(memory, "_search", lambda *_args: [asdict(PhraseRecord(
        phrase="rubber duck", subject="follow my rubber duck",
        note="measured beside a walnut on the table", max_conf=0.58, found=11))])

    assert memory.suggest("please follow my Walnut") == []


def test_a_learned_name_can_retrieve_the_measured_detector_phrase(memory):
    memory.remember(PhraseRecord(phrase="brown dog", subject="Walnut",
                                 max_conf=0.7, found=4))

    assert [r.phrase for r in memory.suggest("please follow my Walnut")] == ["brown dog"]


def test_in_process_search_matches_subject_words_in_a_full_instruction(memory):
    memory._table = None
    memory.remember(PhraseRecord(phrase="brown dog", subject="Walnut",
                                 max_conf=0.7, found=4))

    assert [r.phrase for r in memory.suggest("please follow my Walnut")] == ["brown dog"]
    assert memory.suggest("please follow my pencil") == []
