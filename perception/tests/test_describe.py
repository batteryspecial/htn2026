"""describe(): answering an open question about the scene.

The shape of this is different from every other query. "What's on the table"
names no subject, and an open-vocabulary detector only finds what it is named,
so there is nothing to look for until something supplies candidate words.

Two halves, tested separately: the grounding, which turns geometry into
language, and the sweep, which turns "what is there" into "which of these is
there". Neither is a language model, and perception deliberately does not call
one.
"""

import pytest
from fastapi.testclient import TestClient

from contracts import TrackView
from scene import (
    SCENE_VOCAB,
    build_prompt,
    candidates_for,
    size_of,
    summarize,
    to_objects,
    where,
)
from server.api import Service, create_app
from tests.rig import Rig, boxes

THING = ("thing", 320, 240, 100)


def track(label="laptop", cx=0.0, cy=0.0, area=0.1, conf=0.9):
    return TrackView(track_id=1, label=label, conf=conf, cx=cx, cy=cy, area=area)


# 1. Spatial language ------------------------------------------------------
@pytest.mark.parametrize("cx,cy,expected", [
    (0.0, 0.0, "centre"),
    (-0.8, 0.0, "far left"),
    (-0.3, 0.0, "left"),
    (0.3, 0.0, "right"),
    (0.8, 0.0, "far right"),
    (0.0, -0.6, "top centre"),
    (0.0, 0.6, "bottom centre"),
    (-0.8, 0.6, "bottom far left"),
])
def test_positions_become_phrases_a_person_would_use(cx, cy, expected):
    assert where(cx, cy) == expected


def test_the_phrasing_is_coarse_on_purpose():
    """"left" has to survive the box jitter that "x=-0.31" does not."""
    assert where(-0.30, 0.0) == where(-0.34, 0.02)


@pytest.mark.parametrize("area,expected", [
    (0.5, "very large"), (0.1, "large"), (0.02, "medium"), (0.001, "small")])
def test_area_becomes_a_size_word(area, expected):
    assert size_of(area) == expected


# 2. Grouping and summarising ---------------------------------------------
def test_identical_things_in_one_place_are_counted_once():
    """Three people together is one thing to say, not three."""
    objects = to_objects([track("person", -0.7, 0.0, 0.05),
                          track("person", -0.72, 0.02, 0.05),
                          track("person", -0.69, 0.01, 0.04)])
    assert len(objects) == 1
    assert objects[0].count == 3


def test_the_same_thing_in_different_places_stays_separate():
    objects = to_objects([track("person", -0.7), track("person", 0.7)])
    assert len(objects) == 2


def test_the_biggest_thing_is_mentioned_first():
    objects = to_objects([track("mug", 0.5, 0.0, 0.01), track("laptop", 0.0, 0.0, 0.2)])
    assert objects[0].label == "laptop"


def test_the_summary_is_a_sentence():
    text = summarize(to_objects([track("laptop", 0.0, 0.0, 0.2),
                                 track("mug", 0.6, 0.4, 0.01)]))
    assert text.startswith("I can see") and text.endswith(".")
    assert "laptop" in text and "mug" in text


def test_the_summary_pluralises():
    text = summarize(to_objects([track("person", -0.7), track("person", -0.71)]))
    assert "two people" in text


def test_an_empty_scene_says_so_rather_than_nothing():
    assert "cannot see" in summarize([])


# 3. The prompt handed to the vision model --------------------------------
def test_the_prompt_carries_the_question_and_the_grounding():
    prompt = build_prompt("What is on the table?",
                          to_objects([track("laptop", 0.0, 0.0, 0.2)]), [])
    assert "What is on the table?" in prompt
    assert "laptop" in prompt and "centre" in prompt


def test_the_prompt_admits_the_detector_list_is_incomplete():
    """Otherwise the model trusts it over its own eyes and misses everything
    the sweep had no word for."""
    prompt = build_prompt("What do you see?", to_objects([track()]), [])
    assert "not exhaustive" in prompt


def test_an_empty_scene_tells_the_model_to_use_the_image():
    prompt = build_prompt("What do you see?", [], [])
    assert "from the image alone" in prompt


def test_the_prompt_mentions_what_the_camera_is_doing():
    prompt = build_prompt("What do you see?", [], ["guarding the duck"])
    assert "guarding the duck" in prompt


# 4. Candidate vocabulary --------------------------------------------------
def test_the_default_vocabulary_is_used_when_none_is_given():
    assert candidates_for(None) == list(SCENE_VOCAB)
    assert candidates_for([]) == list(SCENE_VOCAB)


def test_a_caller_vocabulary_wins():
    assert candidates_for(["duck", "duck", "mug"]) == ["duck", "mug"]


def test_the_vocabulary_is_capped():
    """A sweep is one forward pass; an unbounded prompt list is not."""
    assert len(candidates_for([f"thing{i}" for i in range(200)])) == 64


def test_the_default_vocabulary_is_phrased_descriptively():
    """SELECTORS.md, measured: bare nouns find nothing."""
    assert "computer keyboard" in SCENE_VOCAB and "keyboard" not in SCENE_VOCAB
    assert "coffee mug" in SCENE_VOCAB and "mug" not in SCENE_VOCAB


# 5. The endpoint ----------------------------------------------------------
@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.close()


@pytest.fixture
def client(rig):
    svc = Service(registry=rig.registry, capture=rig.capture, builder=rig.builder,
                  loop=rig.loop, health=rig.health, events=rig.event_bus,
                  states=rig.state_bus)
    with TestClient(create_app(svc, run_threads=False)) as c:
        c.rig = rig
        yield c


def test_describe_returns_a_snapshot_and_a_prompt(client):
    client.rig.frames(2, seen=boxes(THING))
    d = client.post("/describe", json={"question": "What is on the table?"}).json()
    assert d["snapshot_url"] == "/snapshot"
    assert "What is on the table?" in d["prompt"]
    assert d["summary"]


def test_describe_sweeps_for_things_no_behavior_asked_about(client):
    """The whole point: an open question names nothing, so something has to
    supply the words."""
    client.rig.frames(2, seen=boxes(THING))
    d = client.post("/describe", json={"candidates": ["thing"]}).json()
    assert d["swept"] is True
    assert [o["label"] for o in d["objects"]] == ["thing"]


def test_the_sweep_does_not_disturb_what_the_camera_is_tracking(client):
    """The reason it runs on a second detector. Re-pointing the live one would
    break every running behaviour for the sake of a question."""
    b = client.post("/behaviors", json={
        "kind": "track", "subject": {"detect": ["thing"], "pick": "ref"}}).json()["id"]
    client.rig.settle()
    client.rig.frames(6, seen=boxes(THING))
    assert client.rig.state_of(b) == "TRACKING"
    before = client.rig.view(b).data["track_id"]
    active_detector = client.rig.loop.world.detector

    client.post("/describe", json={"candidates": ["laptop", "coffee mug"]})

    client.rig.frames(4, seen=boxes(THING))
    assert client.rig.state_of(b) == "TRACKING"
    assert client.rig.view(b).data["track_id"] == before
    assert client.rig.loop.world.detector is active_detector


def test_describe_without_a_sweep_reports_only_what_is_tracked(client):
    client.post("/behaviors", json={"kind": "highlight",
                                    "subject": {"detect": ["thing"]}})
    client.rig.settle()
    client.rig.frames(4, seen=boxes(THING))
    d = client.post("/describe", json={"sweep": False}).json()
    assert d["swept"] is False
    assert [o["label"] for o in d["objects"]] == ["thing"]


def test_describe_says_what_the_camera_is_doing(client):
    client.post("/behaviors", json={"kind": "highlight",
                                    "subject": {"detect": ["thing"]},
                                    "render": {"label": "counting people"}})
    client.rig.settle()
    client.rig.frames(3, seen=boxes(THING))
    d = client.post("/describe", json={"sweep": False}).json()
    assert "counting people" in d["behaviors"]


def test_a_failed_sweep_falls_back_instead_of_failing_the_question(client, monkeypatch):
    """A question that returns nothing is worse than one answered narrowly."""
    client.post("/behaviors", json={"kind": "highlight",
                                    "subject": {"detect": ["thing"]}})
    client.rig.settle()
    client.rig.frames(4, seen=boxes(THING))

    import scene as scene_mod
    monkeypatch.setattr(scene_mod, "sweep",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no model")))
    r = client.post("/describe", json={})
    assert r.status_code == 200
    d = r.json()
    # The reason, not the exception type: the agent reads this.
    assert d["swept"] is False and "no model" in d["detail"]
    assert "reporting tracked objects only" in d["detail"]
    assert [o["label"] for o in d["objects"]] == ["thing"]


def test_describe_with_no_frame_still_answers(client):
    d = client.post("/describe", json={}).json()
    assert d["swept"] is False and d["detail"] == "no frame to look at"
    assert "cannot see" in d["summary"]


def test_perception_never_calls_a_language_model(client):
    """The boundary. If this ever changes, the LLM plumbing has leaked from
    the orchestrator into the reflex layer."""
    import server.api as api_mod

    source = (api_mod.__file__,)
    for path in source:
        text = open(path).read().lower()
        for forbidden in ("openai", "anthropic", "chat.completions", "gpt-"):
            assert forbidden not in text, f"{forbidden} appeared in {path}"


# 6. Choosing what to sweep with ------------------------------------------
def test_the_sweep_never_uses_the_active_detector(rig):
    """The live detector is not static — the agent can swap it mid-run — so
    the spare is chosen per request, not fixed at boot."""
    assert rig.registry.active("detector") == "fake"
    assert rig.registry.sweep_detector() == "scene"

    rig.registry.set_active("detector", "scene")
    assert rig.registry.sweep_detector() != "scene"


def test_a_fixed_vocabulary_detector_is_never_swept_with(rig, monkeypatch):
    """A sweep asks about arbitrary words; a fixed vocabulary could only ever
    answer with its own class list."""
    spare = rig.registry.entry("scene")
    monkeypatch.setattr(type(spare.model), "open_vocab", False)
    assert rig.registry.sweep_detector() is None


def test_with_no_spare_detector_the_question_is_answered_narrowly(client, monkeypatch):
    """Rather than disturbing the one the camera is tracking with."""
    client.post("/behaviors", json={"kind": "highlight",
                                    "subject": {"detect": ["thing"]}})
    client.rig.settle()
    client.rig.frames(4, seen=boxes(THING))
    monkeypatch.setattr(type(client.rig.registry), "sweep_detector",
                        lambda self: None)

    d = client.post("/describe", json={}).json()
    assert d["swept"] is False
    assert "no spare" in d["detail"]
    assert [o["label"] for o in d["objects"]] == ["thing"]


def test_an_entry_marked_for_sweeping_is_preferred(rig):
    assert rig.registry.entry("scene").sweep is True
    assert rig.registry.sweep_detector() == "scene"
