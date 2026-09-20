"""The keyboard skill: the fit, and the behaviour that draws it.

easyocr is not needed for any of this. The list of (character, x, y) is the
contract with the reader, and everything here is geometry over that list.
Whether EasyOCR can actually read keys off a real keyboard, in the room, at
that angle, is the one thing only the demo machine can answer.
"""

import numpy as np
import pytest

from contracts import BehaviorSpec
from skills import keyboard as layout
from tests.rig import Rig, boxes

KEYBOARD = ("keyboard", 320, 300, 300)


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.close()


def photograph(chars: str, scale: float = 40.0, ox: float = 100.0,
               oy: float = 200.0) -> list[tuple[str, float, float]]:
    """What OCR would report for a keyboard lying square to the camera."""
    return [(c, ox + layout.TEMPLATE[c][0] * scale, oy + layout.TEMPLATE[c][1] * scale)
            for c in chars]


# 1. The template ---------------------------------------------------------
def test_the_template_has_every_typeable_key():
    assert set("qwertyuiop") <= set(layout.TEMPLATE)
    assert set("1234567890") <= set(layout.TEMPLATE)
    assert " " in layout.TEMPLATE
    assert len(layout.TEMPLATE) == 10 + 10 + 9 + 7 + 1


def test_the_rows_are_staggered_like_a_real_keyboard():
    """'a' sits right of 'q', and 'z' right of 'a'. If the stagger is ever
    flattened the fit still succeeds and every badge lands slightly wrong."""
    assert layout.TEMPLATE["q"][0] < layout.TEMPLATE["a"][0] < layout.TEMPLATE["z"][0]
    assert layout.TEMPLATE["1"][1] < layout.TEMPLATE["q"][1] < layout.TEMPLATE["a"][1]


# 2. The fit --------------------------------------------------------------
def test_reading_a_few_keys_locates_all_of_them():
    """The whole point: OCR never reads every key, and never the same ones
    twice. Six characters have to be enough to place the other forty-one."""
    matrix = layout.fit(photograph("qwerty"))
    assert matrix is not None
    (x, y), = layout.project(matrix, "p")
    # 'p' was never read, and lands where the template says it must.
    assert abs(x - (100.0 + 9 * 40.0)) < 2.0
    assert abs(y - 200.0) < 2.0


def test_too_few_characters_is_no_fit():
    assert layout.fit(photograph("qw")) is None
    assert layout.fit([]) is None


def test_a_stray_reading_does_not_drag_the_fit():
    """RANSAC earning its place: a letter read off a mug or a sticker is one
    bad correspondence, and least squares would bend the whole keyboard to it."""
    seen = photograph("qwertyuiop")
    clean = layout.fit(seen)
    seen.append(("m", 2000.0, -900.0))
    dirty = layout.fit(seen)
    assert clean is not None and dirty is not None
    before, = layout.project(clean, "g")
    after, = layout.project(dirty, "g")
    assert abs(before[0] - after[0]) < 5.0 and abs(before[1] - after[1]) < 5.0


def test_the_fit_survives_perspective():
    """A keyboard on a desk is never square to the camera."""
    seen = photograph("qwertyuiopasdfghjkl")
    skewed = [(c, x + 0.35 * y, y * 0.7 + 0.15 * x) for c, x, y in seen]
    matrix = layout.fit(skewed)
    assert matrix is not None
    (px, py), = layout.project(matrix, "m")
    tx, ty = layout.TEMPLATE["m"]
    ex, ey = 100.0 + tx * 40.0, 200.0 + ty * 40.0
    assert abs(px - (ex + 0.35 * ey)) < 4.0
    assert abs(py - (ey * 0.7 + 0.15 * ex)) < 4.0


def test_project_skips_characters_that_are_not_keys():
    matrix = layout.fit(photograph("qwerty"))
    assert layout.project(matrix, "q!") == [layout.project(matrix, "q")[0], None]


def test_smoothing_moves_toward_the_new_fit_without_jumping():
    a = layout.fit(photograph("qwertyuiop"))
    b = layout.fit(photograph("qwertyuiop", ox=140.0))
    blended = layout.smooth(a, b, alpha=0.5)
    (x, _), = layout.project(blended, "q")
    assert 100.0 < x < 140.0


def test_the_first_fit_is_used_as_is():
    matrix = layout.fit(photograph("qwerty"))
    assert np.allclose(layout.smooth(None, matrix), matrix)


# 3. The behaviour --------------------------------------------------------
class FakeOCR:
    role = "ocr"
    is_loaded = True

    def __init__(self):
        self.seen: list = []

    def load(self):
        pass

    def latest(self, frame):
        return self.seen


def with_ocr(rig):
    model = FakeOCR()
    rig.registry._entries["ocr"] = type(rig.registry.entry("fake"))(
        name="ocr", role="ocr", type="fake")
    rig.registry._entries["ocr"].model = model
    rig.registry._active["ocr"] = "ocr"
    rig.loop.registry = rig.registry
    return model


def keyboard(rig, **params):
    return rig.builder.add_behavior(BehaviorSpec(
        kind="keyboard", subject={"detect": ["keyboard"]},
        params=params, render={"label": "type"}))


def test_text_is_required():
    from behaviors.kinds import build
    with pytest.raises(ValueError, match="text"):
        build("b1", BehaviorSpec(kind="keyboard", subject={"detect": ["keyboard"]}))


def test_text_with_no_typeable_characters_is_refused():
    from behaviors.kinds import build
    with pytest.raises(ValueError, match="template"):
        build("b1", BehaviorSpec(kind="keyboard", subject={"detect": ["keyboard"]},
                                 params={"text": "!!!"}))


def test_an_unknown_step_mode_names_the_real_ones():
    from behaviors.kinds import build
    with pytest.raises(ValueError, match="step_mode"):
        build("b1", BehaviorSpec(kind="keyboard", subject={"detect": ["keyboard"]},
                                 params={"text": "hi", "step_mode": "interpretive"}))


def test_without_an_ocr_model_it_is_paused(rig):
    b = keyboard(rig, text="hack")
    rig.settle()
    rig.frames(3, seen=boxes(KEYBOARD))
    assert rig.state_of(b) == "PAUSED"
    assert "ocr" in (rig.view(b).detail or "")


def test_it_locks_once_the_keys_are_read(rig):
    model = with_ocr(rig)
    b = keyboard(rig, text="hack")
    rig.settle()
    rig.frames(2, seen=boxes(KEYBOARD))
    assert rig.state_of(b) == "SEARCHING"

    model.seen = photograph("qwertyuiop")
    rig.frames(3, seen=boxes(KEYBOARD))
    assert rig.state_of(b) == "LOCKED"
    assert "keyboard_locked" in rig.types()


def test_it_unlocks_when_the_keyboard_goes_away(rig):
    model = with_ocr(rig)
    b = keyboard(rig, text="hack")
    rig.settle()
    model.seen = photograph("qwertyuiop")
    rig.frames(3, seen=boxes(KEYBOARD))
    assert rig.state_of(b) == "LOCKED"

    model.seen = []
    rig.frames(60, seen=boxes(KEYBOARD))     # past LOST_AFTER_S at 0.05s/frame
    assert rig.state_of(b) == "SEARCHING"
    assert "keyboard_lost" in rig.types()


def test_a_repeated_letter_carries_every_position(rig):
    """The 'h' in "hack the north" is keys 1, 7 and 14. Three badges stacked
    on one key is an unreadable smudge; one badge reading 1·7·14 is the demo."""
    from render.layers import Badges
    model = with_ocr(rig)
    keyboard(rig, text="hack the north")
    rig.settle()
    model.seen = photograph("qwertyuiopasdfghjkl")
    rig.frames(4, seen=boxes(KEYBOARD))

    badges = [ly for ly in rig.loop.view.layers if isinstance(ly, Badges)]
    assert badges, "nothing drawn"
    assert any("·" in t for t in badges[-1].texts)


def test_manual_mode_steps_only_when_asked(rig):
    model = with_ocr(rig)
    bid = keyboard(rig, text="hack", step_mode="manual")
    rig.settle()
    model.seen = photograph("qwertyuiopasdfghjkl")
    rig.frames(4, seen=boxes(KEYBOARD))
    behavior = rig.loop.behaviors[bid]
    assert behavior.cursor == 0

    rig.frames(20, seen=boxes(KEYBOARD))
    assert behavior.cursor == 0, "a manual keyboard advanced on its own"

    behavior.advance()
    rig.frames(2, seen=boxes(KEYBOARD))
    assert behavior.cursor == 1
    assert "step" in rig.types()


def test_advance_over_http(rig):
    """`manual` is driven by the agent, so the route is the real interface."""
    from fastapi.testclient import TestClient

    from server.api import Service, create_app

    model = with_ocr(rig)
    bid = keyboard(rig, text="hack", step_mode="manual")
    rig.settle()
    model.seen = photograph("qwertyuiopasdfghjkl")
    rig.frames(4, seen=boxes(KEYBOARD))

    svc = Service(registry=rig.registry, capture=rig.capture, builder=rig.builder,
                  loop=rig.loop, health=rig.health, events=rig.event_bus,
                  states=rig.state_bus)
    with TestClient(create_app(svc, run_threads=False)) as c:
        assert c.post(f"/behaviors/{bid}/advance").status_code == 202
        rig.frames(2, seen=boxes(KEYBOARD))
        assert rig.loop.behaviors[bid].cursor == 1

        assert c.post("/behaviors/ghost/advance").status_code == 404

        other = rig.builder.add_behavior(BehaviorSpec(
            kind="highlight", subject={"detect": ["thing"]}))
        rig.settle()
        r = c.post(f"/behaviors/{other}/advance")
        assert r.status_code == 422 and "keyboard" in r.json()["detail"]


def test_auto_mode_walks_the_word_on_its_own(rig):
    model = with_ocr(rig)
    bid = keyboard(rig, text="hack", step_mode="auto", step_s=0.1)
    rig.settle()
    model.seen = photograph("qwertyuiopasdfghjkl")
    rig.frames(20, seen=boxes(KEYBOARD))    # 0.05s/frame, so ~10 steps' worth
    assert rig.loop.behaviors[bid].cursor > 0
