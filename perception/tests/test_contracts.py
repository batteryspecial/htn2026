"""Schema guardrails. A malformed spec must be rejected at the door, not
halfway down the pipeline where the car is already moving."""

import pytest
from pydantic import ValidationError

from contracts import Phase, SpecRequest, TaskSpec, Verify


def spec(**over):
    body = {"spec_id": "s1", "targets": [{"ref": "t1", "detect": ["pencil"]}]}
    body.update(over)
    return TaskSpec(**body)


def test_defaults():
    s = spec()
    assert s.mode == "follow"
    assert s.model == "yoloe"
    assert s.targets[0].select == "largest"
    assert s.arbitration.alternate_s == 3.0


def test_verify_uses_class_alias():
    v = Verify(**{"class": "shoe", "text": "a red shoe"})
    assert v.cls == "shoe" and v.min_score == 0.6


@pytest.mark.parametrize(
    "over",
    [
        {"targets": []},
        {"targets": [{"ref": f"t{i}", "detect": ["a"]} for i in range(3)]},
        {"targets": [{"ref": "t1", "detect": []}]},
        {"targets": [{"ref": "t1", "detect": ["a"] * 7}]},
        {"targets": [{"ref": "t1", "detect": ["a"], "select": "vibes"}]},
        {"mode": "teleport"},
        {"arbitration": {"alternate_s": 0}},
        {"nonsense": 1},
    ],
)
def test_rejects_bad_spec(over):
    with pytest.raises(ValidationError):
        spec(**over)


def test_prompt_union_dedups_and_includes_relate():
    s = spec(
        targets=[
            {"ref": "a", "detect": ["person"], "relate": {"keep": "person", "if_contains": "shoe"}},
            {"ref": "b", "detect": ["shoe", "dog"]},
        ]
    )
    assert s.prompt_union() == ["person", "shoe", "dog"]


def test_spec_request_roundtrip():
    r = SpecRequest(instruction_id="i1", spec=spec().model_dump())
    assert r.spec.spec_id == "s1"


def test_only_tracking_is_drivable():
    assert Phase.TRACKING.drivable
    assert not any(p.drivable for p in Phase if p is not Phase.TRACKING)
