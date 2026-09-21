"""An in-process stand-in for the perception service (:8001).

Mirrors the real routes closely enough that a call passing here would pass
there, so these tests exercise the orchestrator's own code — how it builds a
request, how it reads a refusal, what it does next — with no camera and no
model weights.

Faithful in the three places that matter:

* `POST /behaviors` validates with the shared `BehaviorSpec`, then applies the
  same kind-specific param checks the real service applies before accepting
  anything. Both failures come back as 422 `{"detail": ...}`.
* A refusal is a *specific sentence*. Perception goes to trouble over this
  precisely so an agent can read it and fix the call; a stub that answered
  "Unprocessable Entity" would be testing the wrong thing.
* `DELETE /behaviors/{id}` on an unknown id is 404, not a silent success.

`down` and `overrides` let a test turn any of it into the failure it wants.
"""

from __future__ import annotations

import itertools
import json
from typing import Any

import httpx
from pydantic import ValidationError

from contracts import BehaviorSpec

#: What the real service reports under `kinds`. `keyboard` is declared and not
#: built, which is the case the agent has to handle gracefully.
BUILT = ["count_line", "highlight", "pan_to", "pose_trigger", "privacy", "track", "watch"]
PLANNED = ["keyboard"]


def _params_error(spec: BehaviorSpec) -> str | None:
    """The same up-front checks `behaviors/kinds.py` makes when it builds.

    Only the three a plausible agent call actually trips. The real service
    runs the constructor and reports whatever it raises.
    """
    params = spec.params
    if spec.kind == "watch" and not params.get("triggers"):
        return "watch needs a non-empty 'triggers' list"
    if spec.kind == "pan_to" and not params.get("deg"):
        return "pan_to needs 'deg': how far to turn, negative for left"
    if spec.kind == "count_line" and not params.get("line"):
        return "count_line needs a 'line': two points in normalized coordinates"
    return None


class FakePipeline:
    """Routes, plus the knobs a test needs to make them misbehave."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.behaviors: dict[str, dict[str, Any]] = {}
        self._ids = itertools.count(1)
        self._refs = itertools.count(1)

        #: Every call raises a connection error, as if the service were down.
        self.down = False
        #: (METHOD, path) -> (status, payload). Wins over the real route.
        self.overrides: dict[tuple[str, str], tuple[int, Any]] = {}

        #: What `/probe` reports per phrase. Anything unlisted scores 0.00,
        #: which is the interesting case: the wording finds nothing.
        self.probe_scores: dict[str, float] = {}
        self.count_answer = 0
        self.tracks: list[dict[str, Any]] = []
        self.scene_objects: list[dict[str, Any]] = []
        self.health_payload: dict[str, Any] = {
            "status": "ok", "fps": 28.4, "model": "yoloe",
            "camera_ok": True, "behaviors": 0, "device": "cpu", "detail": None,
        }
        self.models_payload: dict[str, Any] = {
            "models": [
                {"name": "yoloe", "role": "detector", "active": True,
                 "available": True, "open_vocab": True},
                {"name": "coco", "role": "detector", "active": False,
                 "available": True, "open_vocab": False},
            ],
            # What perception reports from skills/pose.py. Set to [] to stand in
            # for a machine with no pose model loaded.
            "gestures": ["hand_raised"],
        }
        self.hud = ""

    # 1. Transport ---------------------------------------------------------

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    def handler(self, request: httpx.Request) -> httpx.Response:
        method, path = request.method, request.url.path

        body: Any = None
        if request.content and "json" in request.headers.get("content-type", ""):
            body = json.loads(request.content)
        self.calls.append({"method": method, "path": path, "json": body})

        if self.down:
            raise httpx.ConnectError("connection refused", request=request)

        override = self.overrides.get((method, path))
        if override is not None:
            status, payload = override
            return httpx.Response(status, json=payload)

        return self._route(method, path, body, request)

    def _route(self, method: str, path: str, body: Any,
               request: httpx.Request) -> httpx.Response:
        if path == "/behaviors":
            if method == "POST":
                return self._add(body)
            if method == "GET":
                return httpx.Response(200, json={
                    "behaviors": list(self.behaviors.values()),
                    "kinds": {"available": BUILT, "planned": PLANNED},
                })
            if method == "DELETE":
                self.behaviors.clear()
                return httpx.Response(202, json={"cleared": True})

        if path.startswith("/behaviors/") and path.endswith("/status") and method == "GET":
            behavior_id = path.split("/")[2]
            if behavior_id in self.behaviors:
                return httpx.Response(200, json={
                    "id": behavior_id, "status": "installed", "detail": None,
                })
            return httpx.Response(404, json={
                "detail": f"unknown behaviour {behavior_id!r}",
            })

        if path.startswith("/behaviors/") and method == "DELETE":
            return self._drop(path.rsplit("/", 1)[-1])

        if path == "/probe" and method == "POST":
            return self._probe(body or {})
        if path == "/query/count" and method == "POST":
            return httpx.Response(200, json={
                "count": self.count_answer, "samples": 14,
                "window_s": (body or {}).get("window_s", 1.0)})
        if path == "/query/look" and method == "POST":
            return httpx.Response(200, json={"ts": 0.0, "tracks": self.tracks})
        if path == "/describe" and method == "POST":
            return self._describe(body or {})

        if path == "/references" and method == "POST":
            ref_id = f"r{next(self._refs)}"
            return httpx.Response(201, json={
                "ref_id": ref_id, "thumb_url": f"/references/{ref_id}.jpg"})
        if path == "/health":
            return httpx.Response(200, json={**self.health_payload,
                                             "behaviors": len(self.behaviors)})
        if path == "/models":
            return httpx.Response(200, json=self.models_payload)
        if path == "/model" and method == "POST":
            return self._set_model(body or {})
        if path == "/hud" and method == "POST":
            self.hud = (body or {}).get("text", "")
            return httpx.Response(202, json={"text": self.hud})
        if path == "/snapshot" or path.startswith("/snapshots/"):
            return httpx.Response(200, content=b"\xff\xd8jpeg",
                                  headers={"content-type": "image/jpeg"})

        return httpx.Response(404, json={"detail": f"no route {method} {path}"})

    # 2. Behaviours --------------------------------------------------------

    def _add(self, body: Any) -> httpx.Response:
        try:
            spec = BehaviorSpec(**(body or {}))
        except ValidationError as exc:
            return _reject(str(exc))

        if spec.kind not in BUILT:
            return _reject(f"behaviour kind {spec.kind!r} is not implemented yet; "
                           f"available: {BUILT}")
        bad = _params_error(spec)
        if bad:
            return _reject(bad)

        behavior_id = f"b{next(self._ids)}"
        self.behaviors[behavior_id] = {
            "id": behavior_id,
            "kind": spec.kind,
            "state": "ARMING" if spec.kind == "watch" else "ACTIVE",
            "spec": spec.subject.summary(),
            "label": spec.render.label,
            "matches": 0,
            "track_ids": [],
            "since": 0.0,
            "detail": None,
            "notify": spec.notify,
        }
        return httpx.Response(201, json={"id": behavior_id})

    def _drop(self, behavior_id: str) -> httpx.Response:
        if behavior_id not in self.behaviors:
            return httpx.Response(404, json={"detail": f"no behaviour {behavior_id!r}"})
        del self.behaviors[behavior_id]
        return httpx.Response(202, json={"removed": behavior_id})

    # 3. Asking ------------------------------------------------------------

    def _probe(self, body: dict[str, Any]) -> httpx.Response:
        phrases = body.get("phrases") or []
        results = []
        for phrase in phrases:
            conf = float(self.probe_scores.get(phrase, 0.0))
            results.append({"phrase": phrase, "found": 1 if conf else 0,
                            "mean_conf": conf, "max_conf": conf,
                            "max_area": 0.04 if conf else 0.0})

        hits = [r for r in results if r["found"]]
        best = max(hits, key=lambda r: r["max_conf"])["phrase"] if hits else None
        advice = (f'Use "{best}".' if best else
                  "Nothing matched. Try a broader noun, or move the adjective "
                  "out of detect and into include.")
        return httpx.Response(200, json={"ts": 0.0, "results": results,
                                         "best": best, "advice": advice})

    def _describe(self, body: dict[str, Any]) -> httpx.Response:
        summary = ", ".join(o["label"] for o in self.scene_objects)
        return httpx.Response(200, json={
            "ts": 0.0, "snapshot_url": "/snapshot",
            "objects": self.scene_objects, "summary": summary,
            "prompt": f"Question: {body.get('question', '')}",
            "behaviors": [b["kind"] for b in self.behaviors.values()],
            "swept": bool(body.get("sweep", True)), "detail": None,
        })

    def _set_model(self, body: dict[str, Any]) -> httpx.Response:
        name = body.get("name")
        known = [m["name"] for m in self.models_payload["models"]]
        if name not in known:
            return _reject(f"unknown model {name!r}; available: {known}")
        for model in self.models_payload["models"]:
            model["active"] = model["name"] == name
        self.health_payload["model"] = name
        return httpx.Response(202, json={"accepted": True, "model": name,
                                         "role": "detector"})

    # 4. For assertions ----------------------------------------------------

    def paths(self, method: str | None = None) -> list[str]:
        return [c["path"] for c in self.calls
                if method is None or c["method"] == method]

    def last(self, method: str, path: str) -> dict[str, Any] | None:
        for call in reversed(self.calls):
            if call["method"] == method and call["path"] == path:
                return call
        return None


def _reject(reason: str) -> httpx.Response:
    return httpx.Response(422, json={"detail": reason})
