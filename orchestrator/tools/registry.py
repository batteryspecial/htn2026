"""The agent's tools.

Each one wraps a call in `tools/perception.py` and publishes what it did to
the trace. The argument schemas are the shared pydantic contracts themselves
(`BehaviorSpec`, `Selector`), so the model is shown the same field
descriptions the pipeline validates against — there is no second copy of the
schema to drift.

Tools return strings. A model reads prose better than it reads a dict, and
every failure comes back as a sentence naming what to do next rather than an
exception that would end the turn.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.trace import BUS
from config import CFG
from contracts import BehaviorSpec, Selector
from memory.store import PhraseMemory
from tools.perception import Perception
from tools.scene import answer_scene
from tools.waiting import wait_for_state

log = logging.getLogger("orchestrator.tools")


def _json(value: Any, limit: int = 1200) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "…"


# --- argument schemas ------------------------------------------------------
# Thin wrappers where a tool takes more than one contract object, so the model
# always sees named fields rather than a positional blob.

class StopArgs(BaseModel):
    behavior_id: str = Field(description="The id from GET /behaviors, e.g. 'b3'.")


class ProbeArgs(BaseModel):
    subject: str = Field(description="What the operator called it, e.g. 'the yellow duck'. "
                                     "Used to file the measurement for next time.")
    phrases: list[str] = Field(
        description="Two to four candidate wordings for `detect`, specific through "
                    "general, e.g. ['yellow rubber duck', 'rubber duck', 'toy'].",
        min_length=1, max_length=6)


class CountArgs(BaseModel):
    selector: Selector
    window_s: float = Field(default=1.0, gt=0.0, le=10.0,
                            description="Seconds to take the median over.")


class LookArgs(BaseModel):
    selector: Selector | None = Field(
        default=None, description="Omit to get everything currently tracked.")


class DescribeArgs(BaseModel):
    question: str = Field(description="The operator's question, verbatim.")
    candidates: list[str] = Field(
        default_factory=list, max_length=64,
        description="Classes to sweep for. Empty uses the built-in everyday vocabulary.")
    sweep: bool = Field(default=True,
                        description="Take a detection pass over a broad vocabulary. "
                                    "An open question names nothing, so usually true.")


class ReferenceArgs(BaseModel):
    source: str = Field(default="largest_person",
                        description="What on screen to latch onto: 'largest_person'.")
    label: str | None = Field(default=None, description="A name for it.")


class ModelArgs(BaseModel):
    name: str = Field(description="A model name from GET /models, e.g. 'coco'.")


class HudArgs(BaseModel):
    text: str = Field(description="A short label for what the camera is doing. "
                                  "Shown burned onto the frame.")


class SayArgs(BaseModel):
    text: str = Field(description="One short sentence, spoken aloud to the operator.")


class WaitArgs(BaseModel):
    behavior_id: str = Field(description="The behavior id returned by start_behavior.")
    state: str = Field(default="REACHED", description="State to wait for, e.g. REACHED for pan_to.")
    timeout_s: float = Field(default=60, gt=0, le=90)


def build_tools(perception: Perception, memory: PhraseMemory,
                turn: str = "",
                installations: dict[str, dict[str, Any]] | None = None,
                ) -> list[StructuredTool]:
    """Bind the tools for one turn, so tracing carries the turn id."""
    receipts = installations if installations is not None else {}

    def trace_call(name: str, detail: str, **data: Any) -> None:
        BUS.emit("tool_call", name, detail, turn=turn, **data)

    def trace_result(name: str, detail: str, ok: bool = True, **data: Any) -> None:
        BUS.emit("tool_result" if ok else "error", name, detail,
                 turn=turn, ok=ok, **data)

    async def await_operation(result: dict[str, Any], final: str) -> tuple[bool, str]:
        operation_id = result.get("operation_id")
        if not operation_id:  # compatibility with an older pipeline/fakes
            return True, final
        deadline = asyncio.get_running_loop().time() + CFG.tool_timeout_s
        last: dict[str, Any] | None = None
        while asyncio.get_running_loop().time() < deadline:
            checked = await perception.operation_status(operation_id)
            if checked["ok"]:
                last = checked
                if checked.get("status") == final:
                    return True, final
                if checked.get("status") == "failed":
                    return False, str(checked.get("detail") or "operation failed")
            await asyncio.sleep(0.05)
        return False, str((last or {}).get("detail")
                          or f"operation was not confirmed within {CFG.tool_timeout_s:.0f}s")

    # 1. Behaviours ---------------------------------------------------------

    async def start_behavior(**spec_fields: Any) -> str:
        spec = BehaviorSpec(**spec_fields)
        payload = spec.model_dump(exclude_none=True)
        label = spec.render.label or spec.subject.summary()
        trace_call("start_behavior", f"{spec.kind} · {label}", spec=payload)

        result = await perception.start_behavior(payload)
        if not result["ok"]:
            trace_result("start_behavior", result["error"], ok=False)
            return (f"REFUSED: {result['error']}\n"
                    f"Fix the call and try again — this message names what was wrong.")

        behavior_id = result.get("id", "?")
        deadline = asyncio.get_running_loop().time() + CFG.tool_timeout_s
        status: dict[str, Any] | None = None
        while asyncio.get_running_loop().time() < deadline:
            checked = await perception.behavior_status(behavior_id)
            if checked["ok"]:
                status = checked
                if checked.get("status") in {"installed", "failed"}:
                    break
            await asyncio.sleep(0.05)

        if not status or status.get("status") != "installed":
            detail = ((status or {}).get("detail")
                      or f"installation was not confirmed within {CFG.tool_timeout_s:.0f}s")
            trace_result("start_behavior", detail, ok=False,
                         behavior_id=behavior_id, spec=payload)
            return (f"NOT INSTALLED: {behavior_id}: {detail}. "
                    "Do not claim this behavior is running.")

        receipts[behavior_id] = payload
        trace_result("start_behavior", f"{behavior_id} · {spec.kind} · {label}",
                     behavior_id=behavior_id, spec=payload, ok=True)
        return (f"Installed {behavior_id}: {spec.kind} on {spec.subject.summary()}. "
                "It will keep running every frame until explicitly stopped. "
                "Zero matches means it is still looking, not that it ended.")

    async def stop_behavior(behavior_id: str) -> str:
        trace_call("stop_behavior", behavior_id)
        result = await perception.stop_behavior(behavior_id)
        if not result["ok"]:
            trace_result("stop_behavior", result["error"], ok=False)
            return f"Could not stop {behavior_id}: {result['error']}"
        confirmed, detail = await await_operation(result, "removed")
        if not confirmed:
            trace_result("stop_behavior", detail, ok=False,
                         behavior_id=behavior_id)
            return f"Could not confirm stopping {behavior_id}: {detail}"
        receipts.pop(behavior_id, None)
        trace_result("stop_behavior", f"removed {behavior_id}",
                     behavior_id=behavior_id, ok=True)
        return f"Stopped {behavior_id}."

    async def clear_behaviors() -> str:
        trace_call("clear_behaviors", "everything stops")
        result = await perception.clear_behaviors()
        if not result["ok"]:
            trace_result("clear_behaviors", result["error"], ok=False)
            return f"Could not clear: {result['error']}"
        confirmed, detail = await await_operation(result, "cleared")
        if not confirmed:
            trace_result("clear_behaviors", detail, ok=False)
            return f"Could not confirm clearing behaviors: {detail}"
        receipts.clear()
        trace_result("clear_behaviors", "pipeline is idle", ok=True)
        return "Cleared every behaviour. The pipeline is idle."

    async def list_behaviors() -> str:
        result = await perception.list_behaviors()
        if not result["ok"]:
            return f"Could not read the behaviour list: {result['error']}"
        return _json(result.get("behaviors", []))

    async def wait_for_behavior(behavior_id: str, state: str = "REACHED",
                                timeout_s: float = 60) -> str:
        trace_call("wait_for_behavior", f"{behavior_id} → {state}")
        result = await wait_for_state(perception, behavior_id, state, timeout_s)
        detail = f"{behavior_id} reached {state}" if result["ok"] else result["error"]
        trace_result("wait_for_behavior", detail, ok=result["ok"])
        return detail

    # 2. Asking the pipeline ------------------------------------------------

    async def probe_phrases(subject: str, phrases: list[str]) -> str:
        trace_call("probe_phrases", f"{subject}: {', '.join(phrases)}", phrases=phrases)
        result = await perception.probe(phrases)

        if not result["ok"]:
            trace_result("probe_phrases", result["error"], ok=False)
            return (f"Probe failed: {result['error']}. Proceed with your best "
                    f"guess — send several phrasings, specific through general.")

        if result.get("detail"):
            detail = str(result["detail"])
            trace_result("probe_phrases", detail, ok=False)
            return (f"Probe failed: {detail}. Proceed with a best-effort standing "
                    "behavior; a probe failure must not cancel the instruction.")

        results = result.get("results", [])
        await asyncio.to_thread(memory.remember_probe, subject, results)

        best = result.get("best")
        advice = result.get("advice", "")
        scores = ", ".join(
            f'"{r.get("phrase")}" {r.get("max_conf", 0):.2f}'
            f'{"" if r.get("found") else " (nothing)"}'
            for r in results)
        trace_result("probe_phrases", scores or advice, best=best, results=results)
        return f"Measured right now: {scores}.\n{advice}"

    async def count_objects(selector: Selector, window_s: float = 1.0) -> str:
        trace_call("count_objects", selector.summary())
        result = await perception.count(selector.model_dump(exclude_none=True), window_s)
        if not result["ok"]:
            trace_result("count_objects", result["error"], ok=False)
            return f"Count failed: {result['error']}"
        count = result.get("count", 0)
        trace_result("count_objects", f"{count} over {window_s:.0f}s", count=count)
        return (f"{count}, as a median over {window_s:.0f}s of frames "
                f"({result.get('samples', 0)} samples).")

    async def look(selector: Selector | None = None) -> str:
        trace_call("look", selector.summary() if selector else "everything tracked")
        result = await perception.look(
            selector.model_dump(exclude_none=True) if selector else None)
        if not result["ok"]:
            trace_result("look", result["error"], ok=False)
            return f"Look failed: {result['error']}"
        tracks = result.get("tracks", [])
        trace_result("look", f"{len(tracks)} tracked", tracks=tracks)
        return _json(tracks) if tracks else "Nothing is being tracked right now."

    async def describe_scene(question: str, candidates: list[str] | None = None,
                             sweep: bool = True) -> str:
        trace_call("describe_scene", question)
        result = await perception.describe(question, candidates, sweep)
        if not result["ok"]:
            trace_result("describe_scene", result["error"], ok=False)
            return f"Describe failed: {result['error']}"

        summary = result.get("summary", "")
        objects = result.get("objects", [])
        trace_result("describe_scene", summary or f"{len(objects)} objects",
                     objects=objects, swept=result.get("swept"))

        lines = [f"The camera can see: {summary or 'nothing it recognises'}."]
        lines.append(
            "Treat the detection summary as grounding; do not contradict it "
            "without clear visual evidence.")
        if objects:
            lines.append("Objects: " + _json(objects))
        if result.get("detail"):
            lines.append(f"Note: {result['detail']}")
        try:
            lines.append("Visual answer: " + await answer_scene(perception, question, result))
        except Exception as exc:
            lines.append(f"Visual inspection failed: {exc}. Only the detection summary is available.")
        return "\n".join(lines)

    # 3. References, model, HUD ---------------------------------------------

    async def add_reference(source: str = "largest_person",
                            label: str | None = None) -> str:
        trace_call("add_reference", source)
        result = await perception.add_reference_from_frame(source, label)
        if not result["ok"]:
            trace_result("add_reference", result["error"], ok=False)
            return f"Could not register a reference: {result['error']}"
        ref_id = result.get("ref_id")
        trace_result("add_reference", f"{ref_id} from {source}", ref_id=ref_id)
        return (f"Registered {ref_id}. Put it in the selector as ref_id with "
                f'pick "ref" to hold that one instance across frames.')

    async def set_model(name: str) -> str:
        trace_call("set_model", name)
        result = await perception.set_model(name)
        if not result["ok"]:
            trace_result("set_model", result["error"], ok=False)
            return f"Could not switch model: {result['error']}"
        confirmed, detail = await await_operation(result, "switched")
        if not confirmed:
            trace_result("set_model", detail, ok=False)
            return f"Could not confirm model switch: {detail}"
        trace_result("set_model", f"now {name}")
        return (f"Switched to {name}. Behaviours it cannot serve are paused with "
                f"a reason and resume on their own — do not restart them.")

    async def set_hud(text: str) -> str:
        result = await perception.set_hud(text)
        if not result["ok"]:
            return f"Could not set the HUD: {result['error']}"
        BUS.emit("tool_result", "set_hud", text, turn=turn)
        return f"HUD now reads: {text}"

    async def say(text: str) -> str:
        """Spoken aloud by the frontend. Not a reply — this is for alerts."""
        BUS.emit("say", text, turn=turn)
        return "Said."

    def tool(fn, name: str, description: str,
             args_schema: type[BaseModel] | None = None) -> StructuredTool:
        return StructuredTool.from_function(
            coroutine=fn, name=name, description=description,
            args_schema=args_schema)

    return [
        tool(start_behavior, "start_behavior",
             "Install a standing behaviour. The pipeline runs it every frame "
             "until it is stopped. This is how every camera capability is "
             "switched on. Refusals name exactly what was wrong.",
             BehaviorSpec),
        tool(stop_behavior, "stop_behavior",
             "Stop one behaviour by its id, leaving the rest running.",
             StopArgs),
        tool(clear_behaviors, "clear_behaviors",
             "Stop every behaviour. Use for 'stop' or 'clear everything', and "
             "before installing a replacement objective."),
        tool(list_behaviors, "list_behaviors",
             "What is running right now, with state and match counts. You are "
             "given this at the start of each turn; call it only to re-check "
             "after changing something."),
        tool(wait_for_behavior, "wait_for_behavior",
             "Wait for a behavior's state. After pan_to, wait for REACHED before counting. "
             "If it times out or is stopped, explain that the turn was not completed; do not count yet.",
             WaitArgs),
        tool(probe_phrases, "probe_phrases",
             "Ask the detector whether these wordings find anything in this "
             "room right now. The difference between a behaviour that works "
             "and one that sits ACTIVE matching nothing. Runs on a spare "
             "detector, so it never disturbs tracking.",
             ProbeArgs),
        tool(count_objects, "count_objects",
             "How many things match, as a median over a window so one bad "
             "frame cannot lie. Use for 'how many...' — it installs nothing.",
             CountArgs),
        tool(look, "look",
             "What is being tracked right now, with attribute scores.",
             LookArgs),
        tool(describe_scene, "describe_scene",
             "What is in front of the camera, for an open question like "
             "\"what's on the table?\". Sweeps a broad vocabulary, because an "
             "open-vocabulary detector only finds what it is named.",
             DescribeArgs),
        tool(add_reference, "add_reference",
             "Latch onto what is on screen now and get a ref_id back. This is "
             "how 'follow him' works with no photo. For an uploaded photo the "
             "reference is registered before your turn starts and its id is "
             "given to you.",
             ReferenceArgs),
        tool(set_model, "set_model",
             "Swap the detector. 'coco' is fixed-vocabulary and takes only its "
             "own class names; 'yoloe' is open-vocabulary and takes phrases.",
             ModelArgs),
        tool(set_hud, "set_hud",
             "Set the short label burned onto the video, so the projected "
             "frame says what the camera is doing.",
             HudArgs),
        tool(say, "say",
             "Speak one sentence aloud to the operator. For alerts during an "
             "event turn. Your final reply is spoken anyway, so do not use "
             "this to repeat it.",
             SayArgs),
    ]
