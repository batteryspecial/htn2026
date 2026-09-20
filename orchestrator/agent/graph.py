"""The agent loop, as a LangGraph state machine.

    ground ──▶ think ──▶ (tools? ) ──▶ act ──┐
                 ▲                            │
                 └────────────────────────────┘   up to max_steps
                 │
                 └──▶ finish

`ground` is the node that makes this better than a one-shot compiler. Before
the model sees the instruction it is handed what is actually running, what the
detector is, and what wordings have been measured for subjects like this one.
Then it can probe, read the result, and only then commit.

The loop is capped. A demo that thinks for twenty steps has already lost.
"""

from __future__ import annotations

import logging
import time
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import (
    AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agent.llm import NoModelConfigured, chat_model
from agent.prompts import EVENT_TURN, situation, system_prompt
from app.trace import BUS
from config import CFG
from memory.store import PhraseMemory
from tools.perception import Perception
from tools.registry import build_tools

log = logging.getLogger("orchestrator.agent")


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    turn: str
    #: "user" or "event" — an event turn looks at a snapshot and usually just speaks.
    origin: str
    #: The operator's words, kept for the phrase memory lookup.
    instruction: str
    #: A snapshot to look at, base64 data URL. Event turns and image uploads.
    image_urls: list[str]
    steps: int
    started: float
    reply: str


def _text_of(message: AIMessage) -> str:
    """Model content is a string or a list of parts, depending on provider."""
    content = message.content
    if isinstance(content, str):
        return content.strip()
    parts = [
        part.get("text", "") for part in content
        if isinstance(part, dict) and part.get("type") == "text"
    ]
    return "\n".join(p for p in parts if p).strip()


def build_graph(perception: Perception, memory: PhraseMemory):
    """One compiled graph per process. Tools are bound per turn for tracing."""

    async def ground(state: AgentState) -> dict[str, Any]:
        """Everything true right now, gathered before the model thinks.

        One round trip each to two cheap endpoints, in exchange for the model
        not having to ask what is running. Failures here are not fatal: a
        pipeline that is down is something the agent should be told about, not
        something that should stop the turn.
        """
        turn = state["turn"]
        listing = await perception.list_behaviors()
        health = await perception.health()

        behaviors = listing.get("behaviors", []) if listing["ok"] else []
        health_data = {k: v for k, v in health.items() if k != "ok"} if health["ok"] else None

        open_vocab = True
        # None, not [], until the pipeline answers: "we did not ask" and "there
        # are no gestures" read the same to the model otherwise, and the second
        # one is a claim we have not earned.
        gestures: list[str] | None = None
        models = await perception.models()
        if models["ok"]:
            active = next((m for m in models.get("models", []) if m.get("active")), None)
            if active:
                open_vocab = active.get("open_vocab", True) is not False
            gestures = models.get("gestures", [])

        lines = [situation(behaviors, health_data, open_vocab, gestures)]

        if not health["ok"]:
            lines.append(
                f"WARNING: the pipeline is not answering ({health['error']}). "
                f"Tell the operator plainly; do not pretend a behaviour started."
            )

        # The prior. Measured evidence for subjects like this one, which is
        # the thing that cannot fit in a system prompt and grows with use.
        if state.get("instruction"):
            prior = memory.brief(state["instruction"])
            if prior:
                lines.append(prior)
                BUS.emit("thought", "recalled phrasings",
                         prior.split("\n", 1)[-1][:300], turn=turn)

        BUS.emit("thought", "grounded",
                 f"{len(behaviors)} running", turn=turn, behaviors=behaviors)

        return {"messages": [SystemMessage(content="\n\n".join(lines))]}

    async def think(state: AgentState) -> dict[str, Any]:
        """One model call. Either it asks for tools or it answers."""
        turn = state["turn"]
        tools = build_tools(perception, memory, turn)

        try:
            model = chat_model().bind_tools(tools)
        except NoModelConfigured as exc:
            BUS.emit("error", "no model", str(exc), turn=turn)
            return {"reply": str(exc),
                    "messages": [AIMessage(content=str(exc))]}

        try:
            reply: AIMessage = await model.ainvoke(state["messages"])
        except Exception as exc:                        # noqa: BLE001
            log.exception("model call failed")
            detail = f"the model call failed: {exc}"
            BUS.emit("error", "model", detail, turn=turn)
            return {"reply": detail, "messages": [AIMessage(content=detail)]}

        thought = _text_of(reply)
        if thought and reply.tool_calls:
            BUS.emit("thought", thought[:300], turn=turn)

        return {"messages": [reply], "steps": state.get("steps", 0) + 1}

    async def act(state: AgentState) -> dict[str, Any]:
        """Run the tools the model asked for, in order.

        Sequential on purpose: "stop the old one, then start the new one" is a
        common pair, and running those concurrently is a race that sometimes
        leaves the pipeline empty.
        """
        turn = state["turn"]
        tools = {t.name: t for t in build_tools(perception, memory, turn)}
        last: AIMessage = state["messages"][-1]
        results: list[ToolMessage] = []

        for call in last.tool_calls:
            tool = tools.get(call["name"])
            if tool is None:
                text = f"No such tool: {call['name']}."
            else:
                try:
                    text = await tool.ainvoke(call["args"])
                except Exception as exc:                # noqa: BLE001
                    # A tool that raises would end the turn. The agent can
                    # almost always do something useful with the reason.
                    log.warning("tool %s failed: %s", call["name"], exc)
                    text = f"That call failed: {exc}. Check the arguments and try again."
                    BUS.emit("error", call["name"], str(exc)[:300], turn=turn)
            results.append(ToolMessage(content=str(text), tool_call_id=call["id"]))

        return {"messages": results}

    async def finish(state: AgentState) -> dict[str, Any]:
        """The reply the operator hears."""
        turn = state["turn"]
        last = state["messages"][-1]
        reply = state.get("reply") or (
            _text_of(last) if isinstance(last, AIMessage) else "")

        if not reply:
            reply = "Done."

        elapsed = time.time() - state.get("started", time.time())
        BUS.emit("reply", reply, turn=turn, seconds=round(elapsed, 2))
        BUS.emit("timer", "turn complete", f"{elapsed:.2f}s",
                 turn=turn, seconds=round(elapsed, 2))
        return {"reply": reply}

    def route(state: AgentState) -> Literal["act", "finish"]:
        last = state["messages"][-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return "finish"
        if state.get("steps", 0) >= CFG.max_steps:
            # Stop rather than loop. Whatever is installed stays installed.
            BUS.emit("error", "step limit",
                     f"stopped after {CFG.max_steps} tool calls", turn=state["turn"])
            return "finish"
        return "act"

    graph = StateGraph(AgentState)
    graph.add_node("ground", ground)
    graph.add_node("think", think)
    graph.add_node("act", act)
    graph.add_node("finish", finish)

    graph.add_edge(START, "ground")
    graph.add_edge("ground", "think")
    graph.add_conditional_edges("think", route, {"act": "act", "finish": "finish"})
    graph.add_edge("act", "think")
    graph.add_edge("finish", END)

    return graph.compile()


def opening_messages(instruction: str, origin: str,
                     image_urls: list[str] | None = None) -> list[AnyMessage]:
    """The system prompt, then the turn itself."""
    messages: list[AnyMessage] = [SystemMessage(content=system_prompt())]

    if origin == "event":
        messages.append(SystemMessage(content=EVENT_TURN))

    if image_urls:
        parts: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
        for url in image_urls:
            parts.append({"type": "image_url", "image_url": {"url": url}})
        messages.append(HumanMessage(content=parts))
    else:
        messages.append(HumanMessage(content=instruction))

    return messages
