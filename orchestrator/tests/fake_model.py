"""A scripted chat model, so the loop can be tested without a provider.

The graph asks for a model, binds tools to it, and awaits one call per step.
That is the whole surface, so this implements exactly that — not a
`BaseChatModel` subclass, which would drag in serialisation machinery the
graph never touches.

A turn is written as `says("...")` or `calls(("tool", {...}))`, and several
are chained:

    model = calls(("probe_phrases", {...})).then(says("Tracking the duck."))

The last turn repeats if the loop runs longer than the script, so a test
about the step cap does not have to spell out ten identical turns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage


@dataclass
class Step:
    """One model turn: some words, and zero or more tool calls."""

    text: str = ""
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


class ScriptedModel:
    """Stands in for `chat_model().bind_tools(...)`."""

    def __init__(self, *script: Step, error: Exception | None = None) -> None:
        self.script: list[Step] = list(script) or [Step(text="Done.")]
        self.error = error
        self.bound: list[Any] = []
        #: Every message list the graph handed over, for asserting on what
        #: the model was actually told.
        self.seen: list[list[Any]] = []
        self.step = 0

    def then(self, following: ScriptedModel) -> ScriptedModel:
        """Append another turn. Returns self, so these chain."""
        self.script.extend(following.script)
        return self

    def bind_tools(self, tools: list[Any]) -> ScriptedModel:
        self.bound = tools
        return self

    async def ainvoke(self, messages: list[Any], **_: Any) -> AIMessage:
        if self.error is not None:
            raise self.error

        self.seen.append(list(messages))
        turn = self.script[min(self.step, len(self.script) - 1)]
        self.step += 1

        return AIMessage(
            content=turn.text,
            tool_calls=[
                {"name": name, "args": args, "id": f"call_{self.step}_{i}"}
                for i, (name, args) in enumerate(turn.tool_calls)
            ],
        )

    # For assertions -------------------------------------------------------

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self.bound]

    def prompt_text(self, turn: int = 0) -> str:
        """Everything the model was told on one turn, flattened to text."""
        parts = []
        for message in self.seen[turn]:
            content = message.content
            if isinstance(content, str):
                parts.append(content)
            else:
                parts.extend(p.get("text", "") for p in content
                             if isinstance(p, dict))
        return "\n".join(parts)


def says(text: str) -> ScriptedModel:
    """A turn that answers and asks for nothing."""
    return ScriptedModel(Step(text=text))


def calls(*pairs: tuple[str, dict[str, Any]], text: str = "") -> ScriptedModel:
    """A turn that asks for one or more tools, optionally thinking out loud."""
    return ScriptedModel(Step(text=text, tool_calls=list(pairs)))
