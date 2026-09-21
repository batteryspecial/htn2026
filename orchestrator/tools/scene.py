"""Answer scene questions using perception's grounding and the raw camera image."""
from __future__ import annotations

import base64

from langchain_core.messages import HumanMessage

from agent.llm import vision_model
from tools.perception import Perception


async def answer_scene(perception: Perception, question: str, result: dict) -> str:
    snapshot = await perception.snapshot()
    grounding = result.get("prompt") or f"{question}\nDetected: {result.get('summary', '')}"
    if not snapshot:
        return f"No camera snapshot is available. Detection summary only: {result.get('summary', '')}"
    image = base64.b64encode(snapshot).decode()
    reply = await vision_model().ainvoke([HumanMessage(content=[
        {"type": "text", "text": grounding + "\nAnswer briefly from the image and detections. "
         "Be explicit about uncertainty; detections can miss objects."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}},
    ])])
    if isinstance(reply.content, str):
        return reply.content
    return "\n".join(part.get("text", "") for part in reply.content
                     if isinstance(part, dict) and part.get("type") == "text")
