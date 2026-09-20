"""HTTP and WebSocket surface for the agent layer.

Three rules this file keeps:

1. **HTTP is HTTP, WebSockets are WebSockets.** `/chat` is a request that
   returns a reply. `/ws/trace` is a stream the browser subscribes to. The
   turn's intermediate steps go on the socket, never dribbled into the HTTP
   response.
2. **Nothing here reasons.** Routes marshal input, hand it to the runner, and
   marshal output. The agent lives in `agent/`.
3. **A route never leaves the pipeline's failure unexplained.** Perception
   being down is a thing the operator should be told, not a 500.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from agent.runner import AgentRunner, Attachment
from app.trace import BUS, TraceEntry
from config import CFG
from events.watcher import EventWatcher
from memory.store import PhraseMemory
from tools.perception import Perception

log = logging.getLogger("orchestrator.api")

#: Trace kinds the console's stage strip understands. Everything else stays
#: on /ws/trace, where it is shown in full rather than flattened into a stage
#: name that does not fit.
STAGE_OF = {
    "user": "received",
    "reply": "active",
}


def create_app(*, watch_events: bool = True) -> FastAPI:
    perception = Perception()
    memory = PhraseMemory()
    runner = AgentRunner(perception, memory)
    watcher = EventWatcher(runner, perception)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI):
        if watch_events:
            watcher.start()
        try:
            yield
        finally:
            await watcher.stop()
            await perception.aclose()

    app = FastAPI(title="Retask orchestrator", lifespan=lifespan)

    # The console is a page on another origin. Without this every call it
    # makes fails before reaching a route, and the failure looks like
    # "server down" rather than "CORS".
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])

    app.state.runner = runner
    app.state.perception = perception
    app.state.memory = memory
    app.state.watcher = watcher

    # 1. Turns -----------------------------------------------------------

    @app.post("/chat")
    async def chat(text: str = Form(default=""),
                   images: list[UploadFile] = File(default=[])):
        """One operator turn, with optional photos.

        An upload is registered with perception as a reference *before* the
        agent thinks, so it is handed a ref_id rather than having to discover
        it needs one. "Track that human" with a photo is two steps and this
        is the first.
        """
        if not text.strip() and not images:
            return JSONResponse(status_code=422,
                                content={"detail": "say something, or attach an image"})

        attachments = [
            Attachment(filename=f.filename or "upload.jpg",
                       content_type=f.content_type or "image/jpeg",
                       data=await f.read())
            for f in images
        ]

        result = await runner.user_turn(text, attachments)
        return {"turn": result.turn, "reply": result.reply,
                "seconds": round(result.seconds, 3)}

    @app.post("/instruction")
    async def instruction(body: dict[str, Any]):
        """The console's original contract, kept so LIVE mode never breaks.

        Same loop, no attachments. New work should use /chat.
        """
        text = str(body.get("text") or "").strip()
        if not text:
            return JSONResponse(status_code=422, content={"detail": "empty instruction"})

        result = await runner.user_turn(text)
        return {"instruction_id": result.turn, "reply": result.reply,
                "seconds": round(result.seconds, 3)}

    # 2. Streams ---------------------------------------------------------

    @app.websocket("/ws/trace")
    async def ws_trace(ws: WebSocket):
        """Everything the agent does, as it does it."""
        await ws.accept()
        queue = BUS.subscribe(replay=True)
        try:
            while True:
                entry: TraceEntry = await queue.get()
                await ws.send_json(entry.as_dict())
        except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
            pass
        finally:
            BUS.unsubscribe(queue)

    @app.websocket("/ws/status")
    async def ws_status(ws: WebSocket):
        """The stage strip, derived from the trace.

        Only the few kinds with a stage equivalent are forwarded. Inventing a
        stage name for a tool call would put unknown entries on the strip.
        """
        await ws.accept()
        queue = BUS.subscribe(replay=False)
        try:
            while True:
                entry: TraceEntry = await queue.get()
                stage = STAGE_OF.get(entry.kind)
                if entry.kind == "error":
                    stage = "failed"
                elif entry.kind == "tool_result" and entry.label == "start_behavior":
                    stage = "applied"
                if not stage:
                    continue
                await ws.send_json({
                    "instruction_id": entry.turn,
                    "stage": stage,
                    "ts": entry.ts,
                    "detail": entry.detail or entry.label,
                })
        except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
            pass
        finally:
            BUS.unsubscribe(queue)

    # 3. Status ----------------------------------------------------------

    @app.get("/health")
    async def health():
        downstream = await perception.health()
        return {
            "status": "ok",
            "model": CFG.model,
            "provider": CFG.provider,
            "has_key": bool(CFG.api_key),
            "perception": {
                "base": perception.base,
                "up": downstream["ok"],
                "detail": downstream.get("error") or downstream.get("status"),
            },
            "trace_listeners": BUS.listeners,
        }

    @app.get("/memory")
    async def memory_lookup(subject: str, limit: int = 8):
        """What the phrase memory knows. For tuning, and for the demo story."""
        return {"subject": subject,
                "suggestions": [vars(r) for r in memory.suggest(subject, limit)]}

    # 4. The console, when it has been built -----------------------------
    #
    # Vite's dev server on :3000 is the normal path. This is for a single
    # process on a demo machine, and is mounted last so it never shadows an
    # API route.
    if CFG.static_dir.is_dir():
        app.mount("/", StaticFiles(directory=CFG.static_dir, html=True), name="console")
        log.info("serving the console from %s", CFG.static_dir)

    return app
