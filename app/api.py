"""FastAPI server: SSE stream of graph execution plus a single-page trace viewer.

Local research tool: no auth, binds to 127.0.0.1 by default. Put it behind auth before exposing it.
"""

import json
import uuid
from collections.abc import AsyncIterator
from functools import cache
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.graph import build_context, build_graph, mermaid, trace_events
from app.agent.nodes import AgentContext
from app.config import Settings
from app.retrieval import Retriever

Pipeline = Literal["standard", "llm", "jev", "hybrid"]
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="Agentic RAG with Jev")
# ponytail: in-process checkpointer, lost on restart and grows with threads. Use AsyncSqliteSaver or a
# Postgres saver if server-side replay must survive restarts.
checkpointer = InMemorySaver()


@cache
def _settings() -> Settings:
    return Settings.from_env()


@cache
def _retriever() -> Retriever:
    return Retriever.load(_settings())


@cache
def _runtime(pipeline: str) -> tuple:
    return build_graph(pipeline, checkpointer), build_context(pipeline, _settings(), _retriever())


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/graph/{pipeline}")
def graph_diagram(pipeline: Pipeline) -> dict[str, str]:
    return {"mermaid": mermaid(pipeline)}


@app.get("/api/stream")
async def stream(
    q: Annotated[str, Query(min_length=1, max_length=2000)], pipeline: Pipeline = "hybrid"
) -> StreamingResponse:
    graph, ctx = _runtime(pipeline)
    thread_id = f"api:{pipeline}:{uuid.uuid4().hex[:12]}"
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}
    return StreamingResponse(
        _sse(graph, q, ctx, config),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _sse(graph, q: str, ctx: AgentContext, config: RunnableConfig) -> AsyncIterator[str]:  # type: ignore[no-untyped-def]
    try:
        async for event, data in trace_events(graph, q, ctx, config):
            yield f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"
    except Exception as exc:  # surface failures to the UI instead of silently closing the stream
        yield f"event: graph.error\ndata: {json.dumps({'error': f'{type(exc).__name__}: {exc}'})}\n\n"


@app.get("/api/threads/{thread_id}/history")
async def history(thread_id: str) -> list[dict]:
    parts = thread_id.split(":")
    if len(parts) != 3 or parts[1] not in ("standard", "llm", "jev", "hybrid"):
        raise HTTPException(404, "unknown thread")
    graph, _ = _runtime(parts[1])
    snaps = [s async for s in graph.aget_state_history({"configurable": {"thread_id": thread_id}})]
    return [
        {"step": (s.metadata or {}).get("step"), "next": list(s.next), "pending": s.values.get("pending")}
        for s in reversed(snaps)
    ]
