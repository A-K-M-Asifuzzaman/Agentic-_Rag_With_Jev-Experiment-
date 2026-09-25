"""`ajev` command line: ingest | ask | eval | trace | graph | serve."""

import argparse
import asyncio
import json
import uuid
from pathlib import Path

from langchain_core.runnables import RunnableConfig

from app.agent.graph import PIPELINES, build_context, build_graph, mermaid, trace_events
from app.config import Settings

DEFAULT_DB = ".data/checkpoints.sqlite"


def _print_event(event: str, data: dict) -> None:
    scope = f"{data['scope']}/" if data.get("scope") else ""
    match event:
        case "generation.token":
            print(data["text"], end="", flush=True)
        case "node.started":
            print(f"\n> {scope}{data['node']}")
        case "jev.decision" | "llm.decision" | "rule.decision":
            extra = f" strategy={data['strategy']}" if data.get("strategy") else ""
            print(f"  {event}: {data['stage']} -> {data['action']} (conf {data['confidence']:.2f}){extra}")
        case "route.selected":
            print(f"  route: {data['action']} -> {data['target']}")
        case "retrieval.completed":
            print(f"  {scope}retrieved {len(data['doc_ids'])} [{data['strategy']}] for: {data['query']}")
        case "graph.completed":
            print(
                "\n\n"
                + json.dumps(
                    {k: data[k] for k in ("citations", "abstained", "evidence_status", "metrics")}, indent=1
                )
            )


async def _ask(question: str, pipeline: str, db: str) -> None:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    settings = Settings.from_env()
    Path(db).parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        thread_id = f"ask:{pipeline}:{uuid.uuid4().hex[:8]}"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}
        async for event, data in trace_events(
            build_graph(pipeline, saver), question, build_context(pipeline, settings), config
        ):
            _print_event(event, data)
        print(f"\nthread_id: {thread_id}  (replay: ajev trace {thread_id})")


async def _trace(thread_id: str, db: str) -> None:
    """Print the checkpointed execution path of a past run, oldest step first."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        parts = thread_id.split(":")  # "<run|ask>:<pipeline>:<id>"
        graph = build_graph(parts[1] if len(parts) > 2 and parts[1] in PIPELINES else "hybrid", saver)
        history = [s async for s in graph.aget_state_history({"configurable": {"thread_id": thread_id}})]
        if not history:
            raise SystemExit(f"no checkpoints for {thread_id} in {db}")
        for snap in reversed(history):
            pending = snap.values.get("pending") or {}
            print(
                f"step {(snap.metadata or {}).get('step'):>3}  next={list(snap.next)}  "
                f"pending={pending.get('stage')}:{pending.get('action')}  "
                f"docs={len(snap.values.get('documents') or [])}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(prog="ajev", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="chunk and index data/documents/*.pdf")
    ask = sub.add_parser("ask", help="run one question and stream its trace")
    ask.add_argument("question")
    ask.add_argument("--pipeline", choices=PIPELINES, default="hybrid")
    ask.add_argument("--db", default=DEFAULT_DB)
    ev = sub.add_parser("eval", help="benchmark pipelines on a JSONL dataset")
    ev.add_argument("--dataset", type=Path, default=Path("data/eval/ev_oil_questions.jsonl"))
    ev.add_argument("--pipelines", default=",".join(PIPELINES))
    ev.add_argument("--limit", type=int)
    ev.add_argument("--run-id", default=None, help="reuse an id to resume an interrupted run")
    tr = sub.add_parser("trace", help="replay the checkpoint history of a thread")
    tr.add_argument("thread_id")
    tr.add_argument("--db", default=DEFAULT_DB, help="runs/<run_id>/checkpoints.sqlite for eval threads")
    gr = sub.add_parser("graph", help="print the Mermaid diagram of a pipeline graph")
    gr.add_argument("--pipeline", choices=PIPELINES, default="hybrid")
    sv = sub.add_parser("serve", help="start the API and trace UI")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    match args.cmd:
        case "ingest":
            from app.retrieval import ingest

            print(f"indexed {ingest(Settings.from_env())} chunks")
        case "ask":
            asyncio.run(_ask(args.question, args.pipeline, args.db))
        case "eval":
            from app.eval import run_eval, to_markdown

            run_id = args.run_id or uuid.uuid4().hex[:8]
            summary = asyncio.run(run_eval(args.dataset, args.pipelines.split(","), run_id, args.limit))
            print(f"\nrun {run_id}\n" + to_markdown(summary))
        case "trace":
            asyncio.run(_trace(args.thread_id, args.db))
        case "graph":
            print(mermaid(args.pipeline))
        case "serve":
            import uvicorn

            uvicorn.run("app.api:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
