"""Benchmark the pipelines on one dataset with identical retrieval, generator and corpus.

Two stores, deliberately separate:
  runs/<run_id>/checkpoints.sqlite  LangGraph execution state (replay with `ajev trace <thread_id> --db ...`)
  runs/<run_id>/results.jsonl       research telemetry, one row per (pipeline, question); resumable
"""

import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.agent.graph import build_context, build_graph, initial_state, summarize
from app.config import Settings
from app.retrieval import Retriever

_DECLINE = re.compile(
    r"can't answer|cannot answer|unable to|no information"
    r"|does not (?:provide|mention|contain|include|specify|state)"
    r"|doesn't (?:provide|mention|say)|not (?:covered|provided|mentioned|stated)"
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("\u2013", "-").replace("\u2014", "-"))


def score(item: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """`must_include` facts are all required; `a|b` lists acceptable spellings of one fact."""
    answer = normalize(row.get("answer") or "")
    declined = bool(row.get("abstained")) or bool(_DECLINE.search(answer))
    if not item.get("answerable", True):
        return {"correct": declined, "fact_recall": None, "declined": declined}
    facts = item.get("must_include") or []
    if not facts:
        return {"correct": None, "fact_recall": None, "declined": declined}
    hits = [any(normalize(alt) in answer for alt in fact.split("|")) for fact in facts]
    return {
        "correct": all(hits) and not row.get("abstained"),
        "fact_recall": sum(hits) / len(hits),
        "declined": declined,
    }


# LLM calls spent on controller decisions (vs generation/rewrite/decompose work).
_DECISION_KEYS = {f"llm_calls.{p}" for p in ("route_query", "grade", "next_action", "verify")}


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if not r.get("error")]
    m = [r["metrics"] for r in ok]
    answerable = [r for r in ok if r["answerable"] and r["correct"] is not None]
    unanswerable = [r for r in ok if not r["answerable"]]
    lat = sorted(r["latency_s"] for r in ok)
    return {
        "n": len(rows),
        "errors": len(rows) - len(ok),
        "accuracy": _mean([float(r["correct"]) for r in answerable]),
        "fact_recall": _mean([r["fact_recall"] for r in answerable]),
        "unanswerable_declined": _mean([float(r["correct"]) for r in unanswerable]),
        "false_abstention": _mean([float(bool(r["abstained"])) for r in ok if r["answerable"]]),
        "latency_mean_s": _mean(lat),
        "latency_p95_s": lat[int(0.95 * (len(lat) - 1))] if lat else None,
        "llm_calls": _mean([x.get("llm_calls", 0) for x in m]),
        "jev_calls": _mean([x.get("jev_calls", 0) for x in m]),
        "llm_tokens": _mean([x.get("llm_input_tokens", 0) + x.get("llm_output_tokens", 0) for x in m]),
        "cost_usd": _mean([x.get("llm_cost_usd", 0) + x.get("jev_cost_usd", 0) for x in m]),
        "decision_llm_calls": _mean([sum(v for k, v in x.items() if k in _DECISION_KEYS) for x in m]),
        "escalations": _mean([x.get("escalations", 0) + x.get("escalations.grade", 0) for x in m]),
        "retrieval_rounds": _mean([r["retrieval_attempts"] or 1 for r in ok]),
    }


def to_markdown(summary: dict[str, dict[str, Any]]) -> str:
    cols = list(next(iter(summary.values())).keys())
    lines = ["| pipeline | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    lines += [
        f"| {p} | " + " | ".join("" if s[c] is None else str(s[c]) for c in cols) + " |"
        for p, s in summary.items()
    ]
    return "\n".join(lines) + "\n"


async def run_eval(
    dataset: Path,
    pipelines: list[str],
    run_id: str,
    limit: int | None = None,
    settings: Settings | None = None,
) -> dict[str, dict[str, Any]]:
    settings = settings or Settings.from_env()
    items = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()][:limit]
    out_dir = Path("runs") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    rows = (
        [json.loads(line) for line in results_path.read_text().splitlines()] if results_path.exists() else []
    )
    done = {(r["pipeline"], r["id"]) for r in rows}
    retriever = Retriever.load(settings)

    async with AsyncSqliteSaver.from_conn_string(str(out_dir / "checkpoints.sqlite")) as saver:
        for pipeline in pipelines:
            graph, ctx = build_graph(pipeline, saver), build_context(pipeline, settings, retriever)
            for item in items:
                if (pipeline, item["id"]) in done:
                    continue
                thread_id = f"{run_id}:{pipeline}:{item['id']}"
                row: dict[str, Any] = {
                    "pipeline": pipeline,
                    "id": item["id"],
                    "type": item.get("type"),
                    "answerable": item.get("answerable", True),
                    "thread_id": thread_id,
                }
                start = time.perf_counter()
                try:
                    final = await graph.ainvoke(
                        initial_state(item["question"]),
                        {"configurable": {"thread_id": thread_id}, "recursion_limit": 60},
                        context=ctx,
                    )
                    row |= summarize(final)
                except Exception as exc:  # one failed question must not lose the run
                    row |= {"error": f"{type(exc).__name__}: {exc}", "metrics": {}, "answer": None}
                row["latency_s"] = round(time.perf_counter() - start, 3)
                row |= score(item, row)
                rows.append(row)
                with results_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(
                    f"[{pipeline}] {item['id']}: correct={row['correct']} {row['latency_s']}s"
                    + (f" ERROR {row['error']}" if row.get("error") else "")
                )

    summary = {p: aggregate([r for r in rows if r["pipeline"] == p]) for p in pipelines}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "summary.md").write_text(to_markdown(summary))
    return summary
