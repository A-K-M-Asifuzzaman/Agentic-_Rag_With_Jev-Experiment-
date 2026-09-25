# Agentic RAG with Jev

This project is a research testbed. It asks whether a cheap structured-decision model can replace a generative
LLM as the controller of an agentic RAG loop without losing answer quality.

- **LangGraph** orchestrates every pipeline as an explicit graph of nodes and conditional edges.
- **Jev**, TypeSafe AI's System One model, makes the routing, grading, next-action and verification decisions. It returns typed labels with calibrated probabilities.
- **A generative LLM** writes answers, query rewrites and sub-questions. In the hybrid pipeline it also re-decides whatever Jev is unsure about.

Architecture, design decisions and the analysis of the reference notebook are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Pipelines

| Pipeline | What decides | Graph |
|---|---|---|
| `standard` | nothing (retrieve → rerank → generate) | standard |
| `llm` | the LLM, on every decision | agentic |
| `jev` | Jev, on every decision | agentic |
| `hybrid` | Jev, with confidence-gated escalation to the LLM | agentic |

The three agentic pipelines share one compiled graph. Only the `AgentPolicy` injected through the
LangGraph runtime context changes. Retrieval (vector, BM25, hybrid RRF, entity graph and web, all
followed by the same cross-encoder reranker), the generator model, prompts and loop budgets are identical.

## Setup

```bash
uv sync
cp .env.example .env        # set OPENAI_API_KEY, TYPESAFE_API_KEY, optional TAVILY_API_KEY
uv run ajev ingest          # chunks data/documents/*.pdf into .data/
```

## Usage

```bash
uv run ajev ask "How much have battery pack costs fallen, and what is the weather in Delhi?" --pipeline hybrid
uv run ajev trace <thread_id>              # replay the checkpointed execution path
uv run ajev eval --pipelines standard,llm,jev,hybrid
uv run ajev graph --pipeline hybrid        # Mermaid diagram generated from the compiled graph
uv run ajev serve                          # http://127.0.0.1:8000 live trace UI (SSE)
```

`ajev eval` writes `runs/<run_id>/results.jsonl`, `summary.md` and `checkpoints.sqlite`. It reports:

- accuracy and key-fact recall against `must_include` facts
- the share of unanswerable questions correctly declined
- the false-abstention rate
- latency
- LLM and Jev call counts, including LLM calls spent on controller decisions
- tokens, cost and the escalation rate

Rerun with the same `--run-id` to resume an interrupted run.

The dataset in `data/eval/ev_oil_questions.jsonl` covers single-fact, lexical, multi-hop, conflicting-evidence,
cross-section, unanswerable, web and direct-knowledge questions about the included report.

## Configuration

Every field of `app/config.py:Settings` can be overridden as `AJEV_<FIELD>`. Examples are
`AJEV_ESCALATION_THRESHOLD=0.8`, `AJEV_LLM_MODEL=openai:gpt-5` and `AJEV_MAX_REWRITE_ATTEMPTS=3`.
Jev pricing depends on your TypeSafe plan. Set `AJEV_JEV_PRICE_IN` and `AJEV_JEV_PRICE_OUT` (USD per 1M tokens)
so that cost comparisons include it.

## Development

```bash
uv run pytest            # offline: fake LLM, fake Jev client, fake retriever
uv run ruff check . && uv run ruff format --check .
uv run mypy app tests
```

## Layout

```text
app/
  config.py            settings (AJEV_* env overrides)
  services.py          metered LLM and Jev clients, stream-event helper
  retrieval.py         ingestion; vector, BM25, hybrid RRF, entity-graph, web retrieval; reranker
  agent/
    state.py           typed graph state and reducers
    nodes.py           graph nodes and runtime context
    routing.py         conditional edges: confidence gate, loop guards, Send fan-out
    graph.py           graph builders, pipeline factory, trace streaming
    policies/          AgentPolicy protocol; LLM, Jev and hybrid policies
  eval.py              benchmark runner and scoring
  api.py, static/      SSE API and trace UI
  cli.py               `ajev` entry point
notebooks/agentic_rag.ipynb   reference agentic RAG this design builds on
```
