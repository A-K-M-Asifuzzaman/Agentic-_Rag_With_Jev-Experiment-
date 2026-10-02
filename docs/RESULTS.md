# Benchmark results

Run on 2026-10-03: `data/eval/ev_oil_questions.jsonl`, 14 questions per pipeline, two repeats each
(runs `cmp-r1` and `cmp-r2`, 112 graph executions, 0 errors). All pipelines used gpt-5-mini for
generation, the same retrieval and reranker, and the same budgets. q13 (live weather) has no reference
facts, so 13 questions per repeat, 26 per pipeline, are scored.

Costs use list prices: gpt-5-mini at $0.25 / $2.00 per 1M input/output tokens, and Jev at $0.042 per 1M
input tokens with output free.

## Summary

| | standard | llm | jev | hybrid |
|---|---|---|---|---|
| Correct, all scored | 24/26 | 23/26 | 22/26 | 22/26 |
| Accuracy, answerable | 90.9% | 90.9% | 81.8% | 86.4% |
| Key-fact recall | 93.9% | 93.9% | 87.9% | 93.2% |
| Unanswerable declined | 4/4 | 3/4 | 4/4 | 3/4 |
| False abstention | 0% | 0% | 8.3% | 0% |
| Latency, mean / p95 (s) | 5.9 / 8.9 | 24.2 / 52.5 | 8.0 / 16.7 | 17.9 / 39.7 |
| LLM calls / question | 1.0 | 11.0 | 1.0 | 3.7 |
| LLM calls spent on decisions | 0 | 9.5 | 0 | 2.4 |
| Jev calls / question | 0 | 0 | 8.8 | 9.3 |
| Cost / question (USD) | 0.00120 | 0.00681 | 0.00142 | 0.00339 |
| Cost / correct answer (USD) | 0.00139 | 0.00830 | 0.00181 | 0.00432 |

## Jev controller vs the LLM controller

The `llm` pipeline is the agentic RAG design without Jev: the reference notebook's LLM router, per-passage
LLM grader and rewrite loop, on the same graph.

- **Cost.** The Jev controller costs 4.8x less per question ($0.00142 vs $0.00681) and 4.6x less per
  correct answer. Its 8.8 decisions per question cost $0.00026 in total, and generation is the remaining
  82% of its cost.
- **Latency.** Jev is about 3x faster on average (8.0 s vs 24.2 s) and at p95 (16.7 s vs 52.5 s).
- **Accuracy.** Jev answered 22/26 correctly against 23/26 for the LLM controller. The gap is a single
  question type:
  - **Multi-hop (q08).** Jev failed both repeats. It chose WEB_SEARCH with 0.96–0.97 confidence for a
    question the corpus answers, then abstained.
  - **Unanswerable questions.** Jev declined all four, while the LLM declined three. In q12 the LLM
    controller sent the question to web search and answered with another paper's discount rate.
- **Hybrid.** The hybrid pipeline escalated 2.4 decisions per question to the LLM and fixed q08 in one of
  the two repeats, at half the LLM pipeline's cost. Its two misses both came from web results replacing
  corpus facts.

## Caveats

- With 26 scored answers per pipeline, differences of one or two answers are within run-to-run noise.
  Treat the accuracy columns as "comparable", not as a ranking.
- This corpus is a single 15-page report and most questions are single-fact. The non-agentic `standard`
  pipeline is the cheapest and scores best here. The agentic controllers need harder multi-hop and
  open-domain questions to earn their extra calls.
- q10 failed in every pipeline. The answer spans two chunks and only one was retrieved. This is a
  retrieval limit shared by all pipelines.
- Jev's high confidence on wrong WEB_SEARCH decisions means its confidence-gated escalation (threshold 0.7)
  does not catch the q08 failure. A per-action threshold, or a corpus-first rule before web search, are
  the obvious next experiments.

Reproduce with `uv run ajev eval --run-id <id>` (twice with different ids). Each run writes a per-run
`summary.md` and `results.jsonl` to `runs/<id>/`.
