"""Baseline controller: a generative LLM makes every decision via structured output (the expensive path)."""

import asyncio
import json
from typing import Literal

from pydantic import BaseModel

from app.agent.policies.base import (
    ANALYZE_ACTIONS,
    EVIDENCE_ACTIONS,
    EVIDENCE_STATUS,
    STRATEGIES,
    VERIFY_ACTIONS,
    AgentPolicy,
    EvidenceView,
    passages,
)
from app.agent.state import Decision
from app.retrieval import Doc
from app.services import LLMService, Meter


class RouteOut(BaseModel):
    action: Literal["RETRIEVE", "DECOMPOSE", "DIRECT_ANSWER"]
    strategy: Literal["VECTOR", "BM25", "HYBRID", "GRAPH", "WEB"]


class GradeOut(BaseModel):
    is_relevant: bool


class NextOut(BaseModel):
    evidence_status: Literal["SUFFICIENT", "PARTIAL", "INSUFFICIENT", "CONFLICTING"]
    next_action: Literal["ANSWER", "REWRITE", "GRAPH_SEARCH", "WEB_SEARCH", "ABSTAIN"]


class VerifyOut(BaseModel):
    action: Literal["ACCEPT", "RETRY"]


def _options(title: str, labels: dict[str, str]) -> str:
    return f"{title}:\n" + "\n".join(f"- {k}: {v}" for k, v in labels.items())


class LLMAgentPolicy:
    name = "llm"
    escalation_threshold: float | None = None
    controller: AgentPolicy | None = None

    def __init__(self, llm: LLMService, corpus_description: str):
        self.llm = llm
        self.corpus = corpus_description

    async def route_query(self, query: str, meter: Meter) -> Decision:
        system = (
            f"You route questions for a retrieval agent. Internal corpus: {self.corpus}\n\n"
            + _options("Choose an action", ANALYZE_ACTIONS)
            + "\n\n"
            + _options("Choose a retrieval strategy (used when retrieving)", STRATEGIES)
        )
        out = await self.llm.structured(RouteOut, system, query, meter, "route_query")
        return Decision(
            stage="analyze", action=out.action, strategy=out.strategy, confidence=1.0, decided_by="llm"
        )

    async def grade_evidence(self, query: str, docs: list[Doc], meter: Meter) -> dict[str, float]:
        # One call per passage, as in the reference notebook, but concurrent.
        system = (
            "You are a relevance evaluator. Return is_relevant=true only if the passage contains specific "
            "facts or analysis that help answer the query; false if off-topic or too vague."
        )
        outs = await asyncio.gather(
            *(
                self.llm.structured(
                    GradeOut, system, f"Query: {query}\n\nPassage:\n{d['text']}", meter, "grade"
                )
                for d in docs
            )
        )
        return {d["id"]: float(o.is_relevant) for d, o in zip(docs, outs, strict=True)}

    async def choose_next_action(self, view: EvidenceView, meter: Meter) -> Decision:
        system = (
            "You control a retrieval agent. Judge the evidence gathered so far and pick the next step. "
            "Attempts already used are listed; prefer ANSWER or ABSTAIN when retries are unlikely "
            "to help.\n\n"
            + _options("Evidence status", EVIDENCE_STATUS)
            + "\n\n"
            + _options("Next action", EVIDENCE_ACTIONS)
        )
        user = json.dumps({**view, "evidence": passages(view["evidence"])}, indent=1)
        out = await self.llm.structured(NextOut, system, user, meter, "next_action")
        return Decision(
            stage="evidence",
            action=out.next_action,
            evidence_status=out.evidence_status,
            confidence=1.0,
            decided_by="llm",
        )

    async def verify_answer(self, query: str, answer: str, evidence: list[Doc], meter: Meter) -> Decision:
        system = "You check answers for grounding.\n\n" + _options("Verdict", VERIFY_ACTIONS)
        user = json.dumps({"question": query, "answer": answer, "evidence": passages(evidence)}, indent=1)
        out = await self.llm.structured(VerifyOut, system, user, meter, "verify")
        return Decision(stage="verify", action=out.action, confidence=1.0, decided_by="llm")
