"""Jev controllers. Jev only picks from the label sets; it never generates text.

`HybridAgentPolicy` keeps Jev on every decision but exposes an `escalation_threshold`: the graph's
router sends any Jev decision below it to the `llm_controller` node, and passages Jev is unsure about
are re-graded by the LLM.
"""

import asyncio

from typesafe_sdk import Choice, Noul

from app.agent.policies.base import (
    ANALYZE_ACTIONS,
    EVIDENCE_ACTIONS,
    EVIDENCE_STATUS,
    STRATEGIES,
    AgentPolicy,
    EvidenceView,
    passages,
)
from app.agent.state import Decision
from app.retrieval import Doc
from app.services import JevService, Meter


def noul_confidence(p: float) -> float:
    """Map p(yes) to decision confidence: 0 at p=0.5, 1 at p in {0, 1}."""
    return abs(2 * p - 1)


class JevAgentPolicy:
    name = "jev"
    escalation_threshold: float | None = None
    controller: AgentPolicy | None = None

    def __init__(self, jev: JevService, corpus_description: str):
        self.jev = jev
        self.corpus = corpus_description

    async def route_query(self, query: str, meter: Meter) -> Decision:
        r = await self.jev.ask(
            {"question": query, "internal_corpus": self.corpus},
            {
                "action": Choice(
                    instructions="How should a retrieval agent handle this question?",
                    criteria=ANALYZE_ACTIONS,
                ),
                "strategy": Choice(
                    instructions="Which retrieval strategy best fits this question?", criteria=STRATEGIES
                ),
            },
            meter,
            "route_query",
        )
        action, strategy = r.choices["action"], r.choices["strategy"]
        return Decision(
            stage="analyze",
            action=action.choice,
            strategy=strategy.choice,
            confidence=min(action.confidence, strategy.confidence),
            decided_by="jev",
        )

    async def grade_evidence(self, query: str, docs: list[Doc], meter: Meter) -> dict[str, float]:
        question = {
            "relevant": Noul(
                instructions="Does the passage contain specific facts or analysis that help answer "
                "the question?",
                criteria={
                    "true": "Directly addresses the question with specifics.",
                    "false": "Off-topic, too vague, or unhelpful for this question.",
                },
            )
        }
        rs = await asyncio.gather(
            *(self.jev.ask({"question": query, "passage": d["text"]}, question, meter, "grade") for d in docs)
        )
        return {d["id"]: r.nouls["relevant"].noul for d, r in zip(docs, rs, strict=True)}

    async def choose_next_action(self, view: EvidenceView, meter: Meter) -> Decision:
        r = await self.jev.ask(
            {**view, "evidence": passages(view["evidence"])},
            {
                "evidence_status": Choice(
                    instructions="How well does the evidence answer the question?", criteria=EVIDENCE_STATUS
                ),
                "next_action": Choice(
                    instructions="What should the agent do next? Prefer ANSWER or ABSTAIN when the attempts "
                    "already used make further retrieval unlikely to help.",
                    criteria=EVIDENCE_ACTIONS,
                ),
            },
            meter,
            "next_action",
        )
        status, action = r.choices["evidence_status"], r.choices["next_action"]
        return Decision(
            stage="evidence",
            action=action.choice,
            evidence_status=status.choice,
            confidence=action.confidence,
            decided_by="jev",
        )

    async def verify_answer(self, query: str, answer: str, evidence: list[Doc], meter: Meter) -> Decision:
        r = await self.jev.ask(
            {"question": query, "answer": answer, "evidence": passages(evidence)},
            {
                "supported": Noul(
                    instructions="Is every factual claim in the answer supported by the evidence?"
                )
            },
            meter,
            "verify",
        )
        p = r.nouls["supported"].noul
        return Decision(
            stage="verify",
            action="ACCEPT" if p >= 0.5 else "RETRY",
            confidence=noul_confidence(p),
            decided_by="jev",
        )


class HybridAgentPolicy(JevAgentPolicy):
    name = "hybrid"

    def __init__(
        self,
        jev: JevService,
        corpus_description: str,
        controller: AgentPolicy,
        escalation_threshold: float,
        uncertainty_band: float,
    ):
        super().__init__(jev, corpus_description)
        self.controller = controller
        self.escalation_threshold = escalation_threshold
        self.uncertainty_band = uncertainty_band

    async def grade_evidence(self, query: str, docs: list[Doc], meter: Meter) -> dict[str, float]:
        scores = await super().grade_evidence(query, docs, meter)
        unsure = [d for d in docs if abs(scores[d["id"]] - 0.5) < self.uncertainty_band]
        if unsure and self.controller is not None:
            meter.update({"escalations.grade": len(unsure)})
            scores |= await self.controller.grade_evidence(query, unsure, meter)
        return scores
