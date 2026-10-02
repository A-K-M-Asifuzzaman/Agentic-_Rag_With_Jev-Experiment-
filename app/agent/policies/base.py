"""The controller interface every pipeline variant plugs into the same LangGraph.

Label sets and their descriptions live here and are shared verbatim by the Jev and LLM policies, so the
only experimental variable is which model makes the decision.
"""

from typing import Protocol, TypedDict

from app.agent.state import Decision
from app.retrieval import Doc
from app.services import Meter

ANALYZE_ACTIONS = {
    "RETRIEVE": "The question needs facts from the documents or the web and asks for one thing.",
    "DECOMPOSE": "The question asks for several distinct facts that each need their own retrieval.",
    "DIRECT_ANSWER": "General knowledge or conversational; answering needs no retrieval.",
}

STRATEGIES = {
    "VECTOR": "Conceptual or paraphrased question; semantic similarity search.",
    "BM25": "Exact terms, names, figures or phrases that should match literally.",
    "HYBRID": "Mix of concepts and specific terms; fuse semantic and keyword search.",
    "GRAPH": "Needs connecting facts spread across several sections or entities.",
    "WEB": "Current, real-time or post-report information the internal corpus cannot contain.",
}

EVIDENCE_STATUS = {
    "SUFFICIENT": "The evidence fully answers every part of the question.",
    "PARTIAL": "The evidence answers some parts; at least one part is missing.",
    "INSUFFICIENT": "The evidence does not answer the question.",
    "CONFLICTING": "Passages give contradictory facts for the same point.",
}

EVIDENCE_ACTIONS = {
    "ANSWER": "Evidence is enough (or conflicting but reportable); generate the answer now.",
    "REWRITE": "Evidence is missing but the corpus likely has it; reformulate the search query.",
    "GRAPH_SEARCH": "Related facts were found but connected facts elsewhere in the corpus are needed.",
    "WEB_SEARCH": "The missing information is current or outside the internal corpus.",
    "ABSTAIN": "Further search is unlikely to help; decline to answer.",
}

VERIFY_ACTIONS = {
    "ACCEPT": "Every factual claim in the answer is supported by the evidence.",
    "RETRY": "The answer contains unsupported claims; gather better evidence.",
}


class EvidenceView(TypedDict):
    """What a policy sees when choosing the next action after grading."""

    query: str
    active_query: str
    subquestions: list[str]
    retrieval_strategy: str
    evidence: list[Doc]
    attempts: dict[str, int]


class AgentPolicy(Protocol):
    name: str
    escalation_threshold: float | None  # None: never escalate
    controller: "AgentPolicy | None"  # policy the llm_controller node escalates to

    async def route_query(self, query: str, meter: Meter) -> Decision: ...

    async def grade_evidence(self, query: str, docs: list[Doc], meter: Meter) -> dict[str, float]: ...

    async def choose_next_action(self, view: EvidenceView, meter: Meter) -> Decision: ...

    async def verify_answer(self, query: str, answer: str, evidence: list[Doc], meter: Meter) -> Decision: ...


def passages(docs: list[Doc]) -> list[dict[str, str]]:
    # Untruncated: controllers must judge exactly the text the generator answers from.
    return [{"id": d["id"], "text": d["text"]} for d in docs]
