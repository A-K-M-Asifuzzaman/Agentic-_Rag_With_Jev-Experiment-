"""LangGraph state. Plain TypedDicts only, so checkpoints serialise without custom type registration.

Accumulating fields declare a reducer; everything else is overwritten by the node that owns it.
`query` is written once by the caller and never changed (the reference notebook overwrote it during
decomposition, which leaked the "Step 1: ..." prompt into generation).
"""

import operator
from typing import Annotated, Literal, NotRequired, TypedDict

from app.retrieval import Doc

Stage = Literal["analyze", "evidence", "verify"]


class Decision(TypedDict):
    stage: Stage
    action: str
    confidence: float
    decided_by: str  # "jev" | "llm" | "rule"
    strategy: NotRequired[str]
    evidence_status: NotRequired[str]
    escalated_from: NotRequired[str]  # the low-confidence Jev action the LLM controller replaced


def merge_docs(left: list[Doc], right: list[Doc]) -> list[Doc]:
    """Union by id; parallel retrieval branches and later rounds add to one evidence pool."""
    out, seen = list(left), {d["id"] for d in left}
    for d in right:
        if d["id"] not in seen:
            seen.add(d["id"])
            out.append(d)
    return out


def merge_dict(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {**left, **right}


def add_metrics(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
    return {k: left.get(k, 0) + right.get(k, 0) for k in left.keys() | right.keys()}


class AgentState(TypedDict, total=False):
    query: str
    active_query: str  # latest rewrite, or the original query
    subquestions: list[str]
    rewritten_queries: Annotated[list[str], operator.add]
    retrieval_strategy: str

    documents: Annotated[list[Doc], merge_docs]
    relevance: Annotated[dict[str, float], merge_dict]  # doc id -> p(relevant), graded once per doc

    pending: Decision  # decision the next conditional edge routes on
    decisions: Annotated[list[Decision], operator.add]
    evidence_status: str

    iteration: Annotated[int, operator.add]
    retrieval_attempts: int
    rewrite_attempts: int
    decomposition_attempts: int
    web_search_attempts: int
    graph_search_attempts: int
    verification_attempts: int

    answer: str
    citations: list[str]
    abstained: bool
    metrics: Annotated[dict[str, float], add_metrics]


class RetrievalInput(TypedDict):
    search_query: str
    search_strategy: str


class RetrievalOutput(TypedDict, total=False):
    documents: Annotated[list[Doc], merge_docs]
    metrics: Annotated[dict[str, float], add_metrics]


class RetrievalState(RetrievalInput, RetrievalOutput, total=False):
    candidates: list[Doc]
