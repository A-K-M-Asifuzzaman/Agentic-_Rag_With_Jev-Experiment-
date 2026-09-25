"""Graph nodes. Each does one thing and returns a partial state update; routing lives in `routing.py`.

Dependencies (policy, generator LLM, retriever, settings) arrive through LangGraph's runtime context,
so the same compiled graph serves every controller policy.
"""

from dataclasses import dataclass
from typing import cast

from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

from app.agent.policies.base import AgentPolicy, EvidenceView
from app.agent.state import AgentState, Decision, RetrievalState
from app.config import Settings
from app.retrieval import Doc, RetrieverLike
from app.services import LLMService, Meter, emit


@dataclass
class AgentContext:
    policy: AgentPolicy | None  # None for the standard (non-agentic) pipeline
    llm: LLMService  # generation, rewriting, decomposition
    retriever: RetrieverLike
    settings: Settings


Rt = Runtime[AgentContext]
ATTEMPT_KEYS = ("retrieval_attempts", "rewrite_attempts", "web_search_attempts", "graph_search_attempts")


# --------------------------------------------------------------------------- helpers


def select_evidence(state: AgentState, settings: Settings) -> list[Doc]:
    """Graded-relevant passages (or all passages when nothing was graded), best first."""
    docs = state.get("documents") or []
    rel = state.get("relevance") or {}
    if rel:
        docs = [d for d in docs if rel.get(d["id"], 0.0) >= settings.relevance_threshold]
    return sorted(docs, key=lambda d: (rel.get(d["id"], 0.0), d["score"]), reverse=True)[
        : settings.max_evidence
    ]


def evidence_view(state: AgentState, settings: Settings) -> EvidenceView:
    return EvidenceView(
        query=state["query"],
        active_query=state.get("active_query") or state["query"],
        subquestions=state.get("subquestions") or [],
        retrieval_strategy=state.get("retrieval_strategy") or "HYBRID",
        evidence=select_evidence(state, settings),
        attempts={k: cast(dict[str, int], state).get(k, 0) for k in ATTEMPT_KEYS},
    )


def strategy_for(state: AgentState) -> str:
    """Strategy for the retrieval round the pending decision triggers."""
    pending = state.get("pending")
    action = pending["action"] if pending else ""
    return {"WEB_SEARCH": "WEB", "GRAPH_SEARCH": "GRAPH"}.get(
        action, state.get("retrieval_strategy") or "HYBRID"
    )


def _decided(d: Decision, meter: Meter) -> dict:
    emit(f"{d['decided_by']}.decision", **d)
    return {"pending": d, "decisions": [d], "metrics": dict(meter)}


# --------------------------------------------------------------------------- decision nodes


async def analyze_query(state: AgentState, runtime: Rt) -> dict:
    meter = Meter()
    d = await runtime.context.policy.route_query(state["query"], meter)  # type: ignore[union-attr]
    return {
        **_decided(d, meter),
        "iteration": 1,
        "active_query": state["query"],
        "retrieval_strategy": d.get("strategy", "HYBRID"),
    }


async def grade_documents(state: AgentState, runtime: Rt) -> dict:
    graded = state.get("relevance") or {}
    ungraded = [d for d in state.get("documents") or [] if d["id"] not in graded]
    meter = Meter()
    scores = await runtime.context.policy.grade_evidence(state["query"], ungraded, meter) if ungraded else {}  # type: ignore[union-attr]
    thr = runtime.context.settings.relevance_threshold
    emit("evidence.graded", graded=len(scores), relevant=sum(p >= thr for p in scores.values()))
    update: dict = {
        "relevance": scores,
        "metrics": dict(meter),
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }
    match strategy_for(state):
        case "WEB":
            update["web_search_attempts"] = state.get("web_search_attempts", 0) + 1
        case "GRAPH":
            update["graph_search_attempts"] = state.get("graph_search_attempts", 0) + 1
    return update


async def evaluate_evidence(state: AgentState, runtime: Rt) -> dict:
    meter = Meter()
    d = await runtime.context.policy.choose_next_action(evidence_view(state, runtime.context.settings), meter)  # type: ignore[union-attr]
    return {**_decided(d, meter), "iteration": 1, "evidence_status": d.get("evidence_status", "")}


async def verify_answer(state: AgentState, runtime: Rt) -> dict:
    evidence = select_evidence(state, runtime.context.settings)
    meter = Meter()
    if not evidence:  # direct answer path: nothing to ground against
        d = Decision(stage="verify", action="ACCEPT", confidence=1.0, decided_by="rule")
    else:
        d = await runtime.context.policy.verify_answer(state["query"], state["answer"], evidence, meter)  # type: ignore[union-attr]
    return {
        **_decided(d, meter),
        "iteration": 1,
        "verification_attempts": state.get("verification_attempts", 0) + 1,
    }


async def llm_controller(state: AgentState, runtime: Rt) -> dict:
    """Re-decide a low-confidence Jev decision with the LLM policy (hybrid pipeline only)."""
    ctx, pending = runtime.context, state["pending"]
    controller = ctx.policy.controller  # type: ignore[union-attr]
    assert controller is not None, "llm_controller reached without an escalation policy"
    meter = Meter({"escalations": 1, f"escalations.{pending['stage']}": 1})
    match pending["stage"]:
        case "analyze":
            d = await controller.route_query(state["query"], meter)
        case "evidence":
            d = await controller.choose_next_action(evidence_view(state, ctx.settings), meter)
        case _:
            d = await controller.verify_answer(
                state["query"], state["answer"], select_evidence(state, ctx.settings), meter
            )
    d["escalated_from"] = f"{pending['action']}@{pending['confidence']:.2f}"
    update = _decided(d, meter)
    if "strategy" in d:
        update["retrieval_strategy"] = d["strategy"]
    if "evidence_status" in d:
        update["evidence_status"] = d["evidence_status"]
    return update


# --------------------------------------------------------------------------- LLM work nodes


class SubQuestions(BaseModel):
    sub_queries: list[str] = Field(description="2-4 focused, self-contained search queries", max_length=4)


async def decompose_query(state: AgentState, runtime: Rt) -> dict:
    meter = Meter()
    out = await runtime.context.llm.structured(
        SubQuestions,
        "Break the user's question into focused, self-contained search queries, one per distinct piece of "
        "information. Keep each under 20 words.",
        state["query"],
        meter,
        "decompose",
    )
    emit("query.decomposed", subquestions=out.sub_queries)
    return {
        "subquestions": out.sub_queries or [state["query"]],
        "metrics": dict(meter),
        "decomposition_attempts": state.get("decomposition_attempts", 0) + 1,
    }


async def rewrite_query(state: AgentState, runtime: Rt) -> dict:
    meter = Meter()
    tried = [state["query"], *(state.get("rewritten_queries") or [])]
    new = await runtime.context.llm.text(
        "Retrieval did not surface enough relevant evidence. Write ONE new search query that uses different "
        "terminology or targets the missing part more narrowly. Do not repeat earlier queries. "
        "Return only the query.",
        f"Question: {state['query']}\nEvidence status: {state.get('evidence_status', 'unknown')}\n"
        f"Queries already tried:\n" + "\n".join(f"- {q}" for q in tried),
        meter,
        "rewrite",
    )
    emit("query.rewritten", query=new)
    return {
        "active_query": new,
        "rewritten_queries": [new],
        "subquestions": [],
        "metrics": dict(meter),
        "rewrite_attempts": state.get("rewrite_attempts", 0) + 1,
    }


async def generate(state: AgentState, runtime: Rt) -> dict:
    evidence = select_evidence(state, runtime.context.settings)
    meter = Meter()
    if evidence:
        context = "\n\n".join(
            f"[{i}] ({d['source']}{f' p.{d["page"]}' if d['page'] else ''})\n{d['text']}"
            for i, d in enumerate(evidence, 1)
        )
        system = (
            "Answer the question using only the numbered evidence. Cite passages inline as [n]. If passages "
            "disagree, report both figures and their sources. If a part of the question is not covered, say "
            f"so explicitly instead of guessing.\n\nEvidence:\n{context}"
        )
    else:
        system = "Answer from general knowledge. State that no documents were consulted."
    answer = await runtime.context.llm.text(system, state["query"], meter, "generate")
    citations = [d["source"] + (f"#page={d['page']}" if d["page"] else "") for d in evidence]
    return {"answer": answer, "citations": list(dict.fromkeys(citations)), "metrics": dict(meter)}


async def abstain(state: AgentState, runtime: Rt) -> dict:
    status = state.get("evidence_status") or "INSUFFICIENT"
    return {
        "abstained": True,
        "citations": [],
        "answer": (
            "I can't answer this reliably from the available sources "
            f"(evidence {status.lower()} after {state.get('retrieval_attempts', 0)} retrieval rounds)."
        ),
    }


# --------------------------------------------------------------------------- retrieval subgraph nodes


async def retrieve(state: RetrievalState, runtime: Rt) -> dict:
    q, strategy = state["search_query"], state["search_strategy"]
    emit("retrieval.started", query=q, strategy=strategy)
    docs = await runtime.context.retriever.search(q, strategy, runtime.context.settings.candidate_k)
    return {"candidates": docs}


async def rerank(state: RetrievalState, runtime: Rt) -> dict:
    q, strategy = state["search_query"], state["search_strategy"]
    docs = await runtime.context.retriever.rerank(
        q, state.get("candidates") or [], runtime.context.settings.top_k
    )
    emit("retrieval.completed", query=q, strategy=strategy, doc_ids=[d["id"] for d in docs])
    return {"documents": docs, "metrics": {"retrieval_calls": 1, f"retrieval_calls.{strategy}": 1}}
