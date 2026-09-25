"""Conditional-edge functions. Policies propose; these functions apply the confidence gate and loop guards
and return the next node (or `Send`s for parallel retrieval).
"""

from langgraph.graph import END
from langgraph.runtime import get_runtime
from langgraph.types import Send

from app.agent.nodes import AgentContext, select_evidence, strategy_for
from app.agent.state import AgentState, Decision
from app.services import emit

ROUTES: dict[str, dict[str, str]] = {
    "analyze": {"RETRIEVE": "retrieval", "DECOMPOSE": "decompose_query", "DIRECT_ANSWER": "generate"},
    "evidence": {
        "ANSWER": "generate",
        "REWRITE": "rewrite_query",
        "GRAPH_SEARCH": "retrieval",
        "WEB_SEARCH": "retrieval",
        "ABSTAIN": "abstain",
    },
    "verify": {"ACCEPT": END, "RETRY": "rewrite_query", "ABSTAIN": "abstain"},
}


def needs_escalation(d: Decision, ctx: AgentContext) -> bool:
    threshold = ctx.policy.escalation_threshold if ctx.policy else None
    return threshold is not None and d["decided_by"] == "jev" and d["confidence"] < threshold


def guard(d: Decision, state: AgentState, ctx: AgentContext) -> str:
    """Return the action to execute: the policy's, unless a budget is spent or grounding is impossible."""
    s, action = ctx.settings, d["action"]
    has_evidence = bool(select_evidence(state, s))
    fallback = "ANSWER" if has_evidence else "ABSTAIN"  # answer from the best evidence we have
    retrieval_spent = state.get("retrieval_attempts", 0) >= s.max_retrieval_attempts

    if d["stage"] == "verify":
        if action == "RETRY" and (
            state.get("verification_attempts", 0) > s.max_verification_attempts
            or state.get("rewrite_attempts", 0) >= s.max_rewrite_attempts
            or retrieval_spent
            or state.get("iteration", 0) >= s.max_steps
        ):
            return "ABSTAIN"
        return action
    if state.get("iteration", 0) >= s.max_steps and d["stage"] == "evidence":
        return fallback
    if d["stage"] == "analyze":
        if action == "DECOMPOSE" and state.get("decomposition_attempts", 0) >= s.max_decomposition_attempts:
            return "RETRIEVE"
        return action
    spent = {
        "REWRITE": retrieval_spent or state.get("rewrite_attempts", 0) >= s.max_rewrite_attempts,
        "WEB_SEARCH": retrieval_spent or state.get("web_search_attempts", 0) >= s.max_web_search_attempts,
        "GRAPH_SEARCH": retrieval_spent
        or state.get("graph_search_attempts", 0) >= s.max_graph_search_attempts,
    }
    if spent.get(action):
        return fallback
    if action == "ANSWER" and not has_evidence:  # never generate "grounded" answers from nothing
        return "ABSTAIN" if retrieval_spent or spent["REWRITE"] else "REWRITE"
    return action


def to_retrieval(state: AgentState) -> list[Send]:
    """Fan out one retrieval-subgraph run per active query (sub-questions run in parallel)."""
    strategy = strategy_for(state)
    queries = state.get("subquestions") or [state.get("active_query") or state["query"]]
    return [Send("retrieval", {"search_query": q, "search_strategy": strategy}) for q in queries]


def route_decision(state: AgentState) -> str | list[Send]:
    ctx: AgentContext = get_runtime(AgentContext).context
    d = state["pending"]
    if needs_escalation(d, ctx):
        emit(
            "route.selected",
            stage=d["stage"],
            action="ESCALATE",
            target="llm_controller",
            confidence=d["confidence"],
        )
        return "llm_controller"
    action = guard(d, state, ctx)
    target = ROUTES[d["stage"]][action]
    emit(
        "route.selected",
        stage=d["stage"],
        action=action,
        target=target,
        proposed=d["action"],
        overridden=action != d["action"],
    )
    if target == "retrieval":
        return to_retrieval({**state, "pending": {**d, "action": action}})
    return target
