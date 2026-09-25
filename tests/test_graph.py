from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from app.agent.graph import build_agentic_graph, build_standard_graph, initial_state, trace_events
from app.agent.nodes import AgentContext
from app.config import Settings
from tests.fakes import FakeLLM, FakeRetriever, ScriptedPolicy

SETTINGS = Settings()
CONFIG: RunnableConfig = {"configurable": {"thread_id": "t"}, "recursion_limit": 60}


async def run(
    policy: Any, llm: FakeLLM | None = None, retriever: FakeRetriever | None = None, standard: bool = False
) -> dict[str, Any]:
    ctx = AgentContext(
        policy=policy,
        llm=llm or FakeLLM(),  # type: ignore[arg-type]
        retriever=retriever or FakeRetriever(),
        settings=SETTINGS,
    )
    graph = (build_standard_graph if standard else build_agentic_graph)(InMemorySaver())
    return await graph.ainvoke(initial_state("How much oil do EVs displace?"), CONFIG, context=ctx)


async def test_retrieve_answer_accept():
    out = await run(ScriptedPolicy())
    assert out["answer"] == "generate output"
    assert [d["stage"] for d in out["decisions"]] == ["analyze", "evidence", "verify"]
    assert out["retrieval_attempts"] == 1 and not out.get("abstained")
    assert out["citations"] == ["report.pdf#page=1"]


async def test_rewrite_loop_is_bounded_and_abstains():
    out = await run(ScriptedPolicy(next_actions=("REWRITE",), relevance=0.0))
    assert out["abstained"] is True
    assert out["rewrite_attempts"] == SETTINGS.max_rewrite_attempts
    assert out["retrieval_attempts"] == SETTINGS.max_rewrite_attempts + 1
    assert len(set(out["rewritten_queries"])) >= 1


async def test_answer_without_relevant_evidence_is_not_generated():
    llm = FakeLLM()
    out = await run(ScriptedPolicy(next_actions=("ANSWER",), relevance=0.0), llm=llm)
    assert out["abstained"] is True
    assert "generate" not in llm.calls


async def test_low_confidence_jev_decisions_escalate_to_llm_controller():
    controller = ScriptedPolicy(decided_by="llm")
    out = await run(ScriptedPolicy(confidence=0.3, escalation_threshold=0.7, controller=controller))
    llm_decisions = [d for d in out["decisions"] if d["decided_by"] == "llm"]
    assert [d["stage"] for d in llm_decisions] == ["analyze", "evidence", "verify"]
    assert all(d["escalated_from"].endswith("@0.30") for d in llm_decisions)
    assert out["metrics"]["escalations"] == 3


async def test_confident_jev_decisions_do_not_escalate():
    out = await run(ScriptedPolicy(confidence=0.9, escalation_threshold=0.7, controller=ScriptedPolicy()))
    assert "escalations" not in out["metrics"]


async def test_decomposition_fans_out_parallel_retrieval():
    retriever = FakeRetriever()
    out = await run(
        ScriptedPolicy(route="DECOMPOSE"), llm=FakeLLM(["battery cost", "Delhi weather"]), retriever=retriever
    )
    assert sorted(q for q, _ in retriever.calls) == ["Delhi weather", "battery cost"]
    assert len(out["documents"]) == 2 and out["retrieval_attempts"] == 1


async def test_web_search_budget_falls_back_to_answer():
    retriever = FakeRetriever()
    out = await run(ScriptedPolicy(next_actions=("WEB_SEARCH",)), retriever=retriever)
    assert [s for _, s in retriever.calls] == ["HYBRID", "WEB"]
    assert out["web_search_attempts"] == 1 and out["answer"] == "generate output"


async def test_failed_verification_retries_once_then_abstains():
    out = await run(ScriptedPolicy(verify="RETRY"))
    assert out["verification_attempts"] == 2 and out["rewrite_attempts"] == 1
    assert out["abstained"] is True


async def test_standard_pipeline_retrieves_once_without_controller():
    retriever = FakeRetriever()
    out = await run(None, retriever=retriever, standard=True)
    assert retriever.calls == [("How much oil do EVs displace?", "HYBRID")]
    assert out["answer"] == "generate output" and not out.get("decisions")


async def test_trace_events_cover_graph_lifecycle():
    ctx = AgentContext(policy=ScriptedPolicy(), llm=FakeLLM(), retriever=FakeRetriever(), settings=SETTINGS)  # type: ignore[arg-type]
    events = [e async for e, _ in trace_events(build_agentic_graph(InMemorySaver()), "q", ctx, CONFIG)]
    assert events[0] == "graph.started" and events[-1] == "graph.completed"
    for expected in (
        "node.started",
        "node.completed",
        "jev.decision",
        "route.selected",
        "retrieval.started",
        "retrieval.completed",
    ):
        assert expected in events
