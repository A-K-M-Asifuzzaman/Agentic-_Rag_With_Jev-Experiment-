"""Graph assembly, pipeline factory and trace streaming.

Four pipelines, two graphs:
  standard           -> STANDARD graph (retrieve -> rerank -> generate), no controller
  llm | jev | hybrid -> the same AGENTIC graph; only the AgentPolicy in the runtime context differs
"""

from collections.abc import AsyncIterator
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agent import nodes
from app.agent.nodes import AgentContext
from app.agent.policies.base import AgentPolicy
from app.agent.policies.jev_policy import HybridAgentPolicy, JevAgentPolicy
from app.agent.policies.llm_policy import LLMAgentPolicy
from app.agent.routing import route_decision, to_retrieval
from app.agent.state import AgentState, RetrievalInput, RetrievalOutput, RetrievalState
from app.config import Settings
from app.retrieval import Retriever, RetrieverLike
from app.services import JevService, LLMService

PIPELINES = ("standard", "llm", "jev", "hybrid")
Graph = CompiledStateGraph[Any, Any, Any, Any]


def build_retrieval_subgraph() -> Graph:
    """retrieve -> rerank. Shared by every pipeline so retrieval is identical across experiments."""
    g = StateGraph(
        RetrievalState,
        context_schema=AgentContext,
        input_schema=RetrievalInput,
        output_schema=RetrievalOutput,
    )
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("rerank", nodes.rerank)
    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", END)
    return g.compile(name="retrieval")


def build_agentic_graph(checkpointer: BaseCheckpointSaver | None = None) -> Graph:
    g = StateGraph(AgentState, context_schema=AgentContext)
    g.add_node("analyze_query", nodes.analyze_query)
    g.add_node("decompose_query", nodes.decompose_query)
    g.add_node("retrieval", build_retrieval_subgraph())
    g.add_node("grade_documents", nodes.grade_documents)
    g.add_node("evaluate_evidence", nodes.evaluate_evidence)
    g.add_node("rewrite_query", nodes.rewrite_query)
    g.add_node("llm_controller", nodes.llm_controller)
    g.add_node("generate", nodes.generate)
    g.add_node("verify_answer", nodes.verify_answer)
    g.add_node("abstain", nodes.abstain)

    g.add_edge(START, "analyze_query")
    g.add_conditional_edges(
        "analyze_query", route_decision, ["llm_controller", "retrieval", "decompose_query", "generate"]
    )
    g.add_conditional_edges("decompose_query", to_retrieval, ["retrieval"])
    g.add_edge("retrieval", "grade_documents")
    g.add_edge("grade_documents", "evaluate_evidence")
    g.add_conditional_edges(
        "evaluate_evidence",
        route_decision,
        ["llm_controller", "generate", "rewrite_query", "retrieval", "abstain"],
    )
    g.add_conditional_edges("rewrite_query", to_retrieval, ["retrieval"])
    g.add_conditional_edges(
        "llm_controller",
        route_decision,
        ["retrieval", "decompose_query", "generate", "rewrite_query", "abstain", END],
    )
    g.add_edge("generate", "verify_answer")
    g.add_conditional_edges(
        "verify_answer", route_decision, ["llm_controller", "rewrite_query", "abstain", END]
    )
    g.add_edge("abstain", END)
    return g.compile(checkpointer=checkpointer, name="agentic_rag")


def build_standard_graph(checkpointer: BaseCheckpointSaver | None = None) -> Graph:
    g = StateGraph(AgentState, context_schema=AgentContext)
    g.add_node("retrieval", build_retrieval_subgraph())
    g.add_node("generate", nodes.generate)
    g.add_conditional_edges(START, to_retrieval, ["retrieval"])
    g.add_edge("retrieval", "generate")
    g.add_edge("generate", END)
    return g.compile(checkpointer=checkpointer, name="standard_rag")


def build_graph(pipeline: str, checkpointer: BaseCheckpointSaver | None = None) -> Graph:
    if pipeline not in PIPELINES:
        raise ValueError(f"unknown pipeline {pipeline!r}; choose from {PIPELINES}")
    return build_standard_graph(checkpointer) if pipeline == "standard" else build_agentic_graph(checkpointer)


def mermaid(pipeline: str) -> str:
    """Mermaid for a pipeline. Subgraph ids like `retrieval\\3aretrieve` become `retrieval__retrieve`
    so GitHub renders them and the trace UI can style visited nodes."""
    return build_graph(pipeline).get_graph(xray=1).draw_mermaid().replace("\\3a", "__")


def build_policy(pipeline: str, settings: Settings) -> AgentPolicy | None:
    corpus = settings.corpus_description
    if pipeline == "standard":
        return None
    llm_policy = LLMAgentPolicy(LLMService.from_settings(settings.controller_model, settings), corpus)
    if pipeline == "llm":
        return llm_policy
    jev = JevService.from_settings(settings)
    if pipeline == "jev":
        return JevAgentPolicy(jev, corpus)
    return HybridAgentPolicy(
        jev, corpus, llm_policy, settings.escalation_threshold, settings.grade_uncertainty_band
    )


def build_context(pipeline: str, settings: Settings, retriever: RetrieverLike | None = None) -> AgentContext:
    return AgentContext(
        policy=build_policy(pipeline, settings),
        llm=LLMService.from_settings(settings.llm_model, settings),
        retriever=retriever or Retriever.load(settings),
        settings=settings,
    )


def initial_state(query: str) -> AgentState:
    # The standard pipeline never runs analyze_query, so it keeps this fixed strategy.
    return AgentState(query=query, active_query=query, retrieval_strategy="HYBRID")


def summarize(state: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "answer",
        "citations",
        "abstained",
        "evidence_status",
        "subquestions",
        "rewritten_queries",
        "decisions",
        "retrieval_attempts",
        "rewrite_attempts",
        "iteration",
    )
    out = {k: state.get(k) for k in keys}
    out["metrics"] = {k: round(v, 6) for k, v in sorted((state.get("metrics") or {}).items())}
    return out


async def trace_events(
    graph: Graph, query: str, context: AgentContext, config: RunnableConfig
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Map LangGraph's tasks/custom/messages streams onto the trace event vocabulary used by the UI."""
    yield (
        "graph.started",
        {"graph": graph.name, "query": query, "thread_id": config.get("configurable", {}).get("thread_id")},
    )
    stream: AsyncIterator[Any] = graph.astream(
        initial_state(query),
        config,
        context=context,
        stream_mode=["tasks", "custom", "messages"],
        subgraphs=True,
    )
    async for ns, mode, chunk in stream:
        scope = "/".join(part.split(":")[0] for part in ns)
        if mode == "tasks":
            if "result" in chunk:
                yield "node.completed", {"node": chunk["name"], "scope": scope, "error": chunk.get("error")}
            else:
                yield "node.started", {"node": chunk["name"], "scope": scope}
        elif mode == "custom":
            data = dict(chunk)
            yield data.pop("event"), {**data, "scope": scope}
        elif mode == "messages":
            msg, meta = chunk
            if meta.get("langgraph_node") == "generate" and msg.text:
                yield "generation.token", {"text": msg.text}
    final = await graph.aget_state(config)
    yield "graph.completed", summarize(final.values)
