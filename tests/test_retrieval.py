from app.agent.state import add_metrics, merge_docs
from app.config import Settings
from app.retrieval import EntityGraph, Retriever, rrf
from tests.fakes import doc


def test_rrf_rewards_documents_ranked_by_both_retrievers():
    a, b, c = doc("a"), doc("b"), doc("c")
    fused = rrf([[a, b], [c, b]], k=3)
    assert fused[0]["id"] == "b"


def test_entity_graph_links_shared_rare_entities_and_neighbours():
    chunks = [
        doc("p1", "OPEC cut supply in 2022."),
        doc("p2", "Filler text about markets."),
        doc("p3", "The Inflation Reduction Act raised credits."),
        doc("p4", "OPEC behaviour under the Inflation Reduction Act."),
        *[doc(f"n{i}", "Common Words appear everywhere.") for i in range(10)],
    ]
    graph = EntityGraph(chunks)
    assert "common words" not in graph.idf  # too frequent to discriminate
    expanded = graph.expand(["p1"], k=3)
    assert expanded[0] == "p4" and "p2" in expanded


def test_bm25_strategy_and_rerank_without_model():
    chunks = [
        doc("x", "battery pack costs fell 90 percent"),
        doc("y", "OPEC supply response"),
        doc("z", "fleet turnover and vehicle lifetimes"),
    ]
    r = Retriever(chunks, Settings())
    hits = r._bm25("battery costs", 2)
    assert hits[0]["id"] == "x" and hits[0]["retriever"] == "BM25"


def test_reducers_dedup_documents_and_sum_metrics():
    assert [d["id"] for d in merge_docs([doc("a")], [doc("a"), doc("b"), doc("b")])] == ["a", "b"]
    assert add_metrics({"x": 1, "y": 2}, {"x": 3}) == {"x": 4, "y": 2}
