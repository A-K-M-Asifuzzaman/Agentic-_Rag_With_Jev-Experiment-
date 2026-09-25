from typing import Any

from app.agent.policies.jev_policy import HybridAgentPolicy, JevAgentPolicy
from app.config import Settings
from app.services import JevService, Meter
from tests.fakes import FakeJevClient, ScriptedPolicy, doc


def choice(label: str, confidence: float) -> dict[str, Any]:
    return {"type": "choice", "choice": label, "confidence": confidence, "probabilities": {label: confidence}}


def jev(answer: Any) -> tuple[JevService, FakeJevClient]:
    client = FakeJevClient(answer)
    return JevService(client, Settings(jev_price_in=1.0, jev_price_out=1.0)), client


async def test_route_query_maps_choices_and_takes_min_confidence():
    service, client = jev(
        lambda s, name: choice("DECOMPOSE", 0.9) if name == "action" else choice("BM25", 0.6)
    )
    meter = Meter()
    d = await JevAgentPolicy(service, "corpus").route_query("q", meter)
    assert (d["action"], d["strategy"], d["confidence"], d["decided_by"]) == ("DECOMPOSE", "BM25", 0.6, "jev")
    assert set(client.requests[0]["questions"]["action"].criteria) == {
        "RETRIEVE",
        "DECOMPOSE",
        "DIRECT_ANSWER",
    }
    assert meter["jev_calls"] == 1 and meter["jev_cost_usd"] == 11 / 1e6


async def test_verify_maps_noul_probability_to_action_and_confidence():
    service, _ = jev(lambda s, name: {"type": "noul", "noul": 0.2})
    d = await JevAgentPolicy(service, "corpus").verify_answer("q", "a", [doc("x")], Meter())
    assert d["action"] == "RETRY" and abs(d["confidence"] - 0.6) < 1e-9


async def test_hybrid_regrades_only_uncertain_passages_with_llm():
    unsure, sure = doc("unsure"), doc("sure")
    service, _ = jev(lambda s, name: {"type": "noul", "noul": 0.95 if s["passage"] == sure["text"] else 0.55})
    policy = HybridAgentPolicy(service, "corpus", ScriptedPolicy(relevance=0.0, decided_by="llm"), 0.7, 0.15)
    meter = Meter()
    scores = await policy.grade_evidence("q", [unsure, sure], meter)
    assert scores == {"unsure": 0.0, "sure": 0.95}
    assert meter["escalations.grade"] == 1 and meter["jev_calls"] == 2
