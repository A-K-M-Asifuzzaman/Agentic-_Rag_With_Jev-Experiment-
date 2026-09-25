"""Offline stand-ins for the LLM, Jev, retriever and policies."""

from typing import Any, cast

from typesafe_sdk import SystemOneResponse

from app.agent.state import Decision
from app.retrieval import Doc


def doc(i: str, text: str = "", score: float = 1.0) -> Doc:
    return Doc(id=i, text=text or f"text {i}", source="report.pdf", page=1, retriever="HYBRID", score=score)


class FakeRetriever:
    """Returns one document per distinct query so fan-out and dedup are observable."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def search(self, query: str, strategy: str, k: int) -> list[Doc]:
        self.calls.append((query, strategy))
        return [doc(f"{strategy}:{query}", f"passage about {query}")]

    async def rerank(self, query: str, docs: list[Doc], k: int) -> list[Doc]:
        return docs[:k]


class FakeLLM:
    def __init__(self, sub_queries: list[str] | None = None) -> None:
        self.sub_queries = sub_queries or []
        self.calls: list[str] = []

    async def structured(self, schema: Any, system: str, user: str, meter: Any, purpose: str) -> Any:
        self.calls.append(purpose)
        meter.update({"llm_calls": 1})
        return schema(sub_queries=self.sub_queries)

    async def text(self, system: str, user: str, meter: Any, purpose: str) -> str:
        self.calls.append(purpose)
        meter.update({"llm_calls": 1})
        return f"{purpose} output"


class ScriptedPolicy:
    """AgentPolicy with fixed answers; `next_actions` is consumed in order and the last one repeats."""

    def __init__(
        self,
        *,
        route: str = "RETRIEVE",
        next_actions: tuple[str, ...] = ("ANSWER",),
        verify: str = "ACCEPT",
        relevance: float = 1.0,
        confidence: float = 1.0,
        decided_by: str = "jev",
        escalation_threshold: float | None = None,
        controller: Any = None,
    ) -> None:
        self.name = "scripted"
        self.route, self.next_actions, self.verify = route, list(next_actions), verify
        self.relevance, self.confidence, self.decided_by = relevance, confidence, decided_by
        self.escalation_threshold, self.controller = escalation_threshold, controller

    def _d(self, stage: str, action: str, **extra: Any) -> Decision:
        return cast(
            Decision,
            {
                "stage": stage,
                "action": action,
                "confidence": self.confidence,
                "decided_by": self.decided_by,
                **extra,
            },
        )

    async def route_query(self, query: str, meter: Any) -> Decision:
        return self._d("analyze", self.route, strategy="HYBRID")

    async def grade_evidence(self, query: str, docs: list[Doc], meter: Any) -> dict[str, float]:
        return {d["id"]: self.relevance for d in docs}

    async def choose_next_action(self, view: Any, meter: Any) -> Decision:
        action = self.next_actions.pop(0) if len(self.next_actions) > 1 else self.next_actions[0]
        return self._d("evidence", action, evidence_status="SUFFICIENT")

    async def verify_answer(self, query: str, answer: str, evidence: list[Doc], meter: Any) -> Decision:
        return self._d("verify", self.verify)


class FakeJevClient:
    """Mimics AsyncTypeSafeClient.system_one; `answer(state, name)` returns the raw answer dict."""

    def __init__(self, answer: Any) -> None:
        self.answer = answer
        self.requests: list[dict[str, Any]] = []

    async def system_one(self, state: Any, questions: dict[str, Any]) -> SystemOneResponse:
        self.requests.append({"state": state, "questions": questions})
        return SystemOneResponse.model_validate(
            {
                "model": "jev-test",
                "usage": {"input_tokens": 10, "output_tokens": 1},
                "answers": {name: self.answer(state, name) for name in questions},
            }
        )
