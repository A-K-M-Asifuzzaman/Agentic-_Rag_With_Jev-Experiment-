"""Metered clients for the two model families: a generative LLM and Jev (TypeSafe System One).

Every call records calls, tokens, latency and cost into a per-node `Meter`, which the node returns
as a `metrics` update. That keeps research telemetry inside graph state, where reducers sum it.
"""

from collections import Counter
from time import perf_counter
from typing import Any, TypeVar, cast

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.config import get_stream_writer
from pydantic import BaseModel

from app.config import Settings

Meter = Counter[str]  # metric name -> summed value
T = TypeVar("T", bound=BaseModel)


def emit(event: str, **data: Any) -> None:
    """Send a custom stream event when running inside a graph; no-op elsewhere (tests, scripts)."""
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer({"event": event, **data})


class LLMService:
    def __init__(self, model: BaseChatModel, model_name: str, prices: tuple[float, float] | None = None):
        self.model = model
        self.model_name = model_name
        self.prices = prices

    @classmethod
    def from_settings(cls, model_id: str, settings: Settings) -> "LLMService":
        name = model_id.split(":", 1)[-1]
        return cls(init_chat_model(model_id), name, settings.llm_prices.get(name))

    async def structured(self, schema: type[T], system: str, user: str, meter: Meter, purpose: str) -> T:
        start = perf_counter()
        runnable = self.model.with_structured_output(schema, include_raw=True)
        out = cast(dict[str, Any], await runnable.ainvoke([SystemMessage(system), HumanMessage(user)]))
        self._record(out["raw"], meter, purpose, start)
        if out["parsed"] is None:
            raise ValueError(
                f"{purpose}: model output did not match {schema.__name__}: {out['parsing_error']}"
            )
        return out["parsed"]

    async def text(self, system: str, user: str, meter: Meter, purpose: str) -> str:
        start = perf_counter()
        msg = await self.model.ainvoke([SystemMessage(system), HumanMessage(user)])
        self._record(msg, meter, purpose, start)
        return msg.text.strip()

    def _record(self, msg: AIMessage, meter: Meter, purpose: str, start: float) -> None:
        usage: dict[str, Any] = dict(msg.usage_metadata or {})
        tin, tout = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        ms = (perf_counter() - start) * 1000
        cost = (tin * self.prices[0] + tout * self.prices[1]) / 1e6 if self.prices else 0.0
        meter.update(
            {
                "llm_calls": 1,
                f"llm_calls.{purpose}": 1,
                "llm_input_tokens": tin,
                "llm_output_tokens": tout,
                "llm_cost_usd": cost,
                "llm_latency_ms": ms,
            }
        )
        emit(
            "llm.call",
            purpose=purpose,
            model=self.model_name,
            input_tokens=tin,
            output_tokens=tout,
            latency_ms=round(ms, 1),
        )


class JevService:
    """Thin wrapper over `typesafe_sdk.AsyncTypeSafeClient.system_one`."""

    def __init__(self, client: Any, settings: Settings):
        self.client = client
        self.price_in, self.price_out = settings.jev_price_in, settings.jev_price_out

    @classmethod
    def from_settings(cls, settings: Settings) -> "JevService":
        from typesafe_sdk import AsyncTypeSafeClient

        return cls(AsyncTypeSafeClient(model=settings.jev_model), settings)

    async def ask(self, state: Any, questions: dict[str, Any], meter: Meter, purpose: str) -> Any:
        start = perf_counter()
        resp = await self.client.system_one(state=state, questions=questions)
        ms = (perf_counter() - start) * 1000
        tin, tout = resp.usage.input_tokens or 0, resp.usage.output_tokens or 0
        meter.update(
            {
                "jev_calls": 1,
                f"jev_calls.{purpose}": 1,
                "jev_input_tokens": tin,
                "jev_output_tokens": tout,
                "jev_cost_usd": (tin * self.price_in + tout * self.price_out) / 1e6,
                "jev_latency_ms": ms,
            }
        )
        emit(
            "jev.call", purpose=purpose, model=resp.model, questions=list(questions), latency_ms=round(ms, 1)
        )
        return resp
