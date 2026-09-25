"""Runtime settings. Every field can be overridden with an `AJEV_<FIELD>` environment variable."""

import os
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

CORPUS_DESCRIPTION = (
    "A technical report titled 'Will EVs Dampen the Oil Price Shock?' (April 2026) covering EV adoption "
    "trajectories, oil demand displacement scenarios, fleet turnover, battery cost trends, OPEC+ supply "
    "behaviour, stranded assets, critical-mineral supply chains and oil price volatility through 2050."
)


@dataclass(frozen=True)
class Settings:
    # Models. Generation, rewriting, decomposition and LLM control share one model by default so the
    # controller policy is the only variable between pipelines.
    llm_model: str = "openai:gpt-5-mini"
    controller_model: str = "openai:gpt-5-mini"
    embedding_model: str = "text-embedding-3-small"
    jev_model: str = "jev-latest"

    # Corpus and indexes
    documents_dir: Path = Path("data/documents")
    data_dir: Path = Path(".data")
    corpus_description: str = CORPUS_DESCRIPTION
    chunk_size: int = 900
    chunk_overlap: int = 150

    # Retrieval
    candidate_k: int = 10  # candidates per retriever before reranking
    top_k: int = 5  # kept per query after reranking
    max_evidence: int = 8  # passages handed to the generator
    relevance_threshold: float = 0.5

    # Hybrid policy gates
    escalation_threshold: float = 0.7  # Jev confidence below this escalates to the LLM controller
    grade_uncertainty_band: float = 0.15  # |p(relevant) - 0.5| below this is re-graded by the LLM

    # Loop safety. Routers read these; nothing loops unbounded.
    max_steps: int = 10
    max_retrieval_attempts: int = 4
    max_rewrite_attempts: int = 2
    max_decomposition_attempts: int = 1
    max_web_search_attempts: int = 1
    max_graph_search_attempts: int = 1
    max_verification_attempts: int = 1

    # USD per 1M tokens as (input, output). Jev defaults to 0 because pricing depends on your
    # TypeSafe plan; set AJEV_JEV_PRICE_IN/OUT so cost comparisons include it.
    llm_prices: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {"gpt-5-mini": (0.25, 2.00), "gpt-5": (1.25, 10.00)}
    )
    jev_price_in: float = 0.0
    jev_price_out: float = 0.0

    @property
    def chunks_path(self) -> Path:
        return self.data_dir / "chunks.jsonl"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @classmethod
    def from_env(cls) -> "Settings":
        base = cls()
        overrides: dict[str, object] = {}
        for f in fields(cls):
            raw = os.getenv(f"AJEV_{f.name.upper()}")
            if raw is None or f.name == "llm_prices":
                continue
            current = getattr(base, f.name)
            overrides[f.name] = Path(raw) if isinstance(current, Path) else type(current)(raw)
        return replace(base, **overrides)  # type: ignore[arg-type]
