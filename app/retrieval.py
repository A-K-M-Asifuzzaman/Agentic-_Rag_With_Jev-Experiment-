"""Corpus ingestion and the retrieval backends shared by every pipeline.

Strategies: VECTOR (Chroma), BM25, HYBRID (reciprocal rank fusion of both), GRAPH (hybrid seeds expanded
over an entity/adjacency graph) and WEB (Tavily). All candidates go through the same cross-encoder reranker.
"""

import asyncio
import json
import math
import os
import re
from collections import Counter, defaultdict
from itertools import pairwise
from typing import Literal, Protocol, TypedDict

from app.config import Settings

Strategy = Literal["VECTOR", "BM25", "HYBRID", "GRAPH", "WEB"]


class Doc(TypedDict):
    id: str
    text: str
    source: str
    page: int | None
    retriever: str
    score: float


class RetrieverLike(Protocol):
    async def search(self, query: str, strategy: str, k: int) -> list[Doc]: ...
    async def rerank(self, query: str, docs: list[Doc], k: int) -> list[Doc]: ...


# --------------------------------------------------------------------------- ingestion


def load_chunks(settings: Settings) -> list[Doc]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from pypdf import PdfReader

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    chunks: list[Doc] = []
    for pdf in sorted(settings.documents_dir.glob("*.pdf")):
        for page_no, page in enumerate(PdfReader(pdf).pages, start=1):
            for i, text in enumerate(splitter.split_text(page.extract_text() or "")):
                chunks.append(
                    Doc(
                        id=f"{pdf.stem}:p{page_no}:c{i}",
                        text=text,
                        source=pdf.name,
                        page=page_no,
                        retriever="",
                        score=0.0,
                    )
                )
    return chunks


def ingest(settings: Settings) -> int:
    """Chunk every PDF in documents_dir, persist chunks for BM25/graph, and rebuild the Chroma collection."""
    chunks = load_chunks(settings)
    if not chunks:
        raise SystemExit(f"No PDFs found in {settings.documents_dir}")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.chunks_path.write_text("\n".join(json.dumps(c) for c in chunks))
    store = _vectorstore(settings)
    store.reset_collection()
    store.add_texts(
        [c["text"] for c in chunks],
        ids=[c["id"] for c in chunks],
        metadatas=[{"source": c["source"], "page": c["page"] or 0} for c in chunks],
    )
    return len(chunks)


def _vectorstore(settings: Settings):  # type: ignore[no-untyped-def]
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    return Chroma(
        collection_name="corpus",
        embedding_function=OpenAIEmbeddings(model=settings.embedding_model),
        persist_directory=str(settings.chroma_dir),
    )


# --------------------------------------------------------------------------- fusion and graph

_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
# ponytail: regex entities (capitalised spans, acronyms, quantities). Swap for NER or LLM-extracted triples
# if graph retrieval becomes a research variable of its own.
_ENTITY = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9+\-]*(?:\s+[A-Z][A-Za-z0-9+\-]*)*|\$?\d[\d.,]*\s?(?:percent|%|million|billion))"
)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def rrf(rankings: list[list[Doc]], k: int, c: int = 60) -> list[Doc]:
    """Reciprocal rank fusion: score = sum over rankings of 1 / (c + rank)."""
    scores: dict[str, float] = defaultdict(float)
    by_id: dict[str, Doc] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc["id"]] += 1 / (c + rank)
            by_id.setdefault(doc["id"], doc)
    top = sorted(scores, key=scores.__getitem__, reverse=True)[:k]
    return [Doc(**{**by_id[i], "score": scores[i]}) for i in top]  # type: ignore[typeddict-item]


class EntityGraph:
    """Chunks linked to their neighbours on the page and to chunks sharing a discriminative entity."""

    def __init__(self, chunks: list[Doc], max_df: float = 0.2):
        # A single capitalised word that also occurs in lowercase ("Under", "This") is a sentence start,
        # not an entity; acronyms and proper nouns never appear in lowercase.
        common = {w for c in chunks for w in re.findall(r"\b[a-z]+\b", c["text"])}
        ents = [
            {
                e.strip().lower()
                for e in _ENTITY.findall(c["text"])
                if " " in e.strip() or e.lower() not in common
            }
            for c in chunks
        ]
        df = Counter(e for s in ents for e in s)
        n = len(chunks)
        self.idf = {e: math.log(n / d) for e, d in df.items() if 1 < d <= max(2, max_df * n)}
        self.entities = {c["id"]: {e for e in s if e in self.idf} for c, s in zip(chunks, ents, strict=True)}
        self.adjacent: dict[str, set[str]] = defaultdict(set)
        for a, b in pairwise(chunks):
            if a["source"] == b["source"]:
                self.adjacent[a["id"]].add(b["id"])
                self.adjacent[b["id"]].add(a["id"])

    def expand(self, seed_ids: list[str], k: int) -> list[str]:
        seed_ents = set().union(*(self.entities.get(s, set()) for s in seed_ids))
        scores: dict[str, float] = defaultdict(float)
        for cid, ents in self.entities.items():
            if cid not in seed_ids:
                scores[cid] += sum(self.idf[e] for e in ents & seed_ents)
        for s in seed_ids:
            for nb in self.adjacent[s] - set(seed_ids):
                scores[nb] += 1.0
        return [cid for cid in sorted(scores, key=scores.__getitem__, reverse=True) if scores[cid] > 0][:k]


# --------------------------------------------------------------------------- retriever


class Retriever:
    def __init__(self, chunks: list[Doc], settings: Settings, vectorstore=None, ranker=None):  # type: ignore[no-untyped-def]
        from rank_bm25 import BM25Okapi

        self.settings = settings
        self.chunks = {c["id"]: c for c in chunks}
        self._order = list(self.chunks)
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        self.graph = EntityGraph(chunks)
        self.vectorstore = vectorstore
        self.ranker = ranker

    @classmethod
    def load(cls, settings: Settings) -> "Retriever":
        from flashrank import Ranker

        if not settings.chunks_path.exists():
            raise SystemExit("Index not found. Run `ajev ingest` first.")
        chunks = [json.loads(line) for line in settings.chunks_path.read_text().splitlines() if line]
        ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir=str(settings.data_dir / "models"))
        return cls(chunks, settings, _vectorstore(settings), ranker)

    def _doc(self, cid: str, retriever: str, score: float) -> Doc:
        return Doc(**{**self.chunks[cid], "retriever": retriever, "score": score})  # type: ignore[typeddict-item]

    async def _vector(self, query: str, k: int) -> list[Doc]:
        hits = await self.vectorstore.asimilarity_search_with_relevance_scores(query, k=k)
        return [self._doc(d.id, "VECTOR", s) for d, s in hits if d.id in self.chunks]

    def _bm25(self, query: str, k: int) -> list[Doc]:
        scores = self.bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._doc(self._order[i], "BM25", float(scores[i])) for i in top if scores[i] > 0]

    async def _hybrid(self, query: str, k: int) -> list[Doc]:
        fused = rrf([await self._vector(query, k), self._bm25(query, k)], k)
        return [Doc(**{**d, "retriever": "HYBRID"}) for d in fused]  # type: ignore[typeddict-item]

    async def _graph(self, query: str, k: int) -> list[Doc]:
        seeds = (await self._hybrid(query, k))[: max(2, k // 3)]
        expanded = self.graph.expand([d["id"] for d in seeds], k - len(seeds))
        return [Doc(**{**d, "retriever": "GRAPH"}) for d in seeds] + [  # type: ignore[typeddict-item]
            self._doc(cid, "GRAPH", 0.0) for cid in expanded
        ]

    async def _web(self, query: str, k: int) -> list[Doc]:
        if not os.getenv("TAVILY_API_KEY"):
            return []
        from tavily import AsyncTavilyClient

        resp = await AsyncTavilyClient().search(query, max_results=min(k, 5))
        return [
            Doc(
                id=f"web:{r['url']}",
                text=r["content"],
                source=r["url"],
                page=None,
                retriever="WEB",
                score=float(r.get("score", 0.0)),
            )
            for r in resp.get("results", [])
        ]

    async def search(self, query: str, strategy: str, k: int) -> list[Doc]:
        match strategy:
            case "VECTOR":
                return await self._vector(query, k)
            case "BM25":
                return self._bm25(query, k)
            case "GRAPH":
                return await self._graph(query, k)
            case "WEB":
                return await self._web(query, k)
            case _:
                return await self._hybrid(query, k)

    async def rerank(self, query: str, docs: list[Doc], k: int) -> list[Doc]:
        if not docs or self.ranker is None:
            return docs[:k]
        from flashrank import RerankRequest

        passages = [{"id": d["id"], "text": d["text"]} for d in docs]
        ranked = await asyncio.to_thread(self.ranker.rerank, RerankRequest(query=query, passages=passages))
        by_id = {d["id"]: d for d in docs}
        return [Doc(**{**by_id[r["id"]], "score": float(r["score"])}) for r in ranked[:k]]  # type: ignore[typeddict-item]
