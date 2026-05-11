from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from research_intel.models import utc_now_iso
from research_intel.rag.embedding import EmbeddingModel, cosine_similarity, create_embedding_model, tokenize


SCHEMA_VERSION = 1


@dataclass(slots=True)
class RagChunk:
    chunk_id: str
    title: str
    kind: str
    text: str
    item_id: str = ""
    url: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "title": self.title,
            "kind": self.kind,
            "text": self.text,
            "item_id": self.item_id,
            "url": self.url,
            "source": self.source,
            "metadata": self.metadata,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RagChunk":
        return cls(
            chunk_id=str(payload.get("chunk_id", "")),
            title=str(payload.get("title", "")),
            kind=str(payload.get("kind", "")),
            text=str(payload.get("text", "")),
            item_id=str(payload.get("item_id", "")),
            url=str(payload.get("url", "")),
            source=str(payload.get("source", "")),
            metadata=dict(payload.get("metadata", {})),
            embedding=[float(value) for value in payload.get("embedding", [])],
        )


@dataclass(slots=True)
class RagSearchResult:
    chunk: RagChunk
    score: float
    dense_score: float = 0.0
    keyword_score: float = 0.0
    boost_score: float = 0.0

    def source_payload(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "item_id": self.chunk.item_id,
            "title": self.chunk.title,
            "kind": self.chunk.kind,
            "url": self.chunk.url,
            "source": self.chunk.source,
            "score": round(self.score, 4),
            "dense_score": round(self.dense_score, 4),
            "keyword_score": round(self.keyword_score, 4),
            "boost_score": round(self.boost_score, 4),
            "preview": self.chunk.text[:220],
        }


class RagIndex:
    def __init__(
        self,
        chunks: list[RagChunk] | None = None,
        dimensions: int | None = None,
        built_at: str | None = None,
        embedding_model: EmbeddingModel | None = None,
        embedding_provider: str | None = None,
        embedding_model_name: str | None = None,
    ) -> None:
        self.embedding_model = embedding_model or create_embedding_model(dimensions=dimensions)
        self.dimensions = self.embedding_model.dimensions
        self.embedding_provider = embedding_provider or self.embedding_model.provider
        self.embedding_model_name = embedding_model_name or self.embedding_model.model_name
        self.built_at = built_at or utc_now_iso()
        self.chunks = chunks or []

    @classmethod
    def from_report(
        cls,
        report: dict[str, Any],
        candidates: list[dict[str, Any]] | None = None,
        dimensions: int | None = None,
    ) -> "RagIndex":
        index = cls(dimensions=dimensions)
        chunks = _chunks_from_report(report, candidates or [])
        index.add_chunks(chunks)
        return index

    @classmethod
    def load(cls, path: Path) -> "RagIndex":
        payload = json.loads(path.read_text(encoding="utf-8"))
        dimensions = int(payload.get("dimensions", 384))
        chunks = [RagChunk.from_dict(item) for item in payload.get("chunks", [])]
        return cls(
            chunks=chunks,
            dimensions=dimensions,
            built_at=str(payload.get("built_at", "")) or None,
            embedding_provider=str(payload.get("embedding_provider", "")) or None,
            embedding_model_name=str(payload.get("embedding_model", "")) or None,
        )

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "built_at": self.built_at,
            "dimensions": self.dimensions,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model_name,
            "chunk_count": len(self.chunks),
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }

    def is_compatible_with_current_embedding(self) -> bool:
        model = create_embedding_model()
        return (
            self.embedding_provider == model.provider
            and self.embedding_model_name == model.model_name
            and self.dimensions == model.dimensions
        )

    def add_chunks(self, chunks: list[RagChunk]) -> None:
        seen: set[str] = set()
        output: list[RagChunk] = []
        for chunk in [*self.chunks, *chunks]:
            if not chunk.chunk_id or chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            if not chunk.embedding:
                chunk.embedding = self.embedding_model.embed_document(_search_text(chunk))
            output.append(chunk)
        self.chunks = output

    def search(
        self,
        query: str,
        selected_item_id: str | None = None,
        limit: int | None = None,
    ) -> list[RagSearchResult]:
        top_k = limit if limit is not None else _default_top_k()
        query_vector = self.embedding_model.embed_query(query)
        query_tokens = set(tokenize(query))
        keyword_index = _KeywordIndex(self.chunks)
        scored: list[RagSearchResult] = []
        kind_hints = _kind_hints(query)
        dense_weight = _float_env("RAG_DENSE_WEIGHT", 0.68)
        keyword_weight = _float_env("RAG_KEYWORD_WEIGHT", 0.32)

        for chunk in self.chunks:
            dense_score = cosine_similarity(query_vector, chunk.embedding)
            keyword_score = keyword_index.score(query_tokens, chunk.chunk_id)
            selected_boost = 0.35 if selected_item_id and chunk.item_id == selected_item_id else 0.0
            kind_boost = 0.08 if kind_hints and any(hint in chunk.kind for hint in kind_hints) else 0.0
            boost_score = selected_boost + kind_boost
            score = dense_weight * dense_score + keyword_weight * keyword_score + boost_score
            if score > 0.025 or selected_boost:
                scored.append(
                    RagSearchResult(
                        chunk=chunk,
                        score=score,
                        dense_score=dense_score,
                        keyword_score=keyword_score,
                        boost_score=boost_score,
                    )
                )

        scored.sort(key=lambda result: result.score, reverse=True)
        return scored[:top_k]


def _default_top_k() -> int:
    raw = os.getenv("RAG_TOP_K", "8")
    try:
        return max(1, int(raw))
    except ValueError:
        return 8


class _KeywordIndex:
    def __init__(self, chunks: list[RagChunk]) -> None:
        self.total_docs = max(1, len(chunks))
        self.avg_len = 1.0
        self.term_freqs: dict[str, Counter[str]] = {}
        self.doc_lens: dict[str, int] = {}
        doc_freqs: Counter[str] = Counter()

        lengths: list[int] = []
        for chunk in chunks:
            tokens = tokenize(_search_text(chunk))
            counts = Counter(tokens)
            self.term_freqs[chunk.chunk_id] = counts
            self.doc_lens[chunk.chunk_id] = len(tokens)
            lengths.append(len(tokens))
            doc_freqs.update(counts.keys())

        if lengths:
            self.avg_len = sum(lengths) / len(lengths)
        self.doc_freqs = doc_freqs

    def score(self, query_tokens: set[str], chunk_id: str) -> float:
        if not query_tokens:
            return 0.0
        counts = self.term_freqs.get(chunk_id, Counter())
        doc_len = self.doc_lens.get(chunk_id, 0)
        if not counts or doc_len == 0:
            return 0.0

        k1 = 1.5
        b = 0.75
        raw_score = 0.0
        for token in query_tokens:
            tf = counts.get(token, 0)
            if tf == 0:
                continue
            df = self.doc_freqs.get(token, 0)
            idf = math.log(1 + (self.total_docs - df + 0.5) / (df + 0.5))
            denom = tf + k1 * (1 - b + b * doc_len / self.avg_len)
            raw_score += idf * (tf * (k1 + 1)) / denom

        return raw_score / (raw_score + 6.0) if raw_score > 0 else 0.0


def _float_env(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def _chunks_from_report(report: dict[str, Any], candidates: list[dict[str, Any]]) -> list[RagChunk]:
    chunks: list[RagChunk] = []
    known_candidate_ids: set[str] = set()

    for section, kind in (
        ("top_papers", "paper_analysis"),
        ("top_repos", "repo_analysis"),
        ("top_tools", "tool_analysis"),
    ):
        for item in _list(report.get(section)):
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("item_id", ""))
            chunks.append(
                RagChunk(
                    chunk_id=f"analysis:{section}:{item_id or _slug(item.get('title', ''))}",
                    item_id=item_id,
                    title=str(item.get("title", "")),
                    kind=kind,
                    url=str(item.get("url", "")),
                    source="daily_report",
                    text=_analysis_text(item),
                    metadata={
                        "score": item.get("score", 0),
                        "confidence": item.get("confidence", ""),
                    },
                )
            )

    actions = _list(report.get("actions"))
    if actions:
        chunks.append(
            RagChunk(
                chunk_id="report:recommended_actions",
                title="Recommended actions",
                kind="actions",
                text="\n".join(str(action) for action in actions),
                source="daily_report",
            )
        )

    for index, trend in enumerate(_list(report.get("trends"))):
        if not isinstance(trend, dict):
            continue
        chunks.append(
            RagChunk(
                chunk_id=f"trend:{index}:{_slug(trend.get('topic', 'trend'))}",
                title=str(trend.get("topic", "trend")),
                kind="trend",
                text=_trend_text(trend),
                source="daily_report",
                metadata={
                    "window_days": trend.get("window_days"),
                    "confidence": trend.get("confidence", ""),
                },
            )
        )

    for decision in _list(report.get("filter_decisions")):
        if not isinstance(decision, dict):
            continue
        item_id = str(decision.get("item_id", ""))
        chunks.append(
            RagChunk(
                chunk_id=f"filter:{item_id}",
                item_id=item_id,
                title=f"Filter decision for {item_id}",
                kind="filter_decision",
                text=_filter_text(decision),
                source="filtering_agent",
                metadata={"status": decision.get("status", "")},
            )
        )

    for item in [*_list(report.get("candidates")), *candidates]:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("item_id", ""))
        if not item_id or item_id in known_candidate_ids:
            continue
        known_candidate_ids.add(item_id)
        chunks.append(
            RagChunk(
                chunk_id=f"candidate:{item_id}",
                item_id=item_id,
                title=str(item.get("title", item_id)),
                kind=f"candidate_{item.get('content_type', 'item')}",
                url=str(item.get("url", "")),
                source=str(item.get("source", "candidate")),
                text=_candidate_text(item),
                metadata={
                    "content_type": item.get("content_type", ""),
                    "published_at": item.get("published_at"),
                },
            )
        )

    return [chunk for chunk in chunks if chunk.title or chunk.text]


def _analysis_text(item: dict[str, Any]) -> str:
    parts: list[str] = [
        str(item.get("content_type", "")),
        f"Why it matters: {item.get('why_it_matters', '')}",
        f"Relation to user: {item.get('relation_to_user', '')}",
        f"Technical core: {item.get('technical_core', '')}",
        f"Strengths: {'; '.join(str(value) for value in _list(item.get('strengths')))}",
        f"Limitations: {'; '.join(str(value) for value in _list(item.get('limitations')))}",
        f"Possible actions: {'; '.join(str(value) for value in _list(item.get('possible_actions')))}",
        f"Evidence: {'; '.join(str(value) for value in _list(item.get('evidence')))}",
    ]
    return " ".join(part for part in parts if part.strip(": "))


def _trend_text(trend: dict[str, Any]) -> str:
    return " ".join(
        part
        for part in (
            str(trend.get("summary", "")),
            str(trend.get("user_implication", "")),
            "Signals: " + "; ".join(str(signal) for signal in _list(trend.get("signals"))),
        )
        if part
    )


def _filter_text(decision: dict[str, Any]) -> str:
    return " ".join(
        part
        for part in (
            f"Status: {decision.get('status', '')}",
            f"Relevance score: {decision.get('relevance_score', '')}",
            f"Quality score: {decision.get('quality_score', '')}",
            "Reasons: " + "; ".join(str(reason) for reason in _list(decision.get("reasons"))),
            "Signals: " + json.dumps(decision.get("signals", {}), ensure_ascii=False),
        )
        if part
    )


def _candidate_text(item: dict[str, Any]) -> str:
    return " ".join(
        part
        for part in (
            str(item.get("summary", "")),
            "Tags: " + ", ".join(str(tag) for tag in _list(item.get("tags"))),
            "Authors: " + ", ".join(str(author) for author in _list(item.get("authors"))),
            "Metrics: " + json.dumps(item.get("metrics", {}), ensure_ascii=False),
            "Technical signals: " + json.dumps(item.get("technical_signals", {}), ensure_ascii=False),
            "Links: " + json.dumps(item.get("links", {}), ensure_ascii=False),
        )
        if part and part != "{}"
    )


def _search_text(chunk: RagChunk) -> str:
    return f"{chunk.title}\n{chunk.kind}\n{chunk.source}\n{chunk.text}"


def _kind_hints(query: str) -> set[str]:
    lowered = query.lower()
    hints: set[str] = set()
    if any(term in lowered for term in ("repo", "github", "项目", "baseline", "基线", "代码")):
        hints.add("repo")
    if any(term in lowered for term in ("paper", "论文", "实验", "方法", "evaluation")):
        hints.add("paper")
    if any(term in lowered for term in ("trend", "趋势", "机会", "选题", "方向")):
        hints.add("trend")
    return hints


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _slug(value: Any) -> str:
    text = str(value).lower().strip()
    output = []
    for char in text:
        if char.isalnum():
            output.append(char)
        elif output and output[-1] != "-":
            output.append("-")
    return "".join(output).strip("-")[:80] or "item"
