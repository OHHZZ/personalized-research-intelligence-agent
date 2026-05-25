from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from research_intel.rag.index import RagChunk, RagIndex, RagSearchResult


@dataclass(slots=True)
class PgVectorConfig:
    dsn: str
    table: str = "research_rag_chunks"


class PgVectorStore:
    """Optional PostgreSQL + pgvector storage for RAG chunks.

    The project keeps JSON as the default local store. When `PGVECTOR_DSN` is
    configured and `psycopg[binary]` is installed, this store can persist the
    same chunks into a real vector database and retrieve dense candidates.
    """

    def __init__(self, config: PgVectorConfig) -> None:
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError(
                "psycopg is not installed. Install optional dependencies with "
                "`pip install -e .[pgvector]`, or unset PGVECTOR_DSN."
            ) from exc
        self._psycopg = psycopg
        self.config = config

    @classmethod
    def from_env(cls) -> "PgVectorStore | None":
        dsn = os.getenv("PGVECTOR_DSN") or os.getenv("DATABASE_URL")
        if not dsn:
            return None
        table = os.getenv("PGVECTOR_TABLE", "research_rag_chunks")
        return cls(PgVectorConfig(dsn=dsn, table=table))

    def initialize(self, dimensions: int) -> None:
        table = self._table_sql()
        with self._psycopg.connect(self.config.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                      chunk_id TEXT PRIMARY KEY,
                      item_id TEXT,
                      title TEXT NOT NULL,
                      kind TEXT NOT NULL,
                      body TEXT NOT NULL,
                      url TEXT,
                      source TEXT,
                      metadata JSONB,
                      embedding vector({dimensions}),
                      embedding_provider TEXT,
                      embedding_model TEXT,
                      built_at TIMESTAMPTZ DEFAULT now()
                    )
                    """
                )
                cur.execute(f"CREATE INDEX IF NOT EXISTS {table}_kind_idx ON {table} (kind)")
                cur.execute(f"CREATE INDEX IF NOT EXISTS {table}_item_idx ON {table} (item_id)")
                cur.execute(
                    f"CREATE INDEX IF NOT EXISTS {table}_embedding_idx ON {table} "
                    "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
                )
            conn.commit()

    def upsert_index(self, index: RagIndex) -> int:
        self.initialize(index.dimensions)
        table = self._table_sql()
        rows = [
            (
                chunk.chunk_id,
                chunk.item_id,
                chunk.title,
                chunk.kind,
                chunk.text,
                chunk.url,
                chunk.source,
                json.dumps(chunk.metadata, ensure_ascii=False),
                _vector_literal(chunk.embedding),
                index.embedding_provider,
                index.embedding_model_name,
            )
            for chunk in index.chunks
        ]
        if not rows:
            return 0

        with self._psycopg.connect(self.config.dsn) as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    f"""
                    INSERT INTO {table}
                      (chunk_id, item_id, title, kind, body, url, source, metadata, embedding,
                       embedding_provider, embedding_model)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                      item_id = EXCLUDED.item_id,
                      title = EXCLUDED.title,
                      kind = EXCLUDED.kind,
                      body = EXCLUDED.body,
                      url = EXCLUDED.url,
                      source = EXCLUDED.source,
                      metadata = EXCLUDED.metadata,
                      embedding = EXCLUDED.embedding,
                      embedding_provider = EXCLUDED.embedding_provider,
                      embedding_model = EXCLUDED.embedding_model,
                      built_at = now()
                    """,
                    rows,
                )
            conn.commit()
        return len(rows)

    def search_dense(
        self,
        query_embedding: list[float],
        embedding_provider: str,
        embedding_model: str,
        limit: int = 50,
    ) -> list[RagSearchResult]:
        table = self._table_sql()
        with self._psycopg.connect(self.config.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT chunk_id, item_id, title, kind, body, url, source, metadata,
                           1 - (embedding <=> %s::vector) AS dense_score
                    FROM {table}
                    WHERE embedding_provider = %s AND embedding_model = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (
                        _vector_literal(query_embedding),
                        embedding_provider,
                        embedding_model,
                        _vector_literal(query_embedding),
                        limit,
                    ),
                )
                rows = cur.fetchall()

        results: list[RagSearchResult] = []
        for row in rows:
            metadata: dict[str, Any] = row[7] if isinstance(row[7], dict) else json.loads(row[7] or "{}")
            chunk = RagChunk(
                chunk_id=str(row[0]),
                item_id=str(row[1] or ""),
                title=str(row[2] or ""),
                kind=str(row[3] or ""),
                text=str(row[4] or ""),
                url=str(row[5] or ""),
                source=str(row[6] or ""),
                metadata=metadata,
                embedding=[],
            )
            dense_score = float(row[8] or 0.0)
            results.append(RagSearchResult(chunk=chunk, score=dense_score, dense_score=dense_score))
        return results

    def _table_sql(self) -> str:
        table = self.config.table
        if not table.replace("_", "").isalnum():
            raise ValueError(f"Invalid PGVECTOR_TABLE `{table}`")
        return table


def sync_pgvector_from_env(index: RagIndex) -> None:
    store = PgVectorStore.from_env()
    if store is None:
        return
    store.upsert_index(index)


def pgvector_health() -> dict[str, Any]:
    dsn = os.getenv("PGVECTOR_DSN") or os.getenv("DATABASE_URL") or ""
    if not dsn:
        return {"enabled": False, "status": "disabled", "detail": "PGVECTOR_DSN is not configured"}
    try:
        store = PgVectorStore.from_env()
        if store is None:
            return {"enabled": False, "status": "disabled", "detail": "PGVECTOR_DSN is not configured"}
        with store._psycopg.connect(store.config.dsn, connect_timeout=3) as conn:  # type: ignore[attr-defined]
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                version = str(cur.fetchone()[0])
        return {
            "enabled": True,
            "status": "ok",
            "table": store.config.table,
            "detail": version,
        }
    except Exception as exc:
        return {
            "enabled": True,
            "status": "error",
            "detail": f"{type(exc).__name__}: {exc}",
        }


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in values) + "]"
