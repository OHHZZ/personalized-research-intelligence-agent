from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from typing import Protocol


TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_\-/.+]*|[\u4e00-\u9fff]")
EN_SPLIT_RE = re.compile(r"[_\-/+.]+")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "which",
    "with",
    "我",
    "你",
    "的",
    "了",
    "是",
    "在",
    "和",
    "吗",
    "么",
}


class EmbeddingModel(Protocol):
    provider: str
    model_name: str
    dimensions: int

    def embed_query(self, text: str) -> list[float]:
        ...

    def embed_document(self, text: str) -> list[float]:
        ...


def default_embedding_dim() -> int:
    raw = os.getenv("RAG_EMBEDDING_DIM", "384")
    try:
        return max(64, int(raw))
    except ValueError:
        return 384


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese/English research text for local retrieval."""

    tokens: list[str] = []
    chinese_run: list[str] = []
    for raw_token in TOKEN_RE.findall(text.lower()):
        if "\u4e00" <= raw_token <= "\u9fff":
            tokens.append(raw_token)
            chinese_run.append(raw_token)
            continue

        if chinese_run:
            tokens.extend(_chinese_ngrams(chinese_run))
            chinese_run = []

        parts = [part for part in EN_SPLIT_RE.split(raw_token) if part]
        tokens.append(raw_token)
        tokens.extend(parts)

    if chinese_run:
        tokens.extend(_chinese_ngrams(chinese_run))

    return [token for token in tokens if token and token not in STOPWORDS]


def _chinese_ngrams(chars: list[str]) -> list[str]:
    grams: list[str] = []
    for width in (2, 3):
        if len(chars) < width:
            continue
        grams.extend("".join(chars[index : index + width]) for index in range(len(chars) - width + 1))
    return grams


class HashedEmbeddingModel:
    """Dependency-free local embedding model based on signed feature hashing.

    This is not a replacement for a semantic embedding API. It gives the project
    a deterministic retrieval layer that works offline and can later be swapped
    for DashScope/Qwen embeddings or a vector database without changing the
    assistant contract.
    """

    def __init__(self, dimensions: int | None = None) -> None:
        self.provider = "local_hash"
        self.model_name = "signed-feature-hashing"
        self.dimensions = dimensions or default_embedding_dim()

    def embed_query(self, text: str) -> list[float]:
        return self.embed(text)

    def embed_document(self, text: str) -> list[float]:
        return self.embed(text)

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        counts = Counter(tokenize(text))
        if not counts:
            return vector

        for token, count in counts.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(count))

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [round(value / norm, 6) for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    length = min(len(left), len(right))
    return sum(left[index] * right[index] for index in range(length))


class SentenceTransformerEmbeddingModel:
    """Optional embedding provider backed by sentence-transformers.

    `BAAI/bge-base-en-v1.5` is a good first model for this project because the
    current sources are mostly English abstracts, READMEs, and repo metadata.
    The provider is optional so local development still works without PyTorch.
    """

    def __init__(self, model_name: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Run `pip install sentence-transformers`."
            ) from exc

        self.provider = "sentence_transformers"
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        self._model = SentenceTransformer(self.model_name)
        get_dim = getattr(self._model, "get_embedding_dimension",
                          getattr(self._model, "get_sentence_embedding_dimension", None))
        self.dimensions = int(get_dim() if callable(get_dim) else 768)

    def embed_query(self, text: str) -> list[float]:
        return self._encode(_query_text(text, self.model_name))

    def embed_document(self, text: str) -> list[float]:
        return self._encode(text)

    def _encode(self, text: str) -> list[float]:
        vector = self._model.encode(text, normalize_embeddings=True)
        return [round(float(value), 6) for value in vector.tolist()]


def create_embedding_model(provider: str | None = None, dimensions: int | None = None) -> EmbeddingModel:
    requested = (provider or os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")).strip().lower()
    if requested in {"sentence_transformers", "sentence-transformer", "bge", "bge-base-en-v1.5", ""}:
        return SentenceTransformerEmbeddingModel()
    if requested in {"local_hash", "hash", "hashed"}:
        return HashedEmbeddingModel(dimensions)
    raise ValueError(f"Unknown EMBEDDING_PROVIDER `{requested}`")


def _query_text(text: str, model_name: str) -> str:
    lowered = model_name.lower()
    if "bge" in lowered and len(text.split()) < 64:
        return f"Represent this sentence for searching relevant passages: {text}"
    return text
