from __future__ import annotations

import os

from research_intel.connectors.base import ContentConnector
from research_intel.connectors.http_client import ConnectorError, build_url, get_url, stable_id
from research_intel.models import ContentItem, ContentType, UserProfile


class OpenAlexConnector(ContentConnector):
    source_name = "openalex"

    def fetch(self, profile: UserProfile) -> list[ContentItem]:
        self.last_errors = []
        items: list[ContentItem] = []
        for query in self._queries(profile):
            try:
                items.extend(self._search(query))
            except ConnectorError as exc:
                self.last_errors.append(f"query={query}: {exc}")
                continue
        if self.last_errors and not items:
            raise ConnectorError("; ".join(self.last_errors))
        return _dedupe(items)

    def _queries(self, profile: UserProfile) -> list[str]:
        terms = [
            *profile.research_domains[:3],
            *profile.methods[:2],
            *profile.applications[:2],
        ]
        limit = int(os.getenv("OPENALEX_MAX_QUERIES", os.getenv("LIVE_MAX_QUERIES_PER_SOURCE", "3")))
        return [" ".join(term.strip().split()) for term in terms if term.strip()][:limit]

    def _search(self, query: str, limit: int = 8) -> list[ContentItem]:
        params = {
            "search": query,
            "per-page": min(limit, int(os.getenv("OPENALEX_RESULTS_PER_QUERY", "8"))),
            "sort": os.getenv("OPENALEX_SORT", "relevance_score:desc"),
        }
        mailto = os.getenv("OPENALEX_MAILTO", "").strip()
        if mailto:
            params["mailto"] = mailto
        url = build_url("https://api.openalex.org/works", params)
        payload = get_url(url, timeout=12).json()
        results = payload.get("results", []) if isinstance(payload, dict) else []
        return [self._work_to_item(work) for work in results if isinstance(work, dict)]

    def _work_to_item(self, work: dict[str, object]) -> ContentItem:
        title = str(work.get("title") or work.get("display_name") or "Untitled OpenAlex work")
        abstract = _abstract_from_inverted_index(work.get("abstract_inverted_index"))
        work_id = str(work.get("id") or work.get("doi") or title)
        concepts = work.get("concepts") if isinstance(work.get("concepts"), list) else []
        authorships = work.get("authorships") if isinstance(work.get("authorships"), list) else []
        primary_location = work.get("primary_location") if isinstance(work.get("primary_location"), dict) else {}
        source = primary_location.get("source") if isinstance(primary_location.get("source"), dict) else {}
        pdf_url = str(primary_location.get("pdf_url") or "")
        landing_url = str(primary_location.get("landing_page_url") or "")
        url = str(work.get("doi") or landing_url or work_id)
        tags = _infer_tags(title, abstract, concepts)
        return ContentItem(
            item_id=stable_id("openalex", work_id),
            content_type=ContentType.PAPER,
            title=title,
            url=url,
            source="openalex",
            summary=abstract,
            tags=tags,
            authors=_authors(authorships),
            published_at=str(work.get("publication_date") or ""),
            metrics={
                "citations": float(work.get("cited_by_count") or 0),
            },
            technical_signals=_paper_signals(title, abstract, work, pdf_url),
            links={
                "openalex": work_id,
                "doi": str(work.get("doi") or ""),
                "pdf": pdf_url,
                "landing_page": landing_url,
                "venue": str(source.get("display_name") or "") if isinstance(source, dict) else "",
            },
            raw={
                "type": work.get("type"),
                "publication_year": work.get("publication_year"),
                "is_open_access": (work.get("open_access") or {}).get("is_oa")
                if isinstance(work.get("open_access"), dict)
                else None,
            },
        )


def _abstract_from_inverted_index(value: object) -> str:
    if not isinstance(value, dict):
        return ""
    positions: list[tuple[int, str]] = []
    for token, indexes in value.items():
        if not isinstance(indexes, list):
            continue
        for index in indexes:
            if isinstance(index, int):
                positions.append((index, str(token)))
    return " ".join(token for _, token in sorted(positions))[:1600]


def _authors(authorships: list[object]) -> list[str]:
    output: list[str] = []
    for item in authorships[:8]:
        if not isinstance(item, dict):
            continue
        author = item.get("author")
        if isinstance(author, dict) and author.get("display_name"):
            output.append(str(author["display_name"]))
    return output


def _infer_tags(title: str, abstract: str, concepts: list[object]) -> list[str]:
    concept_names = [
        str(concept.get("display_name", ""))
        for concept in concepts[:6]
        if isinstance(concept, dict)
    ]
    text = f"{title} {abstract} {' '.join(concept_names)}".lower()
    mapping = {
        "video generation": ["video generation", "text-to-video", "video diffusion"],
        "controllable video editing": ["video editing", "controllable", "editing"],
        "diffusion models": ["diffusion"],
        "evaluation benchmark": ["benchmark", "evaluation", "metric"],
        "temporal consistency evaluation": ["temporal consistency", "temporal coherence", "temporal"],
        "multimodal agents": ["agent", "multimodal", "vision-language"],
        "retrieval augmented generation": ["retrieval", "rag"],
        "academic writing": ["scientific writing", "citation"],
        "AI drawing": ["image generation", "drawing"],
    }
    tags = [tag for tag, needles in mapping.items() if any(needle in text for needle in needles)]
    tags.extend(concept_names[:3])
    return _unique(tags)


def _paper_signals(title: str, abstract: str, work: dict[str, object], pdf_url: str) -> dict[str, object]:
    text = f"{title} {abstract}".lower()
    has_eval = any(term in text for term in ("experiment", "evaluation", "benchmark", "metric", "dataset"))
    has_baseline = any(term in text for term in ("baseline", "state-of-the-art", "sota", "compare"))
    citations = float(work.get("cited_by_count") or 0)
    return {
        "has_experiments": has_eval,
        "has_ablation": "ablation" in text,
        "has_strong_baselines": has_baseline,
        "has_code": any(term in text for term in ("github", "code", "implementation")),
        "has_benchmark": "benchmark" in text,
        "baseline_count": 2 if has_baseline else 0,
        "novelty": 6.7 if any(term in text for term in ("novel", "propose", "introduce")) else 5.8,
        "technical_depth": "high" if has_eval and has_baseline else "medium" if has_eval else "low",
        "trend_signal": min(8.0, 5.0 + citations / 250.0),
        "has_known_gap": any(term in text for term in ("limitation", "challenge", "gap")),
        "benchmark_gap": any(term in text for term in ("metric", "benchmark")),
        "technical_core": abstract[:500],
        "openalex_has_pdf": bool(pdf_url),
    }


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = value.strip()
        key = normalized.lower()
        if normalized and key not in seen:
            seen.add(key)
            output.append(normalized)
    return output


def _dedupe(items: list[ContentItem]) -> list[ContentItem]:
    seen: set[str] = set()
    output: list[ContentItem] = []
    for item in items:
        key = item.url.lower() or item.title.lower()
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output
