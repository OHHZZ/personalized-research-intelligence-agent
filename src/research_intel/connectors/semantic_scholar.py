from __future__ import annotations

import os
import time

from research_intel.connectors.base import ContentConnector
from research_intel.connectors.http_client import ConnectorError, build_url, get_url, stable_id
from research_intel.models import ContentItem, ContentType, UserProfile


class SemanticScholarConnector(ContentConnector):
    source_name = "semantic_scholar"

    def fetch(self, profile: UserProfile) -> list[ContentItem]:
        self.last_errors = []
        items: list[ContentItem] = []
        delay = float(os.getenv("SEMANTIC_SCHOLAR_REQUEST_DELAY_SECONDS", "1"))
        for index, query in enumerate(self._queries(profile)):
            if index and delay > 0:
                time.sleep(delay)
            try:
                items.extend(self._search(query))
            except ConnectorError as exc:
                self.last_errors.append(f"query={query}: {exc}")
                if "HTTP 429" in str(exc):
                    break
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
        return [" ".join(term.strip().split()) for term in terms if term.strip()][: int(os.getenv("LIVE_MAX_QUERIES_PER_SOURCE", "3"))]

    def _search(self, query: str, limit: int = 6) -> list[ContentItem]:
        fields = ",".join(
            [
                "paperId",
                "title",
                "abstract",
                "url",
                "year",
                "publicationDate",
                "citationCount",
                "influentialCitationCount",
                "referenceCount",
                "isOpenAccess",
                "openAccessPdf",
                "fieldsOfStudy",
                "externalIds",
                "authors",
            ]
        )
        url = build_url(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            {"query": query, "limit": limit, "fields": fields},
        )
        headers: dict[str, str] = {}
        api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        if api_key:
            headers["x-api-key"] = api_key
        payload = get_url(url, headers=headers, timeout=10).json()
        return [self._paper_to_item(paper) for paper in payload.get("data", [])]

    def _paper_to_item(self, paper: dict[str, object]) -> ContentItem:
        title = str(paper.get("title") or "Untitled Semantic Scholar paper")
        abstract = str(paper.get("abstract") or "")
        paper_id = str(paper.get("paperId") or title)
        external_ids = paper.get("externalIds") if isinstance(paper.get("externalIds"), dict) else {}
        authors = paper.get("authors") if isinstance(paper.get("authors"), list) else []
        fields = paper.get("fieldsOfStudy") if isinstance(paper.get("fieldsOfStudy"), list) else []
        tags = _infer_tags(title, abstract, [str(field) for field in fields])
        open_pdf = paper.get("openAccessPdf") if isinstance(paper.get("openAccessPdf"), dict) else {}
        links = {
            "semantic_scholar": str(paper.get("url") or ""),
            "pdf": str(open_pdf.get("url") or ""),
        }
        if external_ids.get("ArXiv"):
            links["arxiv"] = f"https://arxiv.org/abs/{external_ids['ArXiv']}"
        return ContentItem(
            item_id=stable_id("s2", paper_id),
            content_type=ContentType.PAPER,
            title=title,
            url=str(paper.get("url") or links.get("arxiv") or ""),
            source="semantic_scholar",
            summary=abstract,
            tags=tags,
            authors=[str(author.get("name")) for author in authors if isinstance(author, dict) and author.get("name")],
            published_at=str(paper.get("publicationDate") or paper.get("year") or ""),
            metrics={
                "citations": float(paper.get("citationCount") or 0),
                "influential_citations": float(paper.get("influentialCitationCount") or 0),
                "references": float(paper.get("referenceCount") or 0),
            },
            technical_signals=_paper_signals(title, abstract, paper, links),
            links=links,
            raw={"paperId": paper_id, "externalIds": external_ids},
        )


def _infer_tags(title: str, abstract: str, fields: list[str]) -> list[str]:
    text = f"{title} {abstract} {' '.join(fields)}".lower()
    tags: list[str] = []
    mapping = {
        "video generation": ["video generation", "text-to-video", "video diffusion"],
        "controllable video editing": ["video editing", "controllable", "editing"],
        "diffusion models": ["diffusion"],
        "evaluation benchmark": ["benchmark", "evaluation", "metric"],
        "temporal consistency evaluation": ["temporal consistency", "temporal"],
        "multimodal agents": ["agent", "multimodal", "vision-language"],
        "retrieval augmented generation": ["retrieval", "rag", "citation"],
        "academic writing": ["writing", "citation", "paper"],
        "AI drawing": ["image generation", "drawing"],
    }
    for tag, needles in mapping.items():
        if any(needle in text for needle in needles):
            tags.append(tag)
    tags.extend(fields[:3])
    return _unique(tags)


def _paper_signals(
    title: str,
    abstract: str,
    paper: dict[str, object],
    links: dict[str, str],
) -> dict[str, object]:
    text = f"{title} {abstract}".lower()
    has_eval = any(term in text for term in ("experiment", "evaluation", "benchmark", "dataset", "metric"))
    has_baseline = any(term in text for term in ("baseline", "state-of-the-art", "sota", "compare"))
    citations = float(paper.get("citationCount") or 0)
    influential = float(paper.get("influentialCitationCount") or 0)
    has_code_hint = any(term in text for term in ("code", "github", "implementation")) or bool(links.get("pdf"))
    return {
        "has_experiments": has_eval,
        "has_ablation": "ablation" in text,
        "has_strong_baselines": has_baseline,
        "has_code": "github" in text,
        "has_benchmark": "benchmark" in text,
        "baseline_count": 2 if has_baseline else 0,
        "novelty": 6.7 if any(term in text for term in ("novel", "propose", "introduce")) else 5.8,
        "technical_depth": "high" if has_eval and has_baseline else "medium" if has_eval else "low",
        "trend_signal": min(8.5, 5.0 + citations / 200.0 + influential / 30.0),
        "has_known_gap": any(term in text for term in ("limitation", "challenge", "gap")),
        "benchmark_gap": any(term in text for term in ("metric", "benchmark")),
        "technical_core": abstract[:500],
        "semantic_scholar_open_access": bool(paper.get("isOpenAccess")),
        "semantic_scholar_has_pdf": has_code_hint,
    }


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        key = value.strip().lower()
        if key and key not in seen:
            seen.add(key)
            output.append(value.strip())
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
