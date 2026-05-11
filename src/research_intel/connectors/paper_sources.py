from __future__ import annotations

import re
import os
import time
import xml.etree.ElementTree as ET

from research_intel.connectors.base import ContentConnector
from research_intel.connectors.http_client import ConnectorError, build_url, get_url, stable_id
from research_intel.models import ContentItem, ContentType, UserProfile


ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"


class PaperSourceConnector(ContentConnector):
    source_name = "paper_sources"

    def fetch(self, profile: UserProfile) -> list[ContentItem]:
        """Fetch papers from arXiv.

        arXiv returns Atom XML. This connector keeps the call volume small and
        only uses profile-derived queries. Other paper sources can be added
        behind the same ContentConnector contract.
        """

        self.last_errors = []
        items: list[ContentItem] = []
        delay = float(os.getenv("ARXIV_REQUEST_DELAY_SECONDS", "3"))
        for index, query in enumerate(self._queries(profile)):
            if index and delay > 0:
                time.sleep(delay)
            try:
                items.extend(self._fetch_arxiv(query))
            except ConnectorError as exc:
                self.last_errors.append(f"query={query}: {exc}")
                if "HTTP 429" in str(exc):
                    break
                continue
        if self.last_errors and not items:
            raise ConnectorError("; ".join(self.last_errors))
        return _dedupe(items)

    def _queries(self, profile: UserProfile) -> list[str]:
        seed_terms = [
            *profile.research_domains[:3],
            *profile.methods[:2],
            *profile.applications[:2],
        ]
        queries: list[str] = []
        for term in seed_terms:
            normalized = " ".join(term.strip().split())
            if not normalized:
                continue
            quoted = f'ti:"{normalized}" OR abs:"{normalized}"'
            scoped = f"({quoted}) AND (cat:cs.CV OR cat:cs.AI OR cat:cs.LG OR cat:cs.CL)"
            queries.append(scoped)
        return queries[: int(os.getenv("LIVE_MAX_QUERIES_PER_SOURCE", "3"))]

    def _fetch_arxiv(self, search_query: str, max_results: int = 8) -> list[ContentItem]:
        max_results = int(os.getenv("ARXIV_RESULTS_PER_QUERY", str(max_results)))
        url = build_url(
            "https://export.arxiv.org/api/query",
            {
                "search_query": search_query,
                "start": 0,
                "max_results": max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
        )
        response = get_url(url, timeout=10)
        root = ET.fromstring(response.body)
        return [self._entry_to_item(entry) for entry in root.findall(f"{ATOM}entry")]

    def _entry_to_item(self, entry: ET.Element) -> ContentItem:
        title = _clean_text(entry.findtext(f"{ATOM}title") or "Untitled arXiv paper")
        summary = _clean_text(entry.findtext(f"{ATOM}summary") or "")
        url = entry.findtext(f"{ATOM}id") or ""
        arxiv_id = url.rstrip("/").split("/")[-1] if url else title
        authors = [
            _clean_text(author.findtext(f"{ATOM}name") or "")
            for author in entry.findall(f"{ATOM}author")
        ]
        categories = [
            category.attrib.get("term", "")
            for category in entry.findall(f"{ATOM}category")
            if category.attrib.get("term")
        ]
        tags = _infer_tags(title, summary, categories)
        links = {
            link.attrib.get("title", link.attrib.get("type", "link")): link.attrib.get("href", "")
            for link in entry.findall(f"{ATOM}link")
            if link.attrib.get("href")
        }
        technical_signals = _paper_signals(title, summary, links)
        return ContentItem(
            item_id=stable_id("arxiv", arxiv_id),
            content_type=ContentType.PAPER,
            title=title,
            url=url,
            source="arxiv",
            summary=summary,
            tags=tags,
            authors=[author for author in authors if author],
            published_at=entry.findtext(f"{ATOM}published"),
            metrics={},
            technical_signals=technical_signals,
            links=links,
            raw={
                "arxiv_id": arxiv_id,
                "updated": entry.findtext(f"{ATOM}updated"),
                "categories": categories,
            },
        )


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _infer_tags(title: str, summary: str, categories: list[str]) -> list[str]:
    text = f"{title} {summary}".lower()
    keyword_tags = {
        "video generation": ["video generation", "text-to-video", "video diffusion"],
        "controllable video editing": ["video editing", "controllable", "instruction-guided"],
        "diffusion models": ["diffusion", "latent diffusion"],
        "evaluation benchmark": ["benchmark", "evaluation", "metric", "leaderboard"],
        "temporal consistency evaluation": ["temporal consistency", "temporal"],
        "multimodal agents": ["agent", "multimodal", "vision-language"],
        "retrieval augmented generation": ["retrieval", "rag", "citation"],
        "academic writing": ["scientific writing", "academic writing", "citation"],
        "AI drawing": ["image generation", "drawing", "layout"],
    }
    tags = [tag for tag, needles in keyword_tags.items() if any(needle in text for needle in needles)]
    tags.extend(categories[:3])
    return _unique(tags)


def _paper_signals(title: str, summary: str, links: dict[str, str]) -> dict[str, object]:
    text = f"{title} {summary}".lower()
    has_eval = any(term in text for term in ("experiment", "evaluation", "benchmark", "metric", "dataset"))
    has_ablation = "ablation" in text
    has_baseline = any(term in text for term in ("baseline", "state-of-the-art", "sota", "compare"))
    has_code = any("github" in value.lower() or "code" in key.lower() for key, value in links.items())
    novelty = 6.2
    if any(term in text for term in ("new", "novel", "first", "propose", "introduce")):
        novelty += 0.8
    if any(term in text for term in ("benchmark", "dataset", "evaluation")):
        novelty += 0.5
    depth = "medium" if has_eval else "low"
    if has_eval and (has_ablation or has_baseline):
        depth = "high"
    return {
        "has_experiments": has_eval,
        "has_ablation": has_ablation,
        "has_strong_baselines": has_baseline,
        "has_code": has_code,
        "has_benchmark": "benchmark" in text,
        "baseline_count": 2 if has_baseline else 0,
        "novelty": min(9.0, novelty),
        "technical_depth": depth,
        "trend_signal": 6.0 if has_eval else 4.8,
        "has_known_gap": any(term in text for term in ("limitation", "challenge", "gap", "future work")),
        "benchmark_gap": any(term in text for term in ("metric", "benchmark", "evaluation gap")),
        "technical_core": summary[:500],
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
