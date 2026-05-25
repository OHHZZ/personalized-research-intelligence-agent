from __future__ import annotations

import re
import os
import time
import xml.etree.ElementTree as ET

from research_intel.connectors.base import ContentConnector
from research_intel.connectors.http_client import ConnectorError, build_url, get_url, stable_id
from research_intel.connectors.signal_helpers import dedupe_items, enrich_with_profile_tags, text_paper_signals, unique
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
        return enrich_with_profile_tags(dedupe_items(items), profile)

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
    return unique(categories[:3])


def _paper_signals(title: str, summary: str, links: dict[str, str]) -> dict[str, object]:
    signals = text_paper_signals(title, summary)
    has_code_link = any("github" in value.lower() or "code" in key.lower() for key, value in links.items())
    signals["has_code"] = signals["has_code"] or has_code_link
    signals["trend_signal"] = 6.0 if signals["has_experiments"] else 4.8
    return signals
