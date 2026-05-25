from __future__ import annotations

import os

from research_intel.connectors.base import ContentConnector
from research_intel.connectors.http_client import ConnectorError, build_url, get_url, stable_id
from research_intel.connectors.signal_helpers import dedupe_items, enrich_with_profile_tags, text_paper_signals, unique
from research_intel.models import ContentItem, ContentType, UserProfile


class PapersWithCodeConnector(ContentConnector):
    source_name = "papers_with_code"

    def fetch(self, profile: UserProfile) -> list[ContentItem]:
        self.last_errors = []
        items: list[ContentItem] = []
        for query in self._queries(profile):
            try:
                items.extend(self._search_papers(query))
            except ConnectorError as exc:
                self.last_errors.append(f"papers query={query}: {exc}")
            try:
                items.extend(self._search_repositories(query))
            except ConnectorError as exc:
                self.last_errors.append(f"repos query={query}: {exc}")
                continue
        if self.last_errors and not items:
            raise ConnectorError("; ".join(self.last_errors))
        return enrich_with_profile_tags(dedupe_items(items), profile)

    def _queries(self, profile: UserProfile) -> list[str]:
        terms = [*profile.research_domains[:3], *profile.methods[:2]]
        return [" ".join(term.strip().split()) for term in terms if term.strip()][: int(os.getenv("LIVE_MAX_QUERIES_PER_SOURCE", "2"))]

    def _search_papers(self, query: str) -> list[ContentItem]:
        url = build_url("https://paperswithcode.com/api/v1/papers/", {"q": query})
        payload = get_url(url, timeout=10).json()
        return [self._paper_to_item(item) for item in payload.get("results", [])[:6]]

    def _search_repositories(self, query: str) -> list[ContentItem]:
        url = build_url("https://paperswithcode.com/api/v1/repositories/", {"q": query})
        payload = get_url(url, timeout=10).json()
        return [self._repo_to_item(item) for item in payload.get("results", [])[:6]]

    def _paper_to_item(self, paper: dict[str, object]) -> ContentItem:
        title = str(paper.get("title") or "Untitled Papers with Code paper")
        abstract = str(paper.get("abstract") or "")
        paper_id = str(paper.get("id") or paper.get("arxiv_id") or title)
        links = {
            "paper": str(paper.get("url_abs") or paper.get("url_pdf") or ""),
            "pdf": str(paper.get("url_pdf") or ""),
            "pwc": f"https://paperswithcode.com/paper/{paper_id}",
        }
        signals = text_paper_signals(title, abstract)
        signals["has_code"] = True
        signals["pwc_has_code"] = True
        return ContentItem(
            item_id=stable_id("pwc_paper", paper_id),
            content_type=ContentType.PAPER,
            title=title,
            url=links["paper"] or links["pwc"],
            source="papers_with_code",
            summary=abstract,
            tags=[],
            authors=[str(author) for author in paper.get("authors", [])] if isinstance(paper.get("authors"), list) else [],
            published_at=str(paper.get("published") or ""),
            metrics={},
            technical_signals=signals,
            links=links,
            raw=paper,
        )

    def _repo_to_item(self, repo: dict[str, object]) -> ContentItem:
        url = str(repo.get("url") or repo.get("github_url") or "")
        name = str(repo.get("name") or url.rstrip("/").split("/")[-1] or "Papers with Code repository")
        summary = str(repo.get("description") or "Repository linked from Papers with Code.")
        stars = float(repo.get("stars") or 0)
        return ContentItem(
            item_id=stable_id("pwc_repo", url or name),
            content_type=ContentType.REPO,
            title=name,
            url=url,
            source="papers_with_code",
            summary=summary,
            tags=[],
            authors=[],
            published_at=str(repo.get("created_at") or ""),
            metrics={"stars": stars},
            technical_signals={
                "has_readme": True,
                "has_examples": "demo" in summary.lower() or "example" in summary.lower(),
                "has_tests": "test" in summary.lower() or "benchmark" in summary.lower(),
                "has_license": bool(repo.get("license")),
                "has_paper_link": True,
                "has_code": True,
                "technical_depth": "medium",
                "last_commit_days": 365,
                "readme_quality": "metadata_only",
                "baseline_ready": True,
                "trend_signal": min(9.0, 4.5 + stars / 2000.0),
                "technical_core": summary[:500],
            },
            links={"pwc": str(repo.get("paper_url") or "")},
            raw=repo,
        )


