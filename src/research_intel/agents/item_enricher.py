from __future__ import annotations

import asyncio

from research_intel.models import ContentItem, ContentType


class ItemEnricher:
    """Fetches supplementary data (abstract, citations, repo metrics) via tool registry."""

    def enrich(self, item: ContentItem) -> ContentItem:
        try:
            from research_intel.tools import paper_tools, repo_tools  # noqa: F401
            from research_intel.tools.tool_registry import ToolRegistry

            if item.content_type == ContentType.PAPER and not item.summary.strip():
                arxiv_id = item.raw.get("arxiv_id", "") if isinstance(item.raw, dict) else ""
                if arxiv_id:
                    abstract = ToolRegistry.call("fetch_paper_abstract", arxiv_id=str(arxiv_id))
                    if abstract:
                        item.summary = str(abstract)

            if item.content_type == ContentType.PAPER and item.metrics.get("citations", 0) == 0:
                arxiv_id = item.raw.get("arxiv_id", "") if isinstance(item.raw, dict) else ""
                cites = ToolRegistry.call(
                    "get_citation_count",
                    title=item.title,
                    arxiv_id=str(arxiv_id),
                )
                if cites:
                    item.metrics["citations"] = float(cites)

            if item.content_type == ContentType.REPO and "stars" not in item.metrics:
                from research_intel.tools.repo_tools import _parse_github_url

                parsed = _parse_github_url(item.url)
                if parsed:
                    owner, repo_name = parsed
                    velocity = ToolRegistry.call("get_repo_star_velocity", owner=owner, repo=repo_name)
                    if isinstance(velocity, dict):
                        item.metrics.update(velocity)
        except Exception:
            pass
        return item

    async def enrich_async(self, item: ContentItem) -> ContentItem:
        try:
            return await asyncio.to_thread(self.enrich, item)
        except Exception:
            return item
