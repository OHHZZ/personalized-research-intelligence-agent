"""Paper-related tools: abstract fetching and citation count lookup.

Both tools reuse HTTP helpers from the existing connectors so no new
external dependencies are introduced.  They are registered automatically
on import via ``@ToolRegistry.register``.
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET

from research_intel.connectors.http_client import build_url, get_url
from research_intel.tools.tool_registry import ToolRegistry

_ATOM = "{http://www.w3.org/2005/Atom}"


@ToolRegistry.register("fetch_paper_abstract")
def fetch_paper_abstract(arxiv_id: str) -> str | None:
    """Fetch a paper abstract from arXiv by its arXiv ID.

    Used when a ``ContentItem`` has an empty summary (e.g. because the
    original connector timed out or the abstract field was missing).

    Args:
        arxiv_id: Raw arXiv ID, e.g. ``"2301.12345"`` or ``"arxiv:2301.12345"``.

    Returns:
        Plain-text abstract string, or ``None`` if unavailable.
    """
    clean = re.sub(r"^arxiv[:/]", "", arxiv_id, flags=re.IGNORECASE).strip()
    if not clean:
        return None
    url = build_url(
        "https://export.arxiv.org/api/query",
        {"id_list": clean, "max_results": 1},
    )
    resp = get_url(url, timeout=6)
    root = ET.fromstring(resp.body)
    entry = root.find(f"{_ATOM}entry")
    if entry is None:
        return None
    summary = entry.findtext(f"{_ATOM}summary") or ""
    cleaned = re.sub(r"\s+", " ", summary).strip()
    return cleaned or None


@ToolRegistry.register("get_citation_count")
def get_citation_count(title: str, arxiv_id: str = "") -> int | None:
    """Look up a paper's citation count via the Semantic Scholar API.

    Tries by arXiv ID first (more precise), then falls back to title search.

    Args:
        title: Paper title, used as fallback query.
        arxiv_id: Optional arXiv ID for exact lookup.

    Returns:
        Integer citation count, or ``None`` if the lookup fails.
    """
    headers: dict[str, str] = {}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    # Attempt 1: exact arXiv ID lookup
    clean_arxiv = re.sub(r"^arxiv[:/]", "", arxiv_id, flags=re.IGNORECASE).strip()
    if clean_arxiv:
        url = f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{clean_arxiv}?fields=citationCount"
        resp = get_url(url, headers=headers, timeout=6)
        data = resp.json()
        if isinstance(data, dict) and "citationCount" in data:
            val = data["citationCount"]
            return int(val) if val is not None else None

    # Attempt 2: title search
    if not title.strip():
        return None
    url = build_url(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        {"query": title[:120], "limit": 1, "fields": "citationCount"},
    )
    resp = get_url(url, headers=headers, timeout=6)
    data = resp.json()
    papers = data.get("data", [])
    if papers and isinstance(papers[0], dict):
        val = papers[0].get("citationCount")
        if val is not None:
            count = int(val)
            return count if count > 0 else None
    return None
