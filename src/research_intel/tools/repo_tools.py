"""Repository-related tools: star velocity and recent activity checks.

Reuses GitHub connector token configuration.  Registered automatically on
import via ``@ToolRegistry.register``.
"""
from __future__ import annotations

import os
import re
from datetime import UTC, datetime

from research_intel.connectors.http_client import build_url, get_url
from research_intel.tools.tool_registry import ToolRegistry


def _parse_github_url(url: str) -> tuple[str, str] | None:
    """Extract ``(owner, repo)`` from a ``github.com`` URL.

    Returns ``None`` if the URL doesn't match the expected pattern.
    """
    m = re.search(r"github\.com/([^/\s]+)/([^/?\s#]+)", url)
    if m:
        owner = m.group(1)
        repo = re.sub(r"\.git$", "", m.group(2))
        return owner, repo
    return None


def _github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": os.getenv("GITHUB_API_VERSION", "2022-11-28"),
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


@ToolRegistry.register("get_repo_star_velocity")
def get_repo_star_velocity(owner: str, repo: str) -> dict[str, float] | None:
    """Fetch current star count and related metrics from GitHub REST API.

    Called when a ``ContentItem`` of type ``repo`` is missing ``stars`` in its
    metrics dict (e.g. discovered from a non-GitHub source).

    Args:
        owner: GitHub username or org name.
        repo: Repository name (without ``.git``).

    Returns:
        Dict with keys ``stars``, ``forks``, ``open_issues``, ``watchers``,
        or ``None`` on failure.
    """
    url = f"https://api.github.com/repos/{owner}/{repo}"
    resp = get_url(url, headers=_github_headers(), timeout=6)
    data = resp.json()
    if not isinstance(data, dict) or "stargazers_count" not in data:
        return None
    return {
        "stars": float(data.get("stargazers_count") or 0),
        "forks": float(data.get("forks_count") or 0),
        "open_issues": float(data.get("open_issues_count") or 0),
        "watchers": float(data.get("watchers_count") or 0),
    }


@ToolRegistry.register("check_repo_recent_activity")
def check_repo_recent_activity(owner: str, repo: str) -> dict[str, int] | None:
    """Check how recently a GitHub repo was committed to.

    Args:
        owner: GitHub username or org name.
        repo: Repository name.

    Returns:
        Dict with ``last_commit_days`` and ``recent_commit_count``, or
        ``None`` on failure.
    """
    url = build_url(
        f"https://api.github.com/repos/{owner}/{repo}/commits",
        {"per_page": 5},
    )
    resp = get_url(url, headers=_github_headers(), timeout=6)
    commits = resp.json()
    if not isinstance(commits, list) or not commits:
        return None
    latest_date = commits[0].get("commit", {}).get("committer", {}).get("date", "")
    if not latest_date:
        return None
    try:
        dt = datetime.fromisoformat(latest_date.replace("Z", "+00:00"))
        days = max(0, (datetime.now(UTC) - dt).days)
        return {"last_commit_days": days, "recent_commit_count": len(commits)}
    except ValueError:
        return None
