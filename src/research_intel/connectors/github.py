from __future__ import annotations

import os
from datetime import UTC, datetime

from research_intel.connectors.base import ContentConnector
from research_intel.connectors.http_client import ConnectorError, build_url, get_url, stable_id
from research_intel.models import ContentItem, ContentType, UserProfile


class GitHubConnector(ContentConnector):
    source_name = "github"

    def fetch(self, profile: UserProfile) -> list[ContentItem]:
        """Fetch public GitHub repositories through the REST Search API.

        A token is optional for public resources, but setting GITHUB_TOKEN gives
        higher rate limits and more reliable daily runs.
        """

        items: list[ContentItem] = []
        for query in self._queries(profile):
            try:
                items.extend(self._search(query))
            except ConnectorError:
                continue
        return _dedupe(items)

    def _queries(self, profile: UserProfile) -> list[str]:
        terms = [
            *profile.research_domains[:3],
            *profile.methods[:2],
            *profile.applications[:2],
        ]
        cutoff_year = datetime.now(UTC).year - 1
        queries: list[str] = []
        for term in terms:
            normalized = " ".join(term.strip().split())
            if not normalized:
                continue
            query = f'{normalized} in:name,description,readme stars:>20 pushed:>={cutoff_year}-01-01'
            queries.append(query[:240])
        return queries[: int(os.getenv("LIVE_MAX_QUERIES_PER_SOURCE", "3"))]

    def _search(self, query: str, per_page: int = 6) -> list[ContentItem]:
        url = build_url(
            "https://api.github.com/search/repositories",
            {
                "q": query,
                "sort": "updated",
                "order": "desc",
                "per_page": per_page,
                "page": 1,
            },
        )
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": os.getenv("GITHUB_API_VERSION", "2022-11-28"),
        }
        token = os.getenv("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        payload = get_url(url, headers=headers, timeout=10).json()
        return [self._repo_to_item(repo) for repo in payload.get("items", [])]

    def _repo_to_item(self, repo: dict[str, object]) -> ContentItem:
        full_name = str(repo.get("full_name") or repo.get("name") or "unknown/repo")
        description = str(repo.get("description") or "")
        topics = [str(topic) for topic in repo.get("topics", [])] if isinstance(repo.get("topics"), list) else []
        pushed_at = str(repo.get("pushed_at") or repo.get("updated_at") or "")
        stars = float(repo.get("stargazers_count") or 0)
        forks = float(repo.get("forks_count") or 0)
        language = str(repo.get("language") or "")
        license_payload = repo.get("license") if isinstance(repo.get("license"), dict) else {}
        tags = _infer_tags(full_name, description, topics, language)
        technical_signals = _repo_signals(full_name, description, topics, pushed_at, license_payload, stars)
        return ContentItem(
            item_id=stable_id("github", full_name),
            content_type=ContentType.REPO,
            title=full_name,
            url=str(repo.get("html_url") or ""),
            source="github",
            summary=description or "No repository description provided.",
            tags=tags,
            authors=[str(repo.get("owner", {}).get("login", ""))] if isinstance(repo.get("owner"), dict) else [],
            published_at=str(repo.get("created_at") or ""),
            metrics={
                "stars": stars,
                "forks": forks,
                "open_issues": float(repo.get("open_issues_count") or 0),
            },
            technical_signals=technical_signals,
            links={
                "api": str(repo.get("url") or ""),
                "clone": str(repo.get("clone_url") or ""),
                "homepage": str(repo.get("homepage") or ""),
            },
            raw={
                "full_name": full_name,
                "language": language,
                "pushed_at": pushed_at,
                "archived": bool(repo.get("archived")),
                "disabled": bool(repo.get("disabled")),
            },
        )


def _infer_tags(name: str, description: str, topics: list[str], language: str) -> list[str]:
    text = f"{name} {description} {' '.join(topics)} {language}".lower()
    keyword_tags = {
        "video generation": ["video-generation", "video generation", "text-to-video", "t2v"],
        "controllable video editing": ["video-editing", "video editing", "controllable", "edit"],
        "diffusion models": ["diffusion", "stable-diffusion", "latent-diffusion"],
        "evaluation benchmark": ["benchmark", "evaluation", "metric", "leaderboard"],
        "temporal consistency evaluation": ["temporal", "consistency"],
        "multimodal agents": ["agent", "multimodal", "vlm", "vision-language"],
        "retrieval augmented generation": ["rag", "retrieval", "citation"],
        "academic writing": ["academic", "paper", "writing", "citation"],
        "AI drawing": ["image generation", "drawing", "layout"],
    }
    tags = [tag for tag, needles in keyword_tags.items() if any(needle in text for needle in needles)]
    tags.extend(topics[:5])
    if language:
        tags.append(language)
    return _unique(tags)


def _repo_signals(
    name: str,
    description: str,
    topics: list[str],
    pushed_at: str,
    license_payload: dict[str, object],
    stars: float,
) -> dict[str, object]:
    text = f"{name} {description} {' '.join(topics)}".lower()
    has_examples = any(term in text for term in ("demo", "example", "examples", "inference", "tutorial"))
    has_tests = any(term in text for term in ("test", "pytest", "benchmark", "eval"))
    is_prompt_collection = "prompt" in text and not any(term in text for term in ("code", "model", "framework", "library"))
    last_commit_days = _age_days(pushed_at)
    technical_depth = "medium"
    if any(term in text for term in ("framework", "toolkit", "benchmark", "training", "inference", "pipeline")):
        technical_depth = "high"
    if is_prompt_collection or len(description.strip()) < 24:
        technical_depth = "low"
    return {
        "has_readme": True,
        "has_examples": has_examples,
        "has_tests": has_tests,
        "has_license": bool(license_payload),
        "has_paper_link": any(term in text for term in ("paper", "arxiv", "publication")),
        "has_code": True,
        "technical_depth": technical_depth,
        "last_commit_days": last_commit_days,
        "readme_quality": "thin" if len(description.strip()) < 24 else "metadata_only",
        "baseline_ready": has_examples and not is_prompt_collection,
        "is_prompt_collection": is_prompt_collection,
        "marketing_only": False,
        "trend_signal": 7.2 if stars >= 500 else 6.0 if stars >= 100 else 5.0,
        "has_known_gap": False,
        "technical_core": description[:500] or "Repository metadata did not include a detailed description.",
    }


def _age_days(value: str) -> int:
    if not value:
        return 999
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return 999
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return max(0, (datetime.now(UTC) - parsed).days)


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
