from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from research_intel.connectors.http_client import ConnectorError, get_url
from research_intel.models import ContentItem


README_LIMIT = 24000
TREE_LIMIT = 500
FILE_LIMIT = 12000


@dataclass(slots=True)
class RepoSnapshot:
    full_name: str
    html_url: str = ""
    default_branch: str = "main"
    description: str = ""
    readme: str = ""
    files: list[str] = field(default_factory=list)
    key_files: dict[str, str] = field(default_factory=dict)
    fetched: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "full_name": self.full_name,
            "html_url": self.html_url,
            "default_branch": self.default_branch,
            "description": self.description,
            "readme": self.readme,
            "files": self.files,
            "key_files": self.key_files,
            "fetched": self.fetched,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RepoSnapshot":
        return cls(
            full_name=str(payload.get("full_name", "")),
            html_url=str(payload.get("html_url", "")),
            default_branch=str(payload.get("default_branch", "main")),
            description=str(payload.get("description", "")),
            readme=str(payload.get("readme", "")),
            files=[str(item) for item in payload.get("files", [])],
            key_files={str(key): str(value) for key, value in payload.get("key_files", {}).items()},
            fetched=bool(payload.get("fetched", False)),
            error=str(payload.get("error", "")),
        )

    @property
    def has_readme(self) -> bool:
        return bool(self.readme.strip())

    @property
    def has_examples(self) -> bool:
        return any(_path_has(path, ("example", "examples", "demo", "demos", "inference")) for path in self.files)

    @property
    def has_tests(self) -> bool:
        return any(_path_has(path, ("test", "tests", "pytest")) for path in self.files)

    @property
    def has_license(self) -> bool:
        return any(path.lower().split("/")[-1].startswith("license") for path in self.files)


class GitHubRepoInspector:
    def __init__(self, project_root: Path | str | None = None) -> None:
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.cache_dir = self.project_root / "data" / "runs" / "repo_cache"

    def inspect(self, repo: ContentItem, use_cache: bool = True) -> RepoSnapshot:
        full_name = _repo_full_name(repo)
        if not full_name:
            return RepoSnapshot(full_name=repo.title, description=repo.summary, error="cannot_parse_repo_full_name")

        cache_path = self.cache_dir / f"{_safe_name(full_name)}.json"
        if use_cache and cache_path.exists():
            try:
                return RepoSnapshot.from_dict(json.loads(cache_path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                pass

        try:
            snapshot = self._fetch(full_name)
        except ConnectorError as exc:
            snapshot = RepoSnapshot(
                full_name=full_name,
                html_url=repo.url,
                description=repo.summary,
                fetched=False,
                error=str(exc),
            )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(snapshot.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return snapshot

    def _fetch(self, full_name: str) -> RepoSnapshot:
        repo_payload = self._github_json(f"https://api.github.com/repos/{full_name}")
        default_branch = str(repo_payload.get("default_branch") or "main")
        snapshot = RepoSnapshot(
            full_name=full_name,
            html_url=str(repo_payload.get("html_url") or f"https://github.com/{full_name}"),
            default_branch=default_branch,
            description=str(repo_payload.get("description") or ""),
            fetched=True,
        )
        snapshot.readme = self._fetch_readme(full_name)
        snapshot.files = self._fetch_tree(full_name, default_branch)
        snapshot.key_files = self._fetch_key_files(full_name, default_branch, snapshot.files)
        return snapshot

    def _fetch_readme(self, full_name: str) -> str:
        payload = self._github_json(f"https://api.github.com/repos/{full_name}/readme")
        content = str(payload.get("content") or "")
        encoding = str(payload.get("encoding") or "")
        if content and encoding == "base64":
            return base64.b64decode(content).decode("utf-8", errors="replace")[:README_LIMIT]
        download_url = str(payload.get("download_url") or "")
        if download_url:
            return get_url(download_url, headers=self._headers(), timeout=8).text()[:README_LIMIT]
        return ""

    def _fetch_tree(self, full_name: str, branch: str) -> list[str]:
        payload = self._github_json(f"https://api.github.com/repos/{full_name}/git/trees/{branch}?recursive=1")
        files: list[str] = []
        for item in payload.get("tree", []) if isinstance(payload.get("tree"), list) else []:
            if not isinstance(item, dict) or item.get("type") != "blob":
                continue
            path = str(item.get("path") or "")
            if path:
                files.append(path)
        return files[:TREE_LIMIT]

    def _fetch_key_files(self, full_name: str, branch: str, files: list[str]) -> dict[str, str]:
        wanted_names = {
            "pyproject.toml",
            "requirements.txt",
            "setup.py",
            "environment.yml",
            "package.json",
            "Dockerfile",
        }
        selected = [
            path
            for path in files
            if path.split("/")[-1] in wanted_names or path.lower().startswith(("examples/", "demo/", "demos/"))
        ][:8]
        output: dict[str, str] = {}
        for path in selected:
            url_path = "/".join(_quote_part(part) for part in path.split("/"))
            raw_url = f"https://raw.githubusercontent.com/{full_name}/{branch}/{url_path}"
            try:
                output[path] = get_url(raw_url, headers=self._headers(), timeout=8).text()[:FILE_LIMIT]
            except ConnectorError:
                continue
        return output

    def _github_json(self, url: str) -> dict[str, Any]:
        return get_url(url, headers=self._headers(), timeout=8).json()

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": os.getenv("GITHUB_API_VERSION", "2022-11-28"),
        }
        token = os.getenv("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers


def _repo_full_name(repo: ContentItem) -> str:
    raw_full_name = repo.raw.get("full_name") if isinstance(repo.raw, dict) else None
    if raw_full_name:
        return str(raw_full_name)
    if _looks_like_full_name(repo.title):
        return repo.title.strip()
    parsed = urlparse(repo.url)
    if parsed.netloc.lower() == "github.com":
        parts = [part for part in parsed.path.strip("/").split("/") if part]
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
    return ""


def _looks_like_full_name(value: str) -> bool:
    return bool(re.fullmatch(r"[\w.-]+/[\w.-]+", value.strip()))


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def _path_has(path: str, terms: tuple[str, ...]) -> bool:
    lowered = path.lower()
    return any(term in lowered for term in terms)


def _quote_part(value: str) -> str:
    from urllib.parse import quote

    return quote(value, safe="")
