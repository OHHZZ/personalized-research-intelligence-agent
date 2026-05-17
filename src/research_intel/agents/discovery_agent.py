from __future__ import annotations

import asyncio
from collections.abc import Callable

from research_intel.agents.base import BaseAgent
from research_intel.connectors import (
    ContentConnector,
    GitHubConnector,
    OpenAlexConnector,
    PapersWithCodeConnector,
    PaperSourceConnector,
    SemanticScholarConnector,
)
from research_intel.connectors.http_client import ConnectorError
from research_intel.models import ContentItem, UserProfile
from research_intel.storage import JsonStore

ProgressCallback = Callable[[dict[str, object]], None]


class DiscoveryAgent(BaseAgent):
    """Loads candidate content from sample data, live connectors, or both."""

    name = "discovery-agent"

    def __init__(
        self,
        store: JsonStore,
        sample_name: str = "content_items",
        connectors: list[ContentConnector] | None = None,
    ) -> None:
        self.store = store
        self.sample_name = sample_name
        self.connectors = connectors or [
            PaperSourceConnector(),
            SemanticScholarConnector(),
            OpenAlexConnector(),
            PapersWithCodeConnector(),
            GitHubConnector(),
        ]
        self.last_errors: list[str] = []

    def discover(
        self,
        profile: UserProfile,
        source_mode: str = "hybrid",
        progress: ProgressCallback | None = None,
    ) -> list[ContentItem]:
        source_mode = source_mode.lower().strip()
        if source_mode not in {"sample", "live", "hybrid"}:
            raise ValueError("source_mode must be one of: sample, live, hybrid")

        items: list[ContentItem] = []
        self.last_errors = []

        if source_mode in {"live", "hybrid"}:
            items.extend(self._live_items(profile, progress=progress))

        if source_mode == "sample" or (source_mode == "hybrid" and len(items) < 5):
            if progress:
                progress(
                    {
                        "stage": "discovery",
                        "status": "running",
                        "message": "Loading sample content",
                        "source": self.sample_name,
                    }
                )
            sample_items = self._sample_items()
            items.extend(sample_items)
            if progress:
                progress(
                    {
                        "stage": "discovery",
                        "status": "complete",
                        "message": f"Loaded {len(sample_items)} sample items",
                        "source": self.sample_name,
                        "count": len(sample_items),
                    }
                )

        items = self._dedupe(items)
        self.store.save_content_items(items, "latest_candidates")
        preferred = {value.lower() for value in profile.preferred_content}
        if not preferred:
            return items
        return [item for item in items if item.content_type.value in preferred or item.content_type.value == "tool"]

    def _sample_items(self) -> list[ContentItem]:
        try:
            return self.store.load_content_items(self.sample_name)
        except FileNotFoundError:
            return []

    def _live_items(self, profile: UserProfile, progress: ProgressCallback | None = None) -> list[ContentItem]:
        items: list[ContentItem] = []
        for connector in self.connectors:
            if progress:
                progress(
                    {
                        "stage": "discovery",
                        "status": "running",
                        "message": f"Fetching {connector.source_name}",
                        "connector": connector.source_name,
                    }
                )
            try:
                fetched = connector.fetch(profile)
                items.extend(fetched)
                if progress:
                    progress(
                        {
                            "stage": "discovery",
                            "status": "complete",
                            "message": f"{connector.source_name} returned {len(fetched)} items",
                            "connector": connector.source_name,
                            "count": len(fetched),
                        }
                    )
                if getattr(connector, "last_errors", None):
                    error_count = len(connector.last_errors)
                    self.last_errors.extend(f"{connector.source_name}: {message}" for message in connector.last_errors)
                    if progress:
                        progress(
                            {
                                "stage": "discovery",
                                "status": "warning",
                                "message": f"{connector.source_name} reported {error_count} fetch issue(s); details kept in backend artifacts",
                                "connector": connector.source_name,
                                "source_error_count": error_count,
                            }
                        )
            except ConnectorError as exc:
                self.last_errors.append(f"{connector.source_name}: {exc}")
                if progress:
                    progress(
                        {
                            "stage": "discovery",
                            "status": "warning",
                            "message": f"{connector.source_name} fetch was skipped; details kept in backend artifacts",
                            "connector": connector.source_name,
                            "source_error_count": 1,
                        }
                    )
            except Exception as exc:
                self.last_errors.append(f"{connector.source_name}: {type(exc).__name__}: {exc}")
                if progress:
                    progress(
                        {
                            "stage": "discovery",
                            "status": "warning",
                            "message": f"{connector.source_name} fetch failed; details kept in backend artifacts",
                            "connector": connector.source_name,
                            "source_error_count": 1,
                        }
                    )
        return items

    async def discover_async(
        self,
        profile: UserProfile,
        source_mode: str = "hybrid",
        progress: ProgressCallback | None = None,
    ) -> list[ContentItem]:
        """Async version of discover(): runs all live connectors in parallel."""
        source_mode = source_mode.lower().strip()
        if source_mode not in {"sample", "live", "hybrid"}:
            raise ValueError("source_mode must be one of: sample, live, hybrid")

        items: list[ContentItem] = []
        self.last_errors = []

        if source_mode in {"live", "hybrid"}:
            items.extend(await self._live_items_async(profile, progress=progress))

        if source_mode == "sample" or (source_mode == "hybrid" and len(items) < 5):
            if progress:
                progress(
                    {
                        "stage": "discovery",
                        "status": "running",
                        "message": "Loading sample content",
                        "source": self.sample_name,
                    }
                )
            sample_items = self._sample_items()
            items.extend(sample_items)
            if progress:
                progress(
                    {
                        "stage": "discovery",
                        "status": "complete",
                        "message": f"Loaded {len(sample_items)} sample items",
                        "source": self.sample_name,
                        "count": len(sample_items),
                    }
                )

        items = self._dedupe(items)
        self.store.save_content_items(items, "latest_candidates")
        preferred = {value.lower() for value in profile.preferred_content}
        if not preferred:
            return items
        return [item for item in items if item.content_type.value in preferred or item.content_type.value == "tool"]

    async def _live_items_async(
        self,
        profile: UserProfile,
        progress: ProgressCallback | None = None,
    ) -> list[ContentItem]:
        """Run all connectors concurrently using asyncio.to_thread."""

        async def _fetch_one(connector: ContentConnector) -> list[ContentItem]:
            if progress:
                progress(
                    {
                        "stage": "discovery",
                        "status": "running",
                        "message": f"Fetching {connector.source_name}",
                        "connector": connector.source_name,
                    }
                )
            try:
                fetched: list[ContentItem] = await asyncio.to_thread(connector.fetch, profile)
                if progress:
                    progress(
                        {
                            "stage": "discovery",
                            "status": "complete",
                            "message": f"{connector.source_name} returned {len(fetched)} items",
                            "connector": connector.source_name,
                            "count": len(fetched),
                        }
                    )
                if getattr(connector, "last_errors", None):
                    error_count = len(connector.last_errors)
                    self.last_errors.extend(
                        f"{connector.source_name}: {message}" for message in connector.last_errors
                    )
                    if progress:
                        progress(
                            {
                                "stage": "discovery",
                                "status": "warning",
                                "message": f"{connector.source_name} reported {error_count} fetch issue(s); details kept in backend artifacts",
                                "connector": connector.source_name,
                                "source_error_count": error_count,
                            }
                        )
                return fetched
            except ConnectorError as exc:
                self.last_errors.append(f"{connector.source_name}: {exc}")
                if progress:
                    progress(
                        {
                            "stage": "discovery",
                            "status": "warning",
                            "message": f"{connector.source_name} fetch was skipped; details kept in backend artifacts",
                            "connector": connector.source_name,
                            "source_error_count": 1,
                        }
                    )
                return []
            except Exception as exc:
                self.last_errors.append(f"{connector.source_name}: {type(exc).__name__}: {exc}")
                if progress:
                    progress(
                        {
                            "stage": "discovery",
                            "status": "warning",
                            "message": f"{connector.source_name} fetch failed; details kept in backend artifacts",
                            "connector": connector.source_name,
                            "source_error_count": 1,
                        }
                    )
                return []

        batches = await asyncio.gather(*[_fetch_one(conn) for conn in self.connectors])
        result: list[ContentItem] = []
        for batch in batches:
            result.extend(batch)
        return result

    def _dedupe(self, items: list[ContentItem]) -> list[ContentItem]:
        seen: set[str] = set()
        output: list[ContentItem] = []
        for item in items:
            key = item.url.lower().strip() or item.title.lower().strip()
            if key and key not in seen:
                seen.add(key)
                output.append(item)
        return output
