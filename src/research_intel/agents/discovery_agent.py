from __future__ import annotations

from research_intel.agents.base import BaseAgent
from research_intel.connectors import (
    ContentConnector,
    GitHubConnector,
    PapersWithCodeConnector,
    PaperSourceConnector,
    SemanticScholarConnector,
)
from research_intel.connectors.http_client import ConnectorError
from research_intel.models import ContentItem, UserProfile
from research_intel.storage import JsonStore


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
            PapersWithCodeConnector(),
            GitHubConnector(),
        ]
        self.last_errors: list[str] = []

    def discover(self, profile: UserProfile, source_mode: str = "hybrid") -> list[ContentItem]:
        source_mode = source_mode.lower().strip()
        if source_mode not in {"sample", "live", "hybrid"}:
            raise ValueError("source_mode must be one of: sample, live, hybrid")

        items: list[ContentItem] = []
        self.last_errors = []

        if source_mode in {"live", "hybrid"}:
            items.extend(self._live_items(profile))

        if source_mode == "sample" or (source_mode == "hybrid" and len(items) < 5):
            items.extend(self._sample_items())

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

    def _live_items(self, profile: UserProfile) -> list[ContentItem]:
        items: list[ContentItem] = []
        for connector in self.connectors:
            try:
                items.extend(connector.fetch(profile))
            except ConnectorError as exc:
                self.last_errors.append(f"{connector.source_name}: {exc}")
            except Exception as exc:
                self.last_errors.append(f"{connector.source_name}: {type(exc).__name__}: {exc}")
        return items

    def _dedupe(self, items: list[ContentItem]) -> list[ContentItem]:
        seen: set[str] = set()
        output: list[ContentItem] = []
        for item in items:
            key = item.url.lower().strip() or item.title.lower().strip()
            if key and key not in seen:
                seen.add(key)
                output.append(item)
        return output
