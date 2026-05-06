from __future__ import annotations

from abc import ABC, abstractmethod

from research_intel.models import ContentItem, UserProfile


class ContentConnector(ABC):
    """Connector contract for future real data sources."""

    source_name: str

    @abstractmethod
    def fetch(self, profile: UserProfile) -> list[ContentItem]:
        raise NotImplementedError

