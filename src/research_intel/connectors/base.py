from __future__ import annotations

from abc import ABC, abstractmethod

from research_intel.models import ContentItem, UserProfile


class ContentConnector(ABC):
    """Connector contract for future real data sources."""

    source_name: str
    last_errors: list[str]

    def __init__(self) -> None:
        self.last_errors = []

    @abstractmethod
    def fetch(self, profile: UserProfile) -> list[ContentItem]:
        raise NotImplementedError
