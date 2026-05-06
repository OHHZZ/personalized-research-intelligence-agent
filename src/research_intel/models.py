from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class ContentType(str, Enum):
    PAPER = "paper"
    REPO = "repo"
    ARTICLE = "article"
    TOOL = "tool"
    BENCHMARK = "benchmark"


class FilterStatus(str, Enum):
    REJECT = "reject"
    LOW_PRIORITY = "low_priority"
    CANDIDATE = "candidate"
    HIGH_PRIORITY = "high_priority"


class Confidence(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class UserProfile:
    user_id: str
    display_name: str
    research_domains: list[str]
    methods: list[str] = field(default_factory=list)
    applications: list[str] = field(default_factory=list)
    preferred_content: list[str] = field(default_factory=lambda: ["paper", "repo", "benchmark"])
    excluded_topics: list[str] = field(default_factory=list)
    technical_level: str = "researcher"
    current_goals: list[str] = field(default_factory=list)
    feedback_weights: dict[str, float] = field(default_factory=dict)
    updated_at: str = field(default_factory=utc_now_iso)

    def keywords(self) -> list[str]:
        terms: list[str] = []
        for group in (
            self.research_domains,
            self.methods,
            self.applications,
            self.current_goals,
        ):
            terms.extend(group)
        return unique_normalized(terms)


@dataclass(slots=True)
class ContentItem:
    item_id: str
    content_type: ContentType
    title: str
    url: str
    source: str
    summary: str
    tags: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
    published_at: str | None = None
    discovered_at: str = field(default_factory=utc_now_iso)
    metrics: dict[str, float] = field(default_factory=dict)
    technical_signals: dict[str, Any] = field(default_factory=dict)
    links: dict[str, str] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ContentItem":
        data = dict(payload)
        data["content_type"] = ContentType(data["content_type"])
        return cls(**data)


@dataclass(slots=True)
class FilterDecision:
    item_id: str
    status: FilterStatus
    relevance_score: float
    quality_score: float
    reasons: list[str] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ValueAnalysis:
    item_id: str
    title: str
    content_type: ContentType
    url: str
    score: float
    relevance: float
    novelty: float
    technical_depth: float
    evidence_strength: float
    reproducibility: float
    practical_utility: float
    trend_signal: float
    research_opportunity: float
    why_it_matters: str
    relation_to_user: str
    technical_core: str
    strengths: list[str]
    limitations: list[str]
    possible_actions: list[str]
    evidence: list[str]
    confidence: Confidence = Confidence.MEDIUM


@dataclass(slots=True)
class TrendInsight:
    topic: str
    window_days: int
    summary: str
    signals: list[str]
    user_implication: str
    confidence: Confidence = Confidence.MEDIUM


@dataclass(slots=True)
class DailyReport:
    profile_id: str
    generated_at: str
    top_papers: list[ValueAnalysis]
    top_repos: list[ValueAnalysis]
    top_tools: list[ValueAnalysis]
    trends: list[TrendInsight]
    actions: list[str]
    filter_stats: dict[str, int]
    markdown: str
    filter_decisions: list[FilterDecision] = field(default_factory=list)
    candidates: list[ContentItem] = field(default_factory=list)
    source_mode: str = "unknown"
    candidate_count: int = 0
    source_errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class FeedbackEvent:
    feedback_id: str
    profile_id: str
    item_id: str
    action: str
    note: str = ""
    created_at: str = field(default_factory=utc_now_iso)


def to_plain_dict(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_plain_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [to_plain_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: to_plain_dict(item) for key, item in value.items()}
    return value


def unique_normalized(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        normalized = " ".join(value.lower().strip().split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(normalized)
    return output
