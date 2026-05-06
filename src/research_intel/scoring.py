from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from typing import Iterable

from research_intel.models import ContentItem, ContentType, UserProfile

TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9\-_/+.]*")


def clamp(value: float, lower: float = 0.0, upper: float = 10.0) -> float:
    return max(lower, min(upper, value))


def tokenize(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}


def phrase_score(phrases: Iterable[str], text: str) -> float:
    lowered = text.lower()
    score = 0.0
    for phrase in phrases:
        phrase = phrase.lower().strip()
        if not phrase:
            continue
        if phrase in lowered:
            score += 1.8 if " " in phrase else 1.0
    return score


def relevance_score(profile: UserProfile, item: ContentItem) -> float:
    profile_text = " ".join(profile.keywords())
    item_text = " ".join([item.title, item.summary, " ".join(item.tags)])

    profile_tokens = tokenize(profile_text)
    item_tokens = tokenize(item_text)
    overlap = len(profile_tokens & item_tokens)
    phrase_hits = phrase_score(profile.keywords(), item_text)

    domain_hits = phrase_score(profile.research_domains, item_text) * 1.4
    method_hits = phrase_score(profile.methods, item_text) * 1.1
    app_hits = phrase_score(profile.applications, item_text)

    excluded_hits = phrase_score(profile.excluded_topics, item_text)
    feedback_boost = 0.0
    for tag in item.tags:
        feedback_boost += profile.feedback_weights.get(tag.lower().strip(), 0.0)
    raw = overlap * 0.55 + phrase_hits + domain_hits + method_hits + app_hits + feedback_boost - excluded_hits * 2.0
    return round(clamp(raw), 2)


def quality_score(item: ContentItem) -> float:
    if item.content_type == ContentType.PAPER:
        return _paper_quality(item)
    if item.content_type == ContentType.REPO:
        return _repo_quality(item)
    if item.content_type == ContentType.BENCHMARK:
        return _benchmark_quality(item)
    if item.content_type == ContentType.TOOL:
        return _tool_quality(item)
    return _article_quality(item)


def recency_days(item: ContentItem) -> int | None:
    value = item.published_at or item.discovered_at
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return max(0, (datetime.now(UTC) - parsed).days)


def recency_bonus(item: ContentItem) -> float:
    days = recency_days(item)
    if days is None:
        return 0.0
    if days <= 7:
        return 1.0
    if days <= 30:
        return 0.7
    if days <= 90:
        return 0.35
    return -0.4


def logistic_signal(value: float, midpoint: float, steepness: float = 0.02) -> float:
    return 1.0 / (1.0 + math.exp(-steepness * (value - midpoint)))


def _paper_quality(item: ContentItem) -> float:
    s = item.technical_signals
    score = 2.0
    score += 1.4 if s.get("has_experiments") else -1.2
    score += 1.1 if s.get("has_ablation") else -0.2
    score += 1.0 if s.get("has_strong_baselines") else -0.5
    score += 0.9 if s.get("has_code") else 0.0
    score += 0.7 if s.get("has_benchmark") else 0.0
    score += min(1.2, float(s.get("baseline_count", 0)) * 0.18)
    score += recency_bonus(item)
    return round(clamp(score), 2)


def _repo_quality(item: ContentItem) -> float:
    s = item.technical_signals
    m = item.metrics
    score = 1.6
    score += 1.0 if s.get("has_readme") else -1.2
    score += 1.0 if s.get("has_examples") else -0.4
    score += 0.8 if s.get("has_tests") else -0.1
    score += 0.6 if s.get("has_license") else -0.3
    score += 0.8 if s.get("has_paper_link") else 0.0
    score += 1.1 if s.get("technical_depth") == "high" else 0.4 if s.get("technical_depth") == "medium" else -0.7
    score += logistic_signal(float(m.get("stars", 0)), midpoint=500, steepness=0.006) * 1.1
    score += logistic_signal(float(m.get("stars_30d", 0)), midpoint=80, steepness=0.03) * 1.2
    last_commit_days = float(s.get("last_commit_days", 999))
    score += 0.9 if last_commit_days <= 30 else 0.3 if last_commit_days <= 120 else -0.8
    if s.get("is_prompt_collection"):
        score -= 2.8
    if s.get("readme_quality") == "thin":
        score -= 1.2
    return round(clamp(score), 2)


def _benchmark_quality(item: ContentItem) -> float:
    s = item.technical_signals
    score = 2.5
    score += 1.4 if s.get("has_dataset") else -0.5
    score += 1.2 if s.get("has_metrics") else -0.7
    score += 1.0 if s.get("has_leaderboard") else 0.0
    score += 0.8 if s.get("has_code") else 0.0
    score += recency_bonus(item)
    return round(clamp(score), 2)


def _tool_quality(item: ContentItem) -> float:
    s = item.technical_signals
    score = 2.0
    score += 1.2 if s.get("has_public_eval") else -0.4
    score += 1.0 if s.get("has_api_or_sdk") else 0.0
    score += 0.8 if s.get("technical_report") else -0.2
    score += 0.7 if s.get("supports_research_workflow") else 0.0
    score += recency_bonus(item)
    if s.get("marketing_only"):
        score -= 2.0
    return round(clamp(score), 2)


def _article_quality(item: ContentItem) -> float:
    s = item.technical_signals
    score = 1.8
    score += 1.2 if s.get("has_technical_details") else -1.0
    score += 1.0 if s.get("cites_sources") else -0.4
    score += 0.8 if s.get("has_reproducible_steps") else 0.0
    score += recency_bonus(item)
    if s.get("marketing_only"):
        score -= 2.0
    return round(clamp(score), 2)
