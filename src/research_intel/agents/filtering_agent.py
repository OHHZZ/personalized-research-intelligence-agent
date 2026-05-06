from __future__ import annotations

from collections import Counter

from research_intel.agents.base import BaseAgent
from research_intel.models import ContentItem, FilterDecision, FilterStatus, UserProfile
from research_intel.scoring import quality_score, relevance_score


class FilteringAgent(BaseAgent):
    name = "filtering-agent"

    def filter(self, profile: UserProfile, items: list[ContentItem]) -> list[FilterDecision]:
        return [self._decide(profile, item) for item in items]

    def stats(self, decisions: list[FilterDecision]) -> dict[str, int]:
        counter = Counter(decision.status.value for decision in decisions)
        return {
            "reject": counter.get("reject", 0),
            "low_priority": counter.get("low_priority", 0),
            "candidate": counter.get("candidate", 0),
            "high_priority": counter.get("high_priority", 0),
        }

    def _decide(self, profile: UserProfile, item: ContentItem) -> FilterDecision:
        relevance = relevance_score(profile, item)
        quality = quality_score(item)
        reasons: list[str] = []

        text = " ".join([item.title, item.summary, " ".join(item.tags)]).lower()
        excluded_hits = [topic for topic in profile.excluded_topics if topic.lower() in text]
        if excluded_hits:
            reasons.append(f"matches excluded topic: {', '.join(excluded_hits)}")

        if relevance < 2.0:
            reasons.append("weak relation to the user profile")
        if quality < 3.0:
            reasons.append("low technical or evidence quality")

        if item.technical_signals.get("marketing_only"):
            reasons.append("marketing-heavy content without enough technical detail")
        if item.technical_signals.get("is_prompt_collection"):
            reasons.append("prompt collection rather than a technical contribution")
        if item.technical_signals.get("readme_quality") == "thin":
            reasons.append("README is too thin to evaluate implementation value")

        combined = relevance * 0.58 + quality * 0.42
        if excluded_hits or combined < 3.2:
            status = FilterStatus.REJECT
        elif combined < 4.8:
            status = FilterStatus.LOW_PRIORITY
        elif combined >= 7.0 and relevance >= 5.2 and quality >= 5.0:
            status = FilterStatus.HIGH_PRIORITY
        else:
            status = FilterStatus.CANDIDATE

        if not reasons:
            reasons.append("passes relevance and quality checks")

        return FilterDecision(
            item_id=item.item_id,
            status=status,
            relevance_score=relevance,
            quality_score=quality,
            reasons=reasons,
            signals={
                "combined_score": round(combined, 2),
                "content_type": item.content_type.value,
            },
        )

