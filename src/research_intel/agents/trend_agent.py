from __future__ import annotations

from collections import Counter, defaultdict
from datetime import UTC, datetime

from research_intel.agents.base import BaseAgent
from research_intel.models import Confidence, ContentItem, TrendInsight, UserProfile, ValueAnalysis


class TrendAgent(BaseAgent):
    name = "trend-agent"

    def analyze(
        self,
        profile: UserProfile,
        items: list[ContentItem],
        analyses: list[ValueAnalysis],
    ) -> list[TrendInsight]:
        analysis_by_id = {analysis.item_id: analysis for analysis in analyses}
        insights: list[TrendInsight] = []
        for window in (7, 30, 90):
            insights.extend(self._window_insights(profile, items, analysis_by_id, window))
        return insights[:5]

    def _window_insights(
        self,
        profile: UserProfile,
        items: list[ContentItem],
        analysis_by_id: dict[str, ValueAnalysis],
        window_days: int,
    ) -> list[TrendInsight]:
        recent_items = [item for item in items if self._age_days(item) <= window_days]
        if not recent_items:
            return []

        topic_counts: Counter[str] = Counter()
        topic_scores: defaultdict[str, list[float]] = defaultdict(list)
        topic_sources: defaultdict[str, set[str]] = defaultdict(set)

        profile_terms = set(profile.keywords())
        for item in recent_items:
            for tag in item.tags:
                normalized = tag.lower()
                if normalized in profile_terms or any(term in normalized or normalized in term for term in profile_terms):
                    topic_counts[normalized] += 1
                    topic_sources[normalized].add(item.source)
                    if item.item_id in analysis_by_id:
                        topic_scores[normalized].append(analysis_by_id[item.item_id].trend_signal)

        if not topic_counts:
            for item in recent_items:
                for tag in item.tags[:2]:
                    topic_counts[tag.lower()] += 1
                    topic_sources[tag.lower()].add(item.source)
                    if item.item_id in analysis_by_id:
                        topic_scores[tag.lower()].append(analysis_by_id[item.item_id].trend_signal)

        insights: list[TrendInsight] = []
        for topic, count in topic_counts.most_common(2):
            avg_signal = sum(topic_scores[topic]) / max(1, len(topic_scores[topic]))
            sources = sorted(topic_sources[topic])
            confidence = Confidence.HIGH if count >= 3 and len(sources) >= 2 else Confidence.MEDIUM
            if avg_signal < 5:
                confidence = Confidence.LOW

            summary = (
                f"{topic} appeared in {count} candidate item(s) within {window_days} days, "
                f"with average trend signal {avg_signal:.1f}/10."
            )
            signals = [
                f"sources: {', '.join(sources)}",
                f"candidate_count={count}",
                f"avg_trend_signal={avg_signal:.1f}",
            ]
            implication = self._implication(topic, window_days, avg_signal)
            insights.append(
                TrendInsight(
                    topic=topic,
                    window_days=window_days,
                    summary=summary,
                    signals=signals,
                    user_implication=implication,
                    confidence=confidence,
                )
            )

        return insights

    def _age_days(self, item: ContentItem) -> int:
        value = item.published_at or item.discovered_at
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return 9999
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return max(0, (datetime.now(UTC) - parsed).days)

    def _implication(self, topic: str, window_days: int, avg_signal: float) -> str:
        if avg_signal >= 7:
            return f"Treat {topic} as an active research opportunity; look for evaluation gaps and baseline shortages."
        if window_days <= 7:
            return f"Track {topic} for another week before committing deep research time."
        return f"Use {topic} as background context unless stronger paper-code evidence appears."

