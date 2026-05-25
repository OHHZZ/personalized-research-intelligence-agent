from __future__ import annotations

from dataclasses import dataclass

from research_intel.models import Confidence, ContentItem, ContentType, FilterDecision
from research_intel.scoring import clamp


@dataclass(slots=True)
class ScoreComponents:
    relevance: float
    novelty: float
    technical_depth: float
    evidence_strength: float
    reproducibility: float
    practical_utility: float
    trend_signal: float
    research_opportunity: float
    total: float
    confidence: Confidence


class ValueScorer:
    """Computes all numerical value dimensions and a weighted composite score."""

    WEIGHTS: dict[str, float] = {
        "relevance": 0.18,
        "novelty": 0.12,
        "technical_depth": 0.16,
        "evidence_strength": 0.14,
        "reproducibility": 0.12,
        "practical_utility": 0.12,
        "trend_signal": 0.08,
        "research_opportunity": 0.08,
    }

    def compute(self, item: ContentItem, decision: FilterDecision) -> ScoreComponents:
        relevance = decision.relevance_score
        quality = decision.quality_score

        novelty = self._novelty(item)
        technical_depth = self._technical_depth(item, quality)
        evidence_strength = self._evidence_strength(item)
        reproducibility = self._reproducibility(item)
        utility = self._utility(item)
        trend_signal = self._trend_signal(item)
        opportunity = self._research_opportunity(item, relevance, trend_signal)

        total = (
            relevance * self.WEIGHTS["relevance"]
            + novelty * self.WEIGHTS["novelty"]
            + technical_depth * self.WEIGHTS["technical_depth"]
            + evidence_strength * self.WEIGHTS["evidence_strength"]
            + reproducibility * self.WEIGHTS["reproducibility"]
            + utility * self.WEIGHTS["practical_utility"]
            + trend_signal * self.WEIGHTS["trend_signal"]
            + opportunity * self.WEIGHTS["research_opportunity"]
        )

        confidence = (
            Confidence.HIGH if evidence_strength >= 7 and reproducibility >= 6 else Confidence.MEDIUM
        )
        if evidence_strength < 4 or technical_depth < 4:
            confidence = Confidence.LOW

        return ScoreComponents(
            relevance=round(relevance, 2),
            novelty=round(novelty, 2),
            technical_depth=round(technical_depth, 2),
            evidence_strength=round(evidence_strength, 2),
            reproducibility=round(reproducibility, 2),
            practical_utility=round(utility, 2),
            trend_signal=round(trend_signal, 2),
            research_opportunity=round(opportunity, 2),
            total=round(clamp(total), 2),
            confidence=confidence,
        )

    # ── Dimension methods ─────────────────────────────────────────────────

    def _novelty(self, item: ContentItem) -> float:
        value = float(item.technical_signals.get("novelty", 5.0))
        if item.technical_signals.get("incremental"):
            value -= 1.5
        return clamp(value)

    def _technical_depth(self, item: ContentItem, quality: float) -> float:
        depth = item.technical_signals.get("technical_depth")
        if depth == "high":
            return clamp(max(quality, 7.5))
        if depth == "medium":
            return clamp(max(quality, 5.5))
        if depth == "low":
            return clamp(min(quality, 3.5))
        return clamp(quality)

    def _evidence_strength(self, item: ContentItem) -> float:
        s = item.technical_signals
        score = 3.0
        for key, weight in {
            "has_experiments": 1.5,
            "has_ablation": 1.0,
            "has_strong_baselines": 1.1,
            "has_benchmark": 0.8,
            "has_public_eval": 1.1,
            "cites_sources": 0.7,
            "has_metrics": 1.0,
            "has_leaderboard": 0.7,
        }.items():
            if s.get(key):
                score += weight
        if s.get("marketing_only"):
            score -= 2.0
        return clamp(score)

    def _reproducibility(self, item: ContentItem) -> float:
        s = item.technical_signals
        score = 2.5
        for key, weight in {
            "has_code": 1.2,
            "has_examples": 1.0,
            "has_tests": 0.7,
            "has_dataset": 0.9,
            "has_license": 0.5,
            "has_api_or_sdk": 0.8,
        }.items():
            if s.get(key):
                score += weight
        if s.get("readme_quality") == "thin":
            score -= 1.2
        return clamp(score)

    def _utility(self, item: ContentItem) -> float:
        s = item.technical_signals
        score = 4.0
        if item.content_type in {ContentType.REPO, ContentType.BENCHMARK}:
            score += 1.0
        if s.get("supports_research_workflow"):
            score += 1.1
        if s.get("baseline_ready"):
            score += 1.2
        if s.get("has_examples"):
            score += 0.7
        if s.get("commercial_only"):
            score -= 0.7
        return clamp(score)

    def _trend_signal(self, item: ContentItem) -> float:
        score = float(item.technical_signals.get("trend_signal", 4.5))
        if float(item.metrics.get("stars_30d", 0)) > 150:
            score += 1.0
        if item.technical_signals.get("emerging_topic"):
            score += 1.0
        return clamp(score)

    def _research_opportunity(self, item: ContentItem, relevance: float, trend_signal: float) -> float:
        score = relevance * 0.45 + trend_signal * 0.35
        if item.technical_signals.get("has_known_gap"):
            score += 1.5
        if item.technical_signals.get("benchmark_gap"):
            score += 1.2
        return clamp(score)
