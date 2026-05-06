from __future__ import annotations

import os
from dataclasses import replace

from research_intel.agents.base import BaseAgent
from research_intel.llm import LLMError, QwenChatClient
from research_intel.models import (
    Confidence,
    ContentItem,
    ContentType,
    FilterDecision,
    UserProfile,
    ValueAnalysis,
)
from research_intel.scoring import clamp


class ValueAnalysisAgent(BaseAgent):
    name = "value-analysis-agent"

    def __init__(self, llm_client: QwenChatClient | None = None) -> None:
        self.llm_client = llm_client or QwenChatClient()
        self.llm_limit = int(os.getenv("LLM_ANALYSIS_LIMIT", "10"))
        self.last_llm_errors: list[str] = []

    def analyze(
        self,
        items: list[ContentItem],
        decisions: list[FilterDecision],
        profile: UserProfile | None = None,
    ) -> list[ValueAnalysis]:
        decision_map = {decision.item_id: decision for decision in decisions}
        analyses = [
            self._analyze_item(item, decision_map[item.item_id])
            for item in items
            if item.item_id in decision_map
        ]
        analyses = sorted(analyses, key=lambda item: item.score, reverse=True)
        if self.llm_client.enabled:
            analyses = self._enhance_with_llm(analyses, items, profile)
        return sorted(analyses, key=lambda item: item.score, reverse=True)

    def _enhance_with_llm(
        self,
        analyses: list[ValueAnalysis],
        items: list[ContentItem],
        profile: UserProfile | None,
    ) -> list[ValueAnalysis]:
        item_map = {item.item_id: item for item in items}
        enhanced: list[ValueAnalysis] = []
        self.last_llm_errors = []
        for index, analysis in enumerate(analyses):
            if index >= self.llm_limit or analysis.item_id not in item_map:
                enhanced.append(analysis)
                continue
            try:
                payload = self.llm_client.analyze_item(profile, item_map[analysis.item_id], analysis)
                enhanced.append(self._merge_llm_payload(analysis, payload))
            except (LLMError, ValueError, TypeError, KeyError) as exc:
                self.last_llm_errors.append(f"{analysis.item_id}: {exc}")
                enhanced.append(analysis)
        return enhanced

    def _merge_llm_payload(self, base: ValueAnalysis, payload: dict[str, object]) -> ValueAnalysis:
        confidence_value = str(payload.get("confidence", base.confidence.value)).lower()
        confidence = Confidence(confidence_value) if confidence_value in {item.value for item in Confidence} else base.confidence
        evidence = [*base.evidence, *self._list(payload.get("evidence")), f"analysis_source=llm:{self.llm_client.model}"]
        return replace(
            base,
            score=round(clamp(float(payload.get("score", base.score))), 2),
            relevance=round(clamp(float(payload.get("relevance", base.relevance))), 2),
            novelty=round(clamp(float(payload.get("novelty", base.novelty))), 2),
            technical_depth=round(clamp(float(payload.get("technical_depth", base.technical_depth))), 2),
            evidence_strength=round(clamp(float(payload.get("evidence_strength", base.evidence_strength))), 2),
            reproducibility=round(clamp(float(payload.get("reproducibility", base.reproducibility))), 2),
            practical_utility=round(clamp(float(payload.get("practical_utility", base.practical_utility))), 2),
            trend_signal=round(clamp(float(payload.get("trend_signal", base.trend_signal))), 2),
            research_opportunity=round(clamp(float(payload.get("research_opportunity", base.research_opportunity))), 2),
            why_it_matters=str(payload.get("why_it_matters", base.why_it_matters)),
            relation_to_user=str(payload.get("relation_to_user", base.relation_to_user)),
            technical_core=str(payload.get("technical_core", base.technical_core)),
            strengths=self._list(payload.get("strengths")) or base.strengths,
            limitations=self._list(payload.get("limitations")) or base.limitations,
            possible_actions=self._list(payload.get("possible_actions")) or base.possible_actions,
            evidence=evidence or base.evidence,
            confidence=confidence,
        )

    def _list(self, value: object) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value if str(item).strip()]
        return []

    def _analyze_item(self, item: ContentItem, decision: FilterDecision) -> ValueAnalysis:
        signals = item.technical_signals
        relevance = decision.relevance_score
        quality = decision.quality_score

        novelty = self._novelty(item)
        technical_depth = self._technical_depth(item, quality)
        evidence_strength = self._evidence_strength(item)
        reproducibility = self._reproducibility(item)
        utility = self._utility(item)
        trend_signal = self._trend_signal(item)
        opportunity = self._research_opportunity(item, relevance, trend_signal)

        score = (
            relevance * 0.18
            + novelty * 0.12
            + technical_depth * 0.16
            + evidence_strength * 0.14
            + reproducibility * 0.12
            + utility * 0.12
            + trend_signal * 0.08
            + opportunity * 0.08
        )

        confidence = Confidence.HIGH if evidence_strength >= 7 and reproducibility >= 6 else Confidence.MEDIUM
        if evidence_strength < 4 or technical_depth < 4:
            confidence = Confidence.LOW

        return ValueAnalysis(
            item_id=item.item_id,
            title=item.title,
            content_type=item.content_type,
            url=item.url,
            score=round(clamp(score), 2),
            relevance=round(relevance, 2),
            novelty=round(novelty, 2),
            technical_depth=round(technical_depth, 2),
            evidence_strength=round(evidence_strength, 2),
            reproducibility=round(reproducibility, 2),
            practical_utility=round(utility, 2),
            trend_signal=round(trend_signal, 2),
            research_opportunity=round(opportunity, 2),
            why_it_matters=self._why_it_matters(item),
            relation_to_user=self._relation_to_user(item, relevance),
            technical_core=self._technical_core(item),
            strengths=self._strengths(item),
            limitations=self._limitations(item),
            possible_actions=self._actions(item),
            evidence=self._evidence(item, decision),
            confidence=confidence,
        )

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

    def _why_it_matters(self, item: ContentItem) -> str:
        if item.content_type == ContentType.PAPER:
            return "It may change how the field frames the task or evaluates progress, especially if its experiments are reusable."
        if item.content_type == ContentType.REPO:
            return "It is worth inspecting because implementation quality and runnable demos can turn an idea into a practical baseline."
        if item.content_type == ContentType.BENCHMARK:
            return "It can shape what the community optimizes for and expose under-measured weaknesses."
        if item.content_type == ContentType.TOOL:
            return "It is relevant as a capability signal for research workflows and product-level AI systems."
        return "It is useful if it contains grounded technical analysis rather than surface-level commentary."

    def _relation_to_user(self, item: ContentItem, relevance: float) -> str:
        tags = ", ".join(item.tags[:4]) or "the configured research profile"
        if relevance >= 7:
            return f"Strong match to your current interests through {tags}."
        if relevance >= 5:
            return f"Moderate match; the useful parts are likely around {tags}."
        return f"Weak match; keep it only as peripheral context around {tags}."

    def _technical_core(self, item: ContentItem) -> str:
        core = item.technical_signals.get("technical_core")
        if core:
            return str(core)
        return item.summary

    def _strengths(self, item: ContentItem) -> list[str]:
        s = item.technical_signals
        strengths: list[str] = []
        if s.get("has_experiments"):
            strengths.append("includes experimental validation")
        if s.get("has_strong_baselines"):
            strengths.append("compares against meaningful baselines")
        if s.get("has_code"):
            strengths.append("has code or implementation artifacts")
        if s.get("has_examples"):
            strengths.append("offers runnable examples or demos")
        if s.get("has_leaderboard"):
            strengths.append("provides a leaderboard or shared evaluation target")
        if s.get("baseline_ready"):
            strengths.append("looks suitable for baseline exploration")
        return strengths or ["contains at least some relevant technical signal"]

    def _limitations(self, item: ContentItem) -> list[str]:
        s = item.technical_signals
        limitations: list[str] = []
        if not s.get("has_code") and item.content_type == ContentType.PAPER:
            limitations.append("code is not clearly available yet")
        if not s.get("has_ablation") and item.content_type == ContentType.PAPER:
            limitations.append("ablation evidence may be limited")
        if s.get("readme_quality") == "thin":
            limitations.append("README does not explain implementation details enough")
        if s.get("last_commit_days", 0) and float(s.get("last_commit_days", 0)) > 180:
            limitations.append("repository activity appears stale")
        if s.get("commercial_only"):
            limitations.append("closed commercial access may limit reproducibility")
        if s.get("marketing_only"):
            limitations.append("technical evidence is too thin to trust the claims")
        return limitations or ["main risk is whether the results generalize beyond the reported setting"]

    def _actions(self, item: ContentItem) -> list[str]:
        if item.content_type == ContentType.PAPER:
            actions = ["Read the method and evaluation sections before reading the full paper."]
            if item.technical_signals.get("has_code"):
                actions.append("Check whether the released code can be reused as a baseline.")
            if item.technical_signals.get("benchmark_gap"):
                actions.append("Look for an evaluation gap that can become a short-term project.")
            return actions

        if item.content_type == ContentType.REPO:
            actions = ["Run the demo or minimal inference path first."]
            if item.technical_signals.get("baseline_ready"):
                actions.append("Compare its input/output contract with your baseline needs.")
            actions.append("Inspect core modules before integrating it into a project.")
            return actions

        if item.content_type == ContentType.BENCHMARK:
            return [
                "Review the metric definitions and failure cases.",
                "Check whether your research direction is underrepresented in the benchmark.",
            ]

        if item.content_type == ContentType.TOOL:
            return [
                "Test the tool on one real research workflow task.",
                "Record where the capability is strong, weak, or not reproducible.",
            ]

        return ["Skim for concrete technical claims and save only if it cites primary sources."]

    def _evidence(self, item: ContentItem, decision: FilterDecision) -> list[str]:
        evidence = [f"filter status: {decision.status.value}", f"relevance={decision.relevance_score}", f"quality={decision.quality_score}"]
        for key in (
            "has_experiments",
            "has_code",
            "has_examples",
            "has_tests",
            "has_leaderboard",
            "last_commit_days",
            "stars_30d",
        ):
            if key in item.technical_signals:
                evidence.append(f"{key}={item.technical_signals[key]}")
            if key in item.metrics:
                evidence.append(f"{key}={item.metrics[key]}")
        return evidence
