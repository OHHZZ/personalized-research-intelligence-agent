from __future__ import annotations

import asyncio
import os
from collections.abc import Callable

from research_intel.agents.base import BaseAgent
from research_intel.agents.insight_generator import InsightGenerator
from research_intel.agents.item_enricher import ItemEnricher
from research_intel.agents.llm_enhancer import LLMEnhancer
from research_intel.agents.value_scorer import ValueScorer
from research_intel.llm import QwenChatClient
from research_intel.models import (
    ContentItem,
    FilterDecision,
    UserProfile,
    ValueAnalysis,
)

ProgressCallback = Callable[[dict[str, object]], None]


class ValueAnalysisAgent(BaseAgent):
    """Orchestrates item enrichment, scoring, insight generation, and LLM enhancement.

    Delegates each responsibility to a focused sub-component:
      - ItemEnricher  — fetches missing abstracts / citations / repo metrics
      - ValueScorer   — computes 8 numerical dimensions + weighted composite
      - InsightGenerator — produces human-readable text fields (rule-based)
      - LLMEnhancer   — optionally upgrades text fields with an LLM call
    """

    name = "value-analysis-agent"

    def __init__(self, llm_client: QwenChatClient | None = None) -> None:
        self.llm_client = llm_client or QwenChatClient()
        self._enricher = ItemEnricher()
        self._scorer = ValueScorer()
        self._insight_gen = InsightGenerator()
        self._llm_enhancer = LLMEnhancer(
            self.llm_client,
            limit=int(os.getenv("LLM_ANALYSIS_LIMIT", "10")),
        )

    # llm_limit is read/written by the pipeline supervisor
    @property
    def llm_limit(self) -> int:
        return self._llm_enhancer.limit

    @llm_limit.setter
    def llm_limit(self, value: int) -> None:
        self._llm_enhancer.limit = int(value)

    @property
    def last_llm_errors(self) -> list[str]:
        return self._llm_enhancer.last_errors

    # ── Public API ────────────────────────────────────────────────────────

    async def analyze_async(
        self,
        items: list[ContentItem],
        decisions: list[FilterDecision],
        profile: UserProfile | None = None,
        progress: ProgressCallback | None = None,
    ) -> list[ValueAnalysis]:
        decision_map = {d.item_id: d for d in decisions}
        self._emit_start(progress, len(items))

        enriched = list(await asyncio.gather(
            *[self._enricher.enrich_async(item) for item in items if item.item_id in decision_map]
        ))

        analyses = self._build_sorted(enriched, decision_map)

        if self.llm_client.enabled:
            analyses = await self._llm_enhancer.enhance_async(analyses, enriched, profile, progress=progress)

        return sorted(analyses, key=lambda a: a.score, reverse=True)

    def analyze(
        self,
        items: list[ContentItem],
        decisions: list[FilterDecision],
        profile: UserProfile | None = None,
        progress: ProgressCallback | None = None,
    ) -> list[ValueAnalysis]:
        decision_map = {d.item_id: d for d in decisions}
        self._emit_start(progress, len(items))

        enriched = [self._enricher.enrich(item) for item in items if item.item_id in decision_map]
        analyses = self._build_sorted(enriched, decision_map)

        if self.llm_client.enabled:
            analyses = self._llm_enhancer.enhance(analyses, enriched, profile, progress=progress)

        return sorted(analyses, key=lambda a: a.score, reverse=True)

    # ── Internals ─────────────────────────────────────────────────────────

    def _build_sorted(
        self,
        items: list[ContentItem],
        decision_map: dict[str, FilterDecision],
    ) -> list[ValueAnalysis]:
        analyses = [
            self._build_analysis(item, decision_map[item.item_id])
            for item in items
            if item.item_id in decision_map
        ]
        return sorted(analyses, key=lambda a: a.score, reverse=True)

    def _build_analysis(self, item: ContentItem, decision: FilterDecision) -> ValueAnalysis:
        scores = self._scorer.compute(item, decision)
        insights = self._insight_gen.generate(item, scores.relevance, decision)
        return ValueAnalysis(
            item_id=item.item_id,
            title=item.title,
            content_type=item.content_type,
            url=item.url,
            score=scores.total,
            relevance=scores.relevance,
            novelty=scores.novelty,
            technical_depth=scores.technical_depth,
            evidence_strength=scores.evidence_strength,
            reproducibility=scores.reproducibility,
            practical_utility=scores.practical_utility,
            trend_signal=scores.trend_signal,
            research_opportunity=scores.research_opportunity,
            confidence=scores.confidence,
            **insights,
        )

    def _emit_start(self, progress: ProgressCallback | None, count: int) -> None:
        if progress:
            progress({
                "stage": "value_analysis",
                "status": "running",
                "message": f"Scoring {count} selected items",
                "count": count,
            })
