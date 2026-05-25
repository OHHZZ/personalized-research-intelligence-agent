from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from dataclasses import replace

from research_intel.llm import LLMError, QwenChatClient
from research_intel.models import Confidence, ContentItem, UserProfile, ValueAnalysis
from research_intel.scoring import clamp

ProgressCallback = Callable[[dict[str, object]], None]

# Verbatim templates produced by InsightGenerator — LLM output that exactly
# matches one of these means the model returned the rule-based fallback.
_FALLBACK_WHY: frozenset[str] = frozenset({
    "It may change how the field frames the task or evaluates progress, especially if its experiments are reusable.",
    "It is worth inspecting because implementation quality and runnable demos can turn an idea into a practical baseline.",
    "It can shape what the community optimizes for and expose under-measured weaknesses.",
    "It is relevant as a capability signal for research workflows and product-level AI systems.",
    "It is useful if it contains grounded technical analysis rather than surface-level commentary.",
})


class LLMEnhancer:
    """Optionally enriches ValueAnalysis objects with LLM-generated insights.

    Applies to the top-N analyses (governed by `limit`). Runs with
    bounded concurrency (max 3 parallel calls) in the async variant.
    Both sync and async paths share the same merge and quality-check logic.
    """

    def __init__(self, llm_client: QwenChatClient, limit: int = 10) -> None:
        self.llm_client = llm_client
        self.limit = limit
        self.last_errors: list[str] = []

    # ── Public API ────────────────────────────────────────────────────────

    def enhance(
        self,
        analyses: list[ValueAnalysis],
        items: list[ContentItem],
        profile: UserProfile | None,
        progress: ProgressCallback | None = None,
    ) -> list[ValueAnalysis]:
        item_map = {item.item_id: item for item in items}
        self.last_errors = []
        enhanced: list[ValueAnalysis] = []

        for index, analysis in enumerate(analyses):
            if index >= self.limit or analysis.item_id not in item_map:
                enhanced.append(analysis)
                continue
            self._emit_start(progress, index, len(analyses), analysis)
            result = analysis
            for attempt in range(2):
                try:
                    payload = self.llm_client.analyze_item(profile, item_map[analysis.item_id], analysis)
                    candidate = self._merge_payload(analysis, payload)
                    if self._evaluate_quality(candidate) >= 0.5 or attempt == 1:
                        result = candidate
                        break
                except (LLMError, ValueError, TypeError, KeyError) as exc:
                    self.last_errors.append(f"{analysis.item_id}: {exc}")
                    self._emit_skip(progress, analysis)
                    break
            enhanced.append(result)
            if result is not analysis:
                self._emit_done(progress, analysis)
        return enhanced

    async def enhance_async(
        self,
        analyses: list[ValueAnalysis],
        items: list[ContentItem],
        profile: UserProfile | None,
        progress: ProgressCallback | None = None,
    ) -> list[ValueAnalysis]:
        item_map = {item.item_id: item for item in items}
        self.last_errors = []
        semaphore = asyncio.Semaphore(3)

        async def _one(index: int, analysis: ValueAnalysis) -> ValueAnalysis:
            if index >= self.limit or analysis.item_id not in item_map:
                return analysis
            self._emit_start(progress, index, len(analyses), analysis)
            async with semaphore:
                for attempt in range(2):
                    try:
                        payload = await asyncio.to_thread(
                            self.llm_client.analyze_item, profile, item_map[analysis.item_id], analysis
                        )
                        result = self._merge_payload(analysis, payload)
                        if self._evaluate_quality(result) >= 0.5 or attempt == 1:
                            self._emit_done(progress, analysis)
                            return result
                        await asyncio.sleep(0.5)
                    except (LLMError, ValueError, TypeError, KeyError) as exc:
                        self.last_errors.append(f"{analysis.item_id}: {exc}")
                        self._emit_skip(progress, analysis)
                        return analysis
            return analysis

        return list(await asyncio.gather(*[_one(i, a) for i, a in enumerate(analyses)]))

    # ── Quality evaluation ────────────────────────────────────────────────

    def _evaluate_quality(self, analysis: ValueAnalysis) -> float:
        """Score LLM output quality in [0, 1]. Pure rule check, no extra LLM call."""
        why = analysis.why_it_matters or ""
        if why in _FALLBACK_WHY:
            return 0.0
        score = 1.0
        if len(why) < 60:
            score -= 0.3
        elif not re.search(r"\d+[%x]?|\bSOTA\b|\bstate.of.the.art\b|benchmark|experiment|dataset", why, re.I):
            score -= 0.15
        if len(analysis.technical_core or "") < 50:
            score -= 0.2
        if len(analysis.strengths or []) < 2:
            score -= 0.2
        if not (analysis.possible_actions or []):
            score -= 0.1
        return max(0.0, score)

    # ── Merge ─────────────────────────────────────────────────────────────

    def _merge_payload(self, base: ValueAnalysis, payload: dict[str, object]) -> ValueAnalysis:
        confidence_value = str(payload.get("confidence", base.confidence.value)).lower()
        confidence = (
            Confidence(confidence_value)
            if confidence_value in {c.value for c in Confidence}
            else base.confidence
        )
        evidence = [
            *base.evidence,
            *self._as_list(payload.get("evidence")),
            f"analysis_source=llm:{self.llm_client.model}",
        ]
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
            strengths=self._as_list(payload.get("strengths")) or base.strengths,
            limitations=self._as_list(payload.get("limitations")) or base.limitations,
            possible_actions=self._as_list(payload.get("possible_actions")) or base.possible_actions,
            evidence=evidence or base.evidence,
            confidence=confidence,
        )

    def _as_list(self, value: object) -> list[str]:
        if isinstance(value, list):
            return [str(v) for v in value if str(v).strip()]
        return []

    # ── Progress helpers ──────────────────────────────────────────────────

    def _emit_start(
        self, progress: ProgressCallback | None, index: int, total: int, analysis: ValueAnalysis
    ) -> None:
        if progress:
            progress({
                "stage": "value_analysis",
                "status": "running",
                "message": f"LLM enhancement {index + 1}/{min(total, self.limit)}",
                "item_id": analysis.item_id,
                "title": analysis.title,
            })

    def _emit_done(self, progress: ProgressCallback | None, analysis: ValueAnalysis) -> None:
        if progress:
            progress({
                "stage": "value_analysis",
                "status": "complete",
                "message": f"Enhanced {analysis.title}",
                "item_id": analysis.item_id,
            })

    def _emit_skip(self, progress: ProgressCallback | None, analysis: ValueAnalysis) -> None:
        if progress:
            progress({
                "stage": "value_analysis",
                "status": "warning",
                "message": f"LLM enhancement skipped for {analysis.item_id}; details kept in backend artifacts",
                "item_id": analysis.item_id,
            })
