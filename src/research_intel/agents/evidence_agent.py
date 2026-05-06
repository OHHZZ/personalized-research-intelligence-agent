from __future__ import annotations

from dataclasses import replace

from research_intel.agents.base import BaseAgent
from research_intel.models import Confidence, ContentType, ValueAnalysis


class EvidenceAgent(BaseAgent):
    name = "evidence-agent"

    def review(self, analyses: list[ValueAnalysis]) -> list[ValueAnalysis]:
        return [self._review_one(analysis) for analysis in analyses]

    def _review_one(self, analysis: ValueAnalysis) -> ValueAnalysis:
        issues: list[str] = []

        if analysis.content_type == ContentType.PAPER and analysis.evidence_strength < 5:
            issues.append("evidence check: paper-level evidence is weak")
        if analysis.content_type == ContentType.REPO and analysis.reproducibility < 5:
            issues.append("evidence check: reproducibility signals are not strong enough")
        if analysis.technical_depth < 4:
            issues.append("evidence check: technical depth is likely shallow")

        if not issues:
            return analysis

        confidence = Confidence.LOW if len(issues) >= 2 else Confidence.MEDIUM
        return replace(
            analysis,
            confidence=confidence,
            evidence=[*analysis.evidence, *issues],
            limitations=[*analysis.limitations, *issues],
        )

