from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from research_intel.agents import (
    DiscoveryAgent,
    EvidenceAgent,
    FilteringAgent,
    ProfileAgent,
    RecommendationAgent,
    TrendAgent,
    ValueAnalysisAgent,
)
from research_intel.agents.base import AgentContext
from research_intel.models import DailyReport, FilterStatus
from research_intel.storage import JsonStore


@dataclass(slots=True)
class PipelineResult:
    report: DailyReport
    json_path: Path
    markdown_path: Path


class DailyResearchPipeline:
    def __init__(self, project_root: Path | str | None = None) -> None:
        self.store = JsonStore(project_root)
        self.profile_agent = ProfileAgent(self.store)
        self.discovery_agent = DiscoveryAgent(self.store)
        self.filtering_agent = FilteringAgent()
        self.value_agent = ValueAnalysisAgent()
        self.evidence_agent = EvidenceAgent()
        self.trend_agent = TrendAgent()
        self.recommendation_agent = RecommendationAgent()

    def run(
        self,
        profile_id: str = "default_user",
        report_stem: str = "latest",
        source_mode: str = "hybrid",
    ) -> PipelineResult:
        context = AgentContext(profile_id=profile_id, run_id=uuid4().hex[:8])
        profile = self.profile_agent.load_or_create(context.profile_id)
        discovered = self.discovery_agent.discover(profile, source_mode=source_mode)
        decisions = self.filtering_agent.filter(profile, discovered)
        selected_ids = {
            decision.item_id
            for decision in decisions
            if decision.status in {FilterStatus.CANDIDATE, FilterStatus.HIGH_PRIORITY}
        }
        selected_items = [item for item in discovered if item.item_id in selected_ids]
        analyses = self.value_agent.analyze(selected_items, decisions, profile=profile)
        analyses = self.evidence_agent.review(analyses)
        trends = self.trend_agent.analyze(profile, discovered, analyses)
        report = self.recommendation_agent.build_report(
            profile_id=profile.user_id,
            analyses=analyses,
            trends=trends,
            filter_stats=self.filtering_agent.stats(decisions),
            filter_decisions=decisions,
            candidates=discovered,
            source_mode=source_mode,
            candidate_count=len(discovered),
            source_errors=[*self.discovery_agent.last_errors, *self.value_agent.last_llm_errors],
        )
        self.store.save_run_json("latest_decisions", decisions)
        self.store.save_run_json("latest_analyses", analyses)
        json_path, markdown_path = self.store.save_report(report, stem=report_stem)
        return PipelineResult(report=report, json_path=json_path, markdown_path=markdown_path)
