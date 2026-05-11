from __future__ import annotations

from collections.abc import Callable
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
from research_intel.models import to_plain_dict
from research_intel.rag import RagIndex, sync_pgvector_from_env
from research_intel.storage import JsonStore

ProgressCallback = Callable[[dict[str, object]], None]


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
        progress: ProgressCallback | None = None,
    ) -> PipelineResult:
        context = AgentContext(profile_id=profile_id, run_id=uuid4().hex[:8])
        self._emit(progress, "profile", "running", "Loading research profile", run_id=context.run_id)
        profile = self.profile_agent.load_or_create(context.profile_id)
        self._emit(
            progress,
            "profile",
            "complete",
            f"Loaded profile {profile.display_name or profile.user_id}",
            run_id=context.run_id,
            profile_id=profile.user_id,
        )
        self._emit(
            progress,
            "discovery",
            "running",
            f"Discovering candidates from {source_mode} sources",
            run_id=context.run_id,
            source_mode=source_mode,
        )
        discovered = self.discovery_agent.discover(profile, source_mode=source_mode, progress=progress)
        self._emit(
            progress,
            "discovery",
            "complete",
            f"Collected {len(discovered)} candidates",
            run_id=context.run_id,
            candidate_count=len(discovered),
            source_error_count=len(self.discovery_agent.last_errors),
        )
        self._emit(progress, "filtering", "running", "Filtering candidates", run_id=context.run_id)
        decisions = self.filtering_agent.filter(profile, discovered)
        filter_stats = self.filtering_agent.stats(decisions)
        self._emit(
            progress,
            "filtering",
            "complete",
            "Candidate filtering complete",
            run_id=context.run_id,
            decision_count=len(decisions),
            filter_stats=filter_stats,
        )
        selected_ids = {
            decision.item_id
            for decision in decisions
            if decision.status in {FilterStatus.CANDIDATE, FilterStatus.HIGH_PRIORITY}
        }
        selected_items = [item for item in discovered if item.item_id in selected_ids]
        self._emit(
            progress,
            "value_analysis",
            "running",
            f"Analyzing {len(selected_items)} selected items",
            run_id=context.run_id,
            selected_count=len(selected_items),
        )
        analyses = self.value_agent.analyze(selected_items, decisions, profile=profile, progress=progress)
        self._emit(
            progress,
            "value_analysis",
            "complete",
            f"Built {len(analyses)} value analyses",
            run_id=context.run_id,
            analysis_count=len(analyses),
            llm_error_count=len(self.value_agent.last_llm_errors),
        )
        self._emit(progress, "evidence", "running", "Reviewing evidence", run_id=context.run_id)
        analyses = self.evidence_agent.review(analyses)
        self._emit(progress, "evidence", "complete", "Evidence review complete", run_id=context.run_id)
        self._emit(progress, "trends", "running", "Analyzing trend signals", run_id=context.run_id)
        trends = self.trend_agent.analyze(profile, discovered, analyses)
        self._emit(
            progress,
            "trends",
            "complete",
            f"Found {len(trends)} trend signals",
            run_id=context.run_id,
            trend_count=len(trends),
        )
        pgvector_errors: list[str] = []
        self._emit(progress, "recommendation", "running", "Building daily report", run_id=context.run_id)
        report = self.recommendation_agent.build_report(
            profile_id=profile.user_id,
            analyses=analyses,
            trends=trends,
            filter_stats=filter_stats,
            filter_decisions=decisions,
            candidates=discovered,
            source_mode=source_mode,
            candidate_count=len(discovered),
            source_errors=[*self.discovery_agent.last_errors, *self.value_agent.last_llm_errors],
        )
        self._emit(progress, "recommendation", "complete", "Report recommendations ready", run_id=context.run_id)
        self._emit(progress, "storage", "running", "Saving run artifacts", run_id=context.run_id)
        self.store.save_run_json("latest_decisions", decisions)
        self.store.save_run_json("latest_analyses", analyses)
        rag_index = RagIndex.from_report(to_plain_dict(report), to_plain_dict(discovered))
        rag_index.save(self.store.runs_dir / "latest_rag_index.json")
        self._emit(progress, "rag", "running", "Syncing RAG index to pgvector", run_id=context.run_id)
        try:
            sync_pgvector_from_env(rag_index)
            self._emit(progress, "rag", "complete", "RAG index sync complete", run_id=context.run_id)
        except Exception as exc:
            pgvector_errors.append(f"pgvector: {type(exc).__name__}: {exc}")
            self._emit(
                progress,
                "rag",
                "warning",
                "pgvector sync skipped or failed; details kept in backend artifacts",
                run_id=context.run_id,
            )
        if pgvector_errors:
            report.source_errors.extend(pgvector_errors)
        json_path, markdown_path = self.store.save_report(report, stem=report_stem)
        self._emit(
            progress,
            "storage",
            "complete",
            "Saved report artifacts",
            run_id=context.run_id,
            json_path=str(json_path),
            markdown_path=str(markdown_path),
        )
        return PipelineResult(report=report, json_path=json_path, markdown_path=markdown_path)

    def _emit(
        self,
        progress: ProgressCallback | None,
        stage: str,
        status: str,
        message: str,
        **extra: object,
    ) -> None:
        if progress is None:
            return
        progress({"stage": stage, "status": status, "message": message, **extra})
