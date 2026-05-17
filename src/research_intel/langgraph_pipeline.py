"""LangGraph-based daily research pipeline.

Replaces the linear DailyResearchPipeline with a StateGraph that supports:
- Parallel connector fetching (via DiscoveryAgent.discover_async)
- Parallel LLM enhancement (via ValueAnalysisAgent.analyze_async)
- Conditional routing: fallback to sample on live failure, relax filter on sparse results
- Supervisor node: dynamically adjusts LLM limit and skips unnecessary tools
- Supervisor sits between filter and analysis, inspecting results before choosing strategy

Activate via USE_LANGGRAPH_PIPELINE=true in .env, or --use-langgraph CLI flag.
"""
from __future__ import annotations

import asyncio
import operator
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, StateGraph

from research_intel.agents import (
    DiscoveryAgent,
    EvidenceAgent,
    FilteringAgent,
    ProfileAgent,
    RecommendationAgent,
    TrendAgent,
    ValueAnalysisAgent,
)
from research_intel.models import DailyReport, FilterStatus, to_plain_dict
from research_intel.rag import RagIndex, sync_pgvector_from_env
from research_intel.storage import JsonStore

ProgressCallback = Callable[[dict[str, object]], None]


class PipelineState(TypedDict):
    # ── Config (read-only during execution) ─────────────────────────────
    profile_id: str
    source_mode: str
    report_stem: str
    run_id: str

    # ── Accumulated data (Python objects, local execution only) ──────────
    profile: Any  # UserProfile | None
    candidates: list[Any]  # list[ContentItem]
    filter_decisions: list[Any]  # list[FilterDecision]
    selected_items: list[Any]  # list[ContentItem]
    analyses: list[Any]  # list[ValueAnalysis]
    trends: list[Any]  # list[TrendInsight]
    report: Any  # DailyReport | None

    # ── Control ──────────────────────────────────────────────────────────
    # errors uses the `operator.add` reducer so each node can append without
    # overwriting errors from previous nodes.
    errors: Annotated[list[str], operator.add]
    run_metadata: dict[str, Any]
    retry_count: int


@dataclass(slots=True)
class PipelineResult:
    report: DailyReport
    json_path: Path
    markdown_path: Path


class LangGraphDailyPipeline:
    """Research pipeline implemented as a LangGraph StateGraph.

    Exposes the same ``run()`` interface as ``DailyResearchPipeline`` so it can
    be used as a drop-in replacement.
    """

    def __init__(self, project_root: Path | str | None = None) -> None:
        self.store = JsonStore(project_root)
        self.profile_agent = ProfileAgent(self.store)
        self.discovery_agent = DiscoveryAgent(self.store)
        self.filtering_agent = FilteringAgent()
        self.value_agent = ValueAnalysisAgent()
        self.evidence_agent = EvidenceAgent()
        self.trend_agent = TrendAgent()
        self.recommendation_agent = RecommendationAgent()

    # ── Public entry point ────────────────────────────────────────────────

    def run(
        self,
        profile_id: str = "default_user",
        report_stem: str = "latest",
        source_mode: str = "hybrid",
        progress: ProgressCallback | None = None,
    ) -> PipelineResult:
        """Execute the pipeline synchronously (wraps the async graph via asyncio)."""
        initial_state: PipelineState = {
            "profile_id": profile_id,
            "source_mode": source_mode,
            "report_stem": report_stem,
            "run_id": uuid4().hex[:8],
            "profile": None,
            "candidates": [],
            "filter_decisions": [],
            "selected_items": [],
            "analyses": [],
            "trends": [],
            "report": None,
            "errors": [],
            "run_metadata": {
                "llm_analysis_limit": None,
                "skip_repo_tools": False,
                "filter_relaxed": False,
            },
            "retry_count": 0,
        }

        compiled = self._build_graph(progress)
        final_state = _run_async(compiled.ainvoke(initial_state))

        report: DailyReport | None = final_state.get("report")
        if report is None:
            raise RuntimeError("LangGraph pipeline did not produce a report")

        json_path, markdown_path = self.store.save_report(report, stem=report_stem)
        return PipelineResult(report=report, json_path=json_path, markdown_path=markdown_path)

    # ── Graph construction ────────────────────────────────────────────────

    def _build_graph(self, progress: ProgressCallback | None):  # noqa: ANN202
        """Build and compile the StateGraph, capturing all agents via closure."""

        store = self.store
        profile_agent = self.profile_agent
        discovery_agent = self.discovery_agent
        filtering_agent = self.filtering_agent
        value_agent = self.value_agent
        evidence_agent = self.evidence_agent
        trend_agent = self.trend_agent
        recommendation_agent = self.recommendation_agent

        # ── helper ────────────────────────────────────────────────────────
        def _emit(stage: str, status: str, message: str, **extra: object) -> None:
            if progress:
                progress({"stage": stage, "status": status, "message": message, **extra})

        # ── nodes ─────────────────────────────────────────────────────────

        def load_profile_node(state: PipelineState) -> dict[str, Any]:
            _emit("profile", "running", "Loading research profile", run_id=state["run_id"])
            profile = profile_agent.load_or_create(state["profile_id"])
            _emit(
                "profile", "complete",
                f"Loaded profile {profile.display_name or profile.user_id}",
                run_id=state["run_id"], profile_id=profile.user_id,
            )
            return {"profile": profile}

        async def discover_candidates_node(state: PipelineState) -> dict[str, Any]:
            _emit(
                "discovery", "running",
                f"Discovering candidates from {state['source_mode']} sources",
                run_id=state["run_id"], source_mode=state["source_mode"],
            )
            profile = state["profile"]
            mode = state["source_mode"]

            if mode in {"live", "hybrid"}:
                candidates = await discovery_agent.discover_async(profile, source_mode=mode, progress=progress)
            else:
                candidates = discovery_agent.discover(profile, source_mode=mode, progress=progress)

            _emit(
                "discovery", "complete",
                f"Collected {len(candidates)} candidates",
                run_id=state["run_id"],
                candidate_count=len(candidates),
                source_error_count=len(discovery_agent.last_errors),
            )
            return {
                "candidates": candidates,
                "errors": [f"discovery: {e}" for e in discovery_agent.last_errors],
            }

        def retry_with_sample_node(state: PipelineState) -> dict[str, Any]:
            _emit("discovery", "warning",
                  "Live discovery returned nothing; falling back to sample data",
                  run_id=state["run_id"])
            candidates = discovery_agent.discover(state["profile"], source_mode="sample", progress=progress)
            return {
                "candidates": candidates,
                "source_mode": "sample",
                "retry_count": state["retry_count"] + 1,
            }

        def filter_candidates_node(state: PipelineState) -> dict[str, Any]:
            _emit("filtering", "running", "Filtering candidates", run_id=state["run_id"])
            decisions = filtering_agent.filter(state["profile"], state["candidates"])
            stats = filtering_agent.stats(decisions)
            _emit(
                "filtering", "complete", "Candidate filtering complete",
                run_id=state["run_id"], decision_count=len(decisions), filter_stats=stats,
            )
            selected_ids = {
                d.item_id for d in decisions
                if d.status in {FilterStatus.CANDIDATE, FilterStatus.HIGH_PRIORITY}
            }
            selected_items = [item for item in state["candidates"] if item.item_id in selected_ids]
            return {"filter_decisions": decisions, "selected_items": selected_items}

        def relax_filter_node(state: PipelineState) -> dict[str, Any]:
            _emit("filtering", "warning", "Too few candidates; relaxing thresholds to include LOW_PRIORITY",
                  run_id=state["run_id"])
            extended_ids = {
                d.item_id for d in state["filter_decisions"]
                if d.status in {FilterStatus.CANDIDATE, FilterStatus.HIGH_PRIORITY, FilterStatus.LOW_PRIORITY}
            }
            selected_items = [item for item in state["candidates"] if item.item_id in extended_ids]
            return {
                "selected_items": selected_items,
                "run_metadata": {**state["run_metadata"], "filter_relaxed": True},
                "retry_count": state["retry_count"] + 1,
            }

        def supervisor_node(state: PipelineState) -> dict[str, Any]:
            """Inspect filter results and choose analysis strategy."""
            decisions = state["filter_decisions"]
            high_count = sum(1 for d in decisions if d.status == FilterStatus.HIGH_PRIORITY)
            total_passed = len(state["selected_items"])
            content_types = {item.content_type.value for item in state["selected_items"]}

            meta = dict(state["run_metadata"])

            # Strategy 1: Many high-priority items → raise LLM analysis limit
            if high_count >= 10:
                new_limit = min(high_count, 20)
                meta["llm_analysis_limit"] = new_limit
                _emit(
                    "supervisor", "running",
                    f"Supervisor: {high_count} high-priority items; raising LLM limit to {new_limit}",
                    run_id=state["run_id"],
                )

            # Strategy 2: Only papers → skip repo-specific tool calls
            if content_types and "repo" not in content_types:
                meta["skip_repo_tools"] = True

            meta["total_passed"] = total_passed
            meta["needs_relaxed_filter"] = total_passed < 3
            meta["high_priority_count"] = high_count

            return {"run_metadata": meta}

        async def analyze_value_node(state: PipelineState) -> dict[str, Any]:
            _emit(
                "value_analysis", "running",
                f"Analyzing {len(state['selected_items'])} selected items",
                run_id=state["run_id"], selected_count=len(state["selected_items"]),
            )
            # Apply LLM limit override from supervisor
            llm_limit = state["run_metadata"].get("llm_analysis_limit")
            if llm_limit is not None:
                value_agent.llm_limit = int(llm_limit)

            # Skip repo tools if supervisor flagged it
            if state["run_metadata"].get("skip_repo_tools"):
                value_agent._skip_repo_tools = True  # noqa: SLF001

            analyses = await value_agent.analyze_async(
                state["selected_items"],
                state["filter_decisions"],
                profile=state["profile"],
                progress=progress,
            )
            _emit(
                "value_analysis", "complete",
                f"Built {len(analyses)} value analyses",
                run_id=state["run_id"],
                analysis_count=len(analyses),
                llm_error_count=len(value_agent.last_llm_errors),
            )
            return {
                "analyses": analyses,
                "errors": [f"llm: {e}" for e in value_agent.last_llm_errors],
            }

        def check_evidence_node(state: PipelineState) -> dict[str, Any]:
            _emit("evidence", "running", "Reviewing evidence", run_id=state["run_id"])
            analyses = evidence_agent.review(state["analyses"])
            _emit("evidence", "complete", "Evidence review complete", run_id=state["run_id"])
            return {"analyses": analyses}

        def detect_trends_node(state: PipelineState) -> dict[str, Any]:
            _emit("trends", "running", "Analyzing trend signals", run_id=state["run_id"])
            trends = trend_agent.analyze(state["profile"], state["candidates"], state["analyses"])
            _emit("trends", "complete", f"Found {len(trends)} trend signals",
                  run_id=state["run_id"], trend_count=len(trends))
            return {"trends": trends}

        def build_report_node(state: PipelineState) -> dict[str, Any]:
            _emit("recommendation", "running", "Building daily report", run_id=state["run_id"])
            stats = filtering_agent.stats(state["filter_decisions"]) if state["filter_decisions"] else {}
            report = recommendation_agent.build_report(
                profile_id=state["profile_id"],
                analyses=state["analyses"],
                trends=state["trends"],
                filter_stats=stats,
                filter_decisions=state["filter_decisions"],
                candidates=state["candidates"],
                source_mode=state["source_mode"],
                candidate_count=len(state["candidates"]),
                source_errors=list(state["errors"]),
            )
            _emit("recommendation", "complete", "Report recommendations ready", run_id=state["run_id"])
            return {"report": report}

        def index_rag_node(state: PipelineState) -> dict[str, Any]:
            _emit("storage", "running", "Saving run artifacts", run_id=state["run_id"])
            store.save_run_json("latest_decisions", state["filter_decisions"])
            store.save_run_json("latest_analyses", state["analyses"])

            report = state["report"]
            rag_index = RagIndex.from_report(to_plain_dict(report), to_plain_dict(state["candidates"]))
            rag_index.save(store.runs_dir / "latest_rag_index.json")

            _emit("rag", "running", "Syncing RAG index to pgvector", run_id=state["run_id"])
            pgvector_errors: list[str] = []
            try:
                sync_pgvector_from_env(rag_index)
                _emit("rag", "complete", "RAG index sync complete", run_id=state["run_id"])
            except Exception as exc:
                pgvector_errors.append(f"pgvector: {type(exc).__name__}: {exc}")
                _emit("rag", "warning", "pgvector sync skipped or failed; details kept in backend artifacts",
                      run_id=state["run_id"])

            _emit("storage", "complete", "Saved run artifacts", run_id=state["run_id"])
            return {"errors": pgvector_errors}

        # ── routing functions ─────────────────────────────────────────────

        def route_after_discovery(state: PipelineState) -> str:
            if not state["candidates"] and state["source_mode"] == "live" and state["retry_count"] < 1:
                return "retry_with_sample"
            return "filter_candidates"

        def route_after_filter(state: PipelineState) -> str:
            if len(state["selected_items"]) < 3 and state["retry_count"] < 1:
                return "relax_filter"
            return "supervisor"

        # ── build graph ───────────────────────────────────────────────────

        graph: StateGraph = StateGraph(PipelineState)

        graph.add_node("load_profile", load_profile_node)
        graph.add_node("discover_candidates", discover_candidates_node)
        graph.add_node("retry_with_sample", retry_with_sample_node)
        graph.add_node("filter_candidates", filter_candidates_node)
        graph.add_node("relax_filter", relax_filter_node)
        graph.add_node("supervisor", supervisor_node)
        graph.add_node("analyze_value", analyze_value_node)
        graph.add_node("check_evidence", check_evidence_node)
        graph.add_node("detect_trends", detect_trends_node)
        graph.add_node("build_report", build_report_node)
        graph.add_node("index_rag", index_rag_node)

        graph.set_entry_point("load_profile")
        graph.add_edge("load_profile", "discover_candidates")
        graph.add_conditional_edges(
            "discover_candidates",
            route_after_discovery,
            {"retry_with_sample": "retry_with_sample", "filter_candidates": "filter_candidates"},
        )
        graph.add_edge("retry_with_sample", "filter_candidates")
        graph.add_conditional_edges(
            "filter_candidates",
            route_after_filter,
            {"relax_filter": "relax_filter", "supervisor": "supervisor"},
        )
        graph.add_edge("relax_filter", "supervisor")
        graph.add_edge("supervisor", "analyze_value")
        graph.add_edge("analyze_value", "check_evidence")
        graph.add_edge("check_evidence", "detect_trends")
        graph.add_edge("detect_trends", "build_report")
        graph.add_edge("build_report", "index_rag")
        graph.add_edge("index_rag", END)

        return graph.compile()


# ── asyncio helpers ───────────────────────────────────────────────────────────

def _run_async(coro: Any) -> Any:
    """Run a coroutine from synchronous code, compatible with ThreadingHTTPServer."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Already inside an event loop (e.g., Jupyter) – run in a new thread
            import concurrent.futures

            def _in_thread() -> Any:
                return asyncio.run(coro)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(_in_thread).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)
