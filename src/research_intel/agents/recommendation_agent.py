from __future__ import annotations

from research_intel.agents.base import BaseAgent
from research_intel.models import (
    ContentItem,
    ContentType,
    DailyReport,
    FilterDecision,
    TrendInsight,
    ValueAnalysis,
    utc_now_iso,
)


class RecommendationAgent(BaseAgent):
    name = "recommendation-agent"

    def build_report(
        self,
        profile_id: str,
        analyses: list[ValueAnalysis],
        trends: list[TrendInsight],
        filter_stats: dict[str, int],
        filter_decisions: list[FilterDecision] | None = None,
        candidates: list[ContentItem] | None = None,
        source_mode: str = "unknown",
        candidate_count: int = 0,
        source_errors: list[str] | None = None,
    ) -> DailyReport:
        top_papers = self._top_by_type(analyses, ContentType.PAPER, limit=3)
        top_repos = self._top_by_type(analyses, ContentType.REPO, limit=2)
        top_tools = [
            item
            for item in analyses
            if item.content_type in {ContentType.TOOL, ContentType.BENCHMARK, ContentType.ARTICLE}
        ][:2]
        actions = self._actions(top_papers, top_repos, top_tools, trends)
        generated_at = utc_now_iso()
        report = DailyReport(
            profile_id=profile_id,
            generated_at=generated_at,
            top_papers=top_papers,
            top_repos=top_repos,
            top_tools=top_tools,
            trends=trends[:3],
            actions=actions,
            filter_stats=filter_stats,
            markdown="",
            filter_decisions=filter_decisions or [],
            candidates=candidates or [],
            source_mode=source_mode,
            candidate_count=candidate_count,
            source_errors=source_errors or [],
        )
        report.markdown = self.render_markdown(report)
        return report

    def render_markdown(self, report: DailyReport) -> str:
        lines: list[str] = [
            "# Personalized Research Intelligence Daily Brief",
            "",
            f"- Profile: `{report.profile_id}`",
            f"- Generated at: `{report.generated_at}`",
            f"- Source mode: `{report.source_mode}`",
            f"- Candidate count: `{report.candidate_count}`",
            f"- Filter stats: {report.filter_stats}",
            f"- Source errors: {len(report.source_errors)}",
            "",
            "## Top Papers",
        ]
        lines.extend(self._render_analysis_list(report.top_papers))
        lines.extend(["", "## GitHub / Open Source Projects"])
        lines.extend(self._render_analysis_list(report.top_repos))
        lines.extend(["", "## Tools / Benchmarks / Articles"])
        lines.extend(self._render_analysis_list(report.top_tools))
        lines.extend(["", "## Trend Signals"])
        if report.trends:
            for trend in report.trends:
                lines.extend(
                    [
                        f"### {trend.topic} ({trend.window_days}d, confidence: {trend.confidence.value})",
                        trend.summary,
                        f"Implication: {trend.user_implication}",
                        f"Signals: {'; '.join(trend.signals)}",
                        "",
                    ]
                )
        else:
            lines.append("No strong trend signal found in this run.")
        lines.extend(["", "## Recommended Actions"])
        for index, action in enumerate(report.actions, start=1):
            lines.append(f"{index}. {action}")
        lines.append("")
        return "\n".join(lines)

    def _render_analysis_list(self, analyses: list[ValueAnalysis]) -> list[str]:
        if not analyses:
            return ["No item selected for this section."]

        lines: list[str] = []
        for item in analyses:
            lines.extend(
                [
                    f"### {item.title}",
                    f"- Type: `{item.content_type.value}`",
                    f"- Score: `{item.score}/10`, confidence: `{item.confidence.value}`",
                    f"- URL: {item.url}",
                    f"- Why it matters: {item.why_it_matters}",
                    f"- Relation to you: {item.relation_to_user}",
                    f"- Technical core: {item.technical_core}",
                    f"- Strengths: {'; '.join(item.strengths)}",
                    f"- Limitations: {'; '.join(item.limitations)}",
                    f"- Suggested next step: {item.possible_actions[0]}",
                    "",
                ]
            )
        return lines

    def _top_by_type(
        self,
        analyses: list[ValueAnalysis],
        content_type: ContentType,
        limit: int,
    ) -> list[ValueAnalysis]:
        return [item for item in analyses if item.content_type == content_type][:limit]

    def _actions(
        self,
        papers: list[ValueAnalysis],
        repos: list[ValueAnalysis],
        tools: list[ValueAnalysis],
        trends: list[TrendInsight],
    ) -> list[str]:
        actions: list[str] = []
        if papers:
            actions.append(f"Deep-read `{papers[0].title}` with focus on evaluation and limitations.")
        if repos:
            actions.append(f"Run the minimal demo for `{repos[0].title}` and inspect its core modules.")
        if trends:
            actions.append(f"Track `{trends[0].topic}` for a week and record missing benchmarks or weak baselines.")
        if tools:
            actions.append(f"Test `{tools[0].title}` against one real research workflow before adopting it.")
        actions.append("Mark irrelevant or useful items after reading so the profile can be adjusted.")
        return actions[:5]
