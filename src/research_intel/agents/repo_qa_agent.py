from __future__ import annotations

from research_intel.agents.base import BaseAgent
from research_intel.models import ContentItem, ContentType


class RepoQAAgent(BaseAgent):
    name = "repo-qa-agent"

    def answer(self, repo: ContentItem, question: str) -> str:
        if repo.content_type != ContentType.REPO:
            return "This item is not a GitHub repository, so Repo QA cannot analyze it."

        q = question.lower()
        if any(term in q for term in ("run", "install", "运行", "安装", "demo")):
            return self._running_answer(repo)
        if any(term in q for term in ("baseline", "基线", "适合")):
            return self._baseline_answer(repo)
        if any(term in q for term in ("quality", "bug", "代码质量", "潜在")):
            return self._quality_answer(repo)
        if any(term in q for term in ("develop", "二次开发", "extend", "扩展")):
            return self._extension_answer(repo)
        return self._overview_answer(repo)

    def _running_answer(self, repo: ContentItem) -> str:
        has_examples = repo.technical_signals.get("has_examples")
        readme_quality = repo.technical_signals.get("readme_quality", "unknown")
        if has_examples:
            return (
                f"`{repo.title}` looks runnable enough for a first demo because examples are present. "
                "Start from README setup, then locate the smallest inference or demo script before touching training code."
            )
        return (
            f"`{repo.title}` does not expose strong runnable-example signals. "
            f"README quality is `{readme_quality}`, so treat setup cost as a risk."
        )

    def _baseline_answer(self, repo: ContentItem) -> str:
        if repo.technical_signals.get("baseline_ready"):
            return (
                f"`{repo.title}` is a plausible baseline candidate. "
                "Validate license, data requirements, and whether its input/output contract matches your experiment."
            )
        return (
            f"`{repo.title}` is not clearly baseline-ready yet. "
            "Use it for idea inspection unless the core inference path is easy to reproduce."
        )

    def _quality_answer(self, repo: ContentItem) -> str:
        signals = repo.technical_signals
        risks: list[str] = []
        if not signals.get("has_tests"):
            risks.append("limited test signal")
        if signals.get("last_commit_days", 999) > 180:
            risks.append("stale commit activity")
        if signals.get("readme_quality") == "thin":
            risks.append("thin README")
        if not signals.get("has_license"):
            risks.append("license unclear")
        risk_text = "; ".join(risks) if risks else "no major metadata-level risk found"
        return f"Metadata-level quality check for `{repo.title}`: {risk_text}."

    def _extension_answer(self, repo: ContentItem) -> str:
        core = repo.technical_signals.get("technical_core", repo.summary)
        return (
            f"For second-stage development on `{repo.title}`, first isolate the core path: {core} "
            "Then add your own adapter or evaluation wrapper instead of modifying model internals immediately."
        )

    def _overview_answer(self, repo: ContentItem) -> str:
        return (
            f"`{repo.title}`: {repo.summary} "
            f"Tags: {', '.join(repo.tags)}. "
            "Ask about running, baseline suitability, code quality, or extension to get a more targeted answer."
        )

