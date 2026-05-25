from __future__ import annotations

from research_intel.models import ContentItem, ContentType, FilterDecision


class InsightGenerator:
    """Generates human-readable textual insights for a research content item."""

    def generate(
        self,
        item: ContentItem,
        relevance: float,
        decision: FilterDecision,
    ) -> dict[str, object]:
        return {
            "why_it_matters": self._why_it_matters(item),
            "relation_to_user": self._relation_to_user(item, relevance),
            "technical_core": self._technical_core(item),
            "strengths": self._strengths(item),
            "limitations": self._limitations(item),
            "possible_actions": self._actions(item),
            "evidence": self._evidence(item, decision),
        }

    # ── Text generation ───────────────────────────────────────────────────

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
        return str(core) if core else item.summary

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
        evidence = [
            f"filter status: {decision.status.value}",
            f"relevance={decision.relevance_score}",
            f"quality={decision.quality_score}",
        ]
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
