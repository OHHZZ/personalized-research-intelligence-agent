from __future__ import annotations

from typing import Any

from research_intel.models import ContentItem, UserProfile


# ── Tag utilities ─────────────────────────────────────────────────────────────

def profile_tags(text: str, keywords: list[str]) -> list[str]:
    """Return profile keywords that appear verbatim in text (case-insensitive)."""
    text_lower = text.lower()
    return [kw for kw in keywords if kw.strip() and kw.strip().lower() in text_lower]


def enrich_with_profile_tags(items: list[ContentItem], profile: UserProfile) -> list[ContentItem]:
    """Prepend profile-matched keywords to each item's tag list.

    Runs after items are built so helper functions don't need to receive the
    profile object.  Existing tags are kept; new matches go to the front so
    they surface first in the UI.
    """
    keywords = profile.keywords()
    if not keywords:
        return items
    for item in items:
        search_text = f"{item.title} {item.summary} {' '.join(item.tags)}"
        matched = profile_tags(search_text, keywords)
        if matched:
            item.tags = unique([*matched, *item.tags])
    return items


# ── Paper signal extraction ───────────────────────────────────────────────────

def text_paper_signals(title: str, abstract: str) -> dict[str, Any]:
    """Derive technical signals from title + abstract text.

    Shared across paper-type connectors so each connector only adds its
    source-specific extra fields on top of this base dict.
    """
    combined = f"{title} {abstract}".lower()

    has_eval = any(t in combined for t in (
        "experiment", "evaluation", "benchmark", "dataset", "metric",
    ))
    has_ablation = "ablation" in combined
    has_baseline = any(t in combined for t in (
        "baseline", "state-of-the-art", "sota", "compared to", "outperform",
    ))
    has_code = any(t in combined for t in ("github", "code", "implementation", "repository"))

    novelty = 5.8
    if any(t in combined for t in ("novel", "propose", "introduce", "new approach", "first")):
        novelty = 6.7
    if any(t in combined for t in ("benchmark", "dataset", "evaluation")):
        novelty = min(7.5, novelty + 0.4)

    depth = "high" if (has_eval and has_baseline) else "medium" if has_eval else "low"

    return {
        "has_experiments": has_eval,
        "has_ablation": has_ablation,
        "has_strong_baselines": has_baseline,
        "has_code": has_code,
        "has_benchmark": "benchmark" in combined,
        "baseline_count": 2 if has_baseline else 0,
        "novelty": novelty,
        "technical_depth": depth,
        "has_known_gap": any(t in combined for t in (
            "limitation", "challenge", "gap", "future work", "open problem",
        )),
        "benchmark_gap": "benchmark" in combined or "evaluation gap" in combined,
        "technical_core": abstract[:500],
        "cites_sources": any(t in combined for t in ("citation", "reference", "related work")),
        "has_metrics": any(t in combined for t in ("accuracy", "f1", "bleu", "rouge", "perplexity", "%")),
        "has_public_eval": has_eval and ("leaderboard" in combined or "public" in combined),
    }


# ── Collection utilities ──────────────────────────────────────────────────────

def unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        nv = v.strip()
        key = nv.lower()
        if nv and key not in seen:
            seen.add(key)
            out.append(nv)
    return out


def dedupe_items(items: list[ContentItem]) -> list[ContentItem]:
    """Deduplicate items by URL (falling back to title)."""
    seen: set[str] = set()
    out: list[ContentItem] = []
    for item in items:
        key = (item.url or item.title).lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out
