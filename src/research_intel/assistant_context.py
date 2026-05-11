from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research_intel.models import ContentItem, ContentType
from research_intel.rag import RagChunk, RagSearchResult


def content_payloads(project_root: Path, report: dict[str, Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    payloads.extend(_latest_candidate_payloads(project_root))
    for item in report.get("candidates", []):
        if isinstance(item, dict):
            payloads.append(item)
    for section in ("top_papers", "top_repos", "top_tools"):
        for item in report.get(section, []):
            if isinstance(item, dict):
                payloads.append(analysis_to_content_payload(item))
    return payloads


def content_items_from_payloads(payloads: list[dict[str, Any]]) -> list[ContentItem]:
    items: list[ContentItem] = []
    seen: set[str] = set()
    for payload in payloads:
        item_id = str(payload.get("item_id", ""))
        if not item_id or item_id in seen or not payload.get("content_type"):
            continue
        try:
            items.append(ContentItem.from_dict(payload))
            seen.add(item_id)
        except (TypeError, ValueError, KeyError):
            continue
    return items


def analysis_to_content_payload(item: dict[str, Any]) -> dict[str, Any]:
    content_type = str(item.get("content_type", "") or _content_type_from_section_item(item))
    return {
        "item_id": str(item.get("item_id", "")),
        "content_type": content_type,
        "title": str(item.get("title", "")),
        "url": str(item.get("url", "")),
        "source": str(item.get("source", "daily_report")),
        "summary": _analysis_summary(item),
        "tags": _list(item.get("tags")),
        "authors": _list(item.get("authors")),
        "published_at": item.get("published_at"),
        "metrics": dict(item.get("metrics", {})) if isinstance(item.get("metrics"), dict) else {},
        "technical_signals": _technical_signals(item),
        "links": dict(item.get("links", {})) if isinstance(item.get("links"), dict) else {},
        "raw": {"analysis": item},
    }


def selected_item_result(
    report: dict[str, Any],
    payloads: list[dict[str, Any]],
    selected_item_id: str | None,
) -> RagSearchResult | None:
    if not selected_item_id:
        return None
    selected_payload = next(
        (item for item in payloads if str(item.get("item_id", "")) == selected_item_id),
        None,
    )
    analysis = analysis_for_item(report, selected_item_id)
    if selected_payload is None and analysis is not None:
        selected_payload = analysis_to_content_payload(analysis)
    if selected_payload is None:
        return None
    title = str(selected_payload.get("title", selected_item_id))
    content_type = str(selected_payload.get("content_type", "item"))
    chunk = RagChunk(
        chunk_id=f"selected:{selected_item_id}",
        item_id=selected_item_id,
        title=title,
        kind=f"selected_{content_type}",
        url=str(selected_payload.get("url", "")),
        source="selected_item",
        text=selected_context_text(selected_payload, analysis),
    )
    return RagSearchResult(chunk=chunk, score=1.5, boost_score=1.5)


def ensure_selected_result(
    report: dict[str, Any],
    payloads: list[dict[str, Any]],
    selected_item_id: str | None,
    retrieved: list[RagSearchResult],
) -> list[RagSearchResult]:
    if not selected_item_id or any(result.chunk.item_id == selected_item_id for result in retrieved):
        return retrieved
    result = selected_item_result(report, payloads, selected_item_id)
    if result is None:
        return retrieved
    return [result, *retrieved]


def selected_context_text(payload: dict[str, Any], analysis: dict[str, Any] | None = None) -> str:
    if analysis is None:
        analysis = analysis_from_payload(payload)
    parts = [
        f"Content type: {payload.get('content_type', '')}",
        f"Title: {payload.get('title', '')}",
        f"Summary: {payload.get('summary', '')}",
    ]
    if analysis:
        parts.extend(
            [
                f"Why it matters: {analysis.get('why_it_matters', '')}",
                f"Relation to user: {analysis.get('relation_to_user', '')}",
                f"Technical core: {analysis.get('technical_core', '')}",
                "Strengths: " + "; ".join(str(item) for item in _list(analysis.get("strengths"))[:4]),
                "Limitations: " + "; ".join(str(item) for item in _list(analysis.get("limitations"))[:4]),
                "Possible actions: " + "; ".join(str(item) for item in _list(analysis.get("possible_actions"))[:4]),
                "Evidence: " + "; ".join(str(item) for item in _list(analysis.get("evidence"))[:4]),
            ]
        )
    else:
        parts.extend(
            [
                "Tags: " + ", ".join(str(item) for item in _list(payload.get("tags"))),
                "Authors: " + ", ".join(str(item) for item in _list(payload.get("authors"))),
                "Metrics: " + json.dumps(payload.get("metrics", {}), ensure_ascii=False),
                "Technical signals: " + json.dumps(payload.get("technical_signals", {}), ensure_ascii=False),
            ]
        )
    return " ".join(part for part in parts if part and not part.endswith(": "))


def analysis_for_item(report: dict[str, Any], item_id: str) -> dict[str, Any] | None:
    for section in ("top_papers", "top_repos", "top_tools"):
        for item in report.get(section, []):
            if isinstance(item, dict) and item.get("item_id") == item_id:
                return item
    return None


def analysis_from_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    raw = payload.get("raw")
    if isinstance(raw, dict) and isinstance(raw.get("analysis"), dict):
        return raw["analysis"]
    return None


def _latest_candidate_payloads(project_root: Path) -> list[dict[str, Any]]:
    path = project_root / "data" / "runs" / "latest_candidates.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []


def _analysis_summary(item: dict[str, Any]) -> str:
    return " ".join(
        part
        for part in (
            str(item.get("technical_core", "")),
            str(item.get("why_it_matters", "")),
            str(item.get("relation_to_user", "")),
        )
        if part
    )


def _technical_signals(item: dict[str, Any]) -> dict[str, Any]:
    signals = dict(item.get("technical_signals", {})) if isinstance(item.get("technical_signals"), dict) else {}
    if item.get("technical_core"):
        signals["technical_core"] = item["technical_core"]
    if item.get("reproducibility") is not None:
        signals["reproducibility_score"] = item["reproducibility"]
    if item.get("practical_utility") is not None:
        signals["practical_utility_score"] = item["practical_utility"]
    return signals


def _content_type_from_section_item(item: dict[str, Any]) -> str:
    url = str(item.get("url", "")).lower()
    if "github.com" in url:
        return ContentType.REPO.value
    return ContentType.PAPER.value


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
