from __future__ import annotations

from typing import Any


def evaluate_assistant_response(
    question: str,
    answer: str,
    sources: list[dict[str, Any]],
    llm_enabled: bool,
) -> dict[str, Any]:
    """Return lightweight self-feedback for debugging assistant behavior."""

    warnings: list[str] = []
    stripped_answer = answer.strip()
    lower_question = question.lower()
    is_identity_question = any(
        term in lower_question
        for term in ("你是谁", "你是什么", "什么agent", "什么 agent", "who are you", "what are you")
    )

    if not stripped_answer:
        warnings.append("empty_answer")
    if not sources and not is_identity_question:
        warnings.append("no_retrieved_sources")
    if len(stripped_answer) < 20 and not is_identity_question:
        warnings.append("very_short_answer")
    if not llm_enabled:
        warnings.append("llm_disabled_local_fallback")

    return {
        "status": "ok" if not warnings else "needs_review",
        "warnings": warnings,
        "source_count": len(sources),
        "llm_enabled": llm_enabled,
    }
