from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

from research_intel.models import ContentItem, UserProfile, ValueAnalysis, to_plain_dict


class LLMError(RuntimeError):
    pass


class QwenChatClient:
    """DashScope/Qwen compatible-mode chat client for structured analysis."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: int = 45,
    ) -> None:
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.model = model or os.getenv("LLM_MODEL", "qwen-plus")
        self.base_url = (base_url or os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")).rstrip("/")
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.api_key) and os.getenv("ENABLE_LLM_ANALYSIS", "false").lower() in {"1", "true", "yes"}

    def analyze_item(
        self,
        profile: UserProfile | None,
        item: ContentItem,
        rule_analysis: ValueAnalysis,
    ) -> dict[str, Any]:
        if not self.enabled:
            raise LLMError("LLM analysis is disabled")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a rigorous research intelligence analyst. "
                        "Return only valid JSON with no markdown fences. "
                        "Ground judgments in the provided metadata. "
                        "Do not invent experiments, code, datasets, or results that are not evidenced."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "schema": self._schema_description(),
                            "user_profile": to_plain_dict(profile) if profile else {},
                            "content_item": to_plain_dict(item),
                            "rule_analysis": to_plain_dict(rule_analysis),
                            "task": (
                                "Assess the item's research value for this user. Prefer concrete, evidence-backed "
                                "limitations and actions over generic summaries. Return numeric scores from 0 to 10."
                            ),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }
        response = self._post(payload)
        return self._extract_json(response)

    def answer_question(self, question: str, context: list[dict[str, str]]) -> str:
        if not self.enabled:
            raise LLMError("LLM analysis is disabled")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a research assistant for a personalized research intelligence system. "
                        "Answer in the user's language. Use only the provided context. "
                        "If the context is insufficient, say what is missing and suggest the next useful action. "
                        "Do not invent papers, repos, experiments, metrics, or repo details. "
                        "Write in natural prose. Avoid emoji and decorative symbols. "
                        "Use simple paragraphs and short bullet lists only when they improve readability."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": question,
                            "retrieved_context": context,
                            "answer_requirements": [
                                "Be concise but specific.",
                                "Mention which context item supports the answer when useful.",
                                "For baseline questions, discuss suitability and risk.",
                                "For trend questions, separate signal from recommendation.",
                            ],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": 0.2,
        }
        response = self._post(payload)
        return self._extract_text(response)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMError(f"Qwen/DashScope API HTTP {exc.code}: {detail[:500]}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"Qwen/DashScope API network error: {exc.reason}") from exc

    def _extract_json(self, response: dict[str, Any]) -> dict[str, Any]:
        choices = response.get("choices", [])
        if not choices:
            raise LLMError("Qwen response did not contain choices")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise LLMError("Qwen response did not contain message content")
        return json.loads(_strip_json_fence(content))

    def _extract_text(self, response: dict[str, Any]) -> str:
        choices = response.get("choices", [])
        if not choices:
            raise LLMError("Qwen response did not contain choices")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise LLMError("Qwen response did not contain message content")
        return content.strip()

    def _schema_description(self) -> dict[str, Any]:
        return {
            "score": "number 0-10",
            "relevance": "number 0-10",
            "novelty": "number 0-10",
            "technical_depth": "number 0-10",
            "evidence_strength": "number 0-10",
            "reproducibility": "number 0-10",
            "practical_utility": "number 0-10",
            "trend_signal": "number 0-10",
            "research_opportunity": "number 0-10",
            "why_it_matters": "string",
            "relation_to_user": "string",
            "technical_core": "string",
            "strengths": "array of strings",
            "limitations": "array of strings",
            "possible_actions": "array of strings",
            "evidence": "array of strings",
            "confidence": "one of low, medium, high",
        }


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return stripped
