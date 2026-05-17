from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from research_intel.agents.base import BaseAgent
from research_intel.agents.repo_qa_agent import RepoQAAgent
from research_intel.assistant_context import ensure_selected_result
from research_intel.llm import LLMError, QwenChatClient
from research_intel.models import ContentItem, ContentType
from research_intel.rag import RagChunk, RagIndex, RagSearchResult


@dataclass(slots=True)
class AssistantAnswer:
    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    mode: str = "local"
    llm_error: str | None = None
    grounding: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "mode": self.mode,
            "llm_error": self.llm_error,
            "grounding": self.grounding,
        }


class ResearchAssistantAgent(BaseAgent):
    name = "research-assistant-agent"

    def __init__(self, llm_client: QwenChatClient | None = None) -> None:
        self.llm_client = llm_client or QwenChatClient()

    def answer(
        self,
        question: str,
        report: dict[str, Any],
        candidates: list[ContentItem],
        selected_item_id: str | None = None,
        retrieved: list[RagSearchResult] | None = None,
    ) -> AssistantAnswer:
        question = question.strip()
        if not question:
            return AssistantAnswer(answer="请输入一个具体问题。", mode="empty")

        lowered = question.lower()
        selected_item = self._find_item(candidates, selected_item_id) if selected_item_id else None
        if self._asks_identity(lowered):
            return AssistantAnswer(answer=self._identity_answer(), mode="identity")

        retrieved = retrieved if retrieved is not None else self._retrieve(question, report, candidates, selected_item_id)
        retrieved = ensure_selected_result(
            report,
            [self._candidate_payload(item) for item in candidates],
            selected_item_id,
            retrieved,
        )
        if self._asks_repo(lowered) and not selected_item:
            retrieved = self._prioritize_report_top_repo(report, retrieved)
        sources = [result.source_payload() for result in retrieved]

        if selected_item and selected_item.content_type == ContentType.REPO and self._asks_repo(lowered) and not self.llm_client.enabled:
            return AssistantAnswer(
                answer=RepoQAAgent().answer(selected_item, question),
                sources=sources,
                mode="repo_qa_local",
            )

        if self.llm_client.enabled:
            try:
                answer = self.llm_client.answer_question(
                    question,
                    [self._llm_context(result) for result in retrieved],
                )
                grounding = self._check_grounding(answer, retrieved)
                return AssistantAnswer(answer=answer, sources=sources, mode="qwen_rag", grounding=grounding)
            except LLMError as exc:
                fallback = self._fallback_answer(question, lowered, report, candidates, selected_item, retrieved)
                return AssistantAnswer(
                    answer=f"千问调用失败，以下是基于本地 RAG 检索的回答：{exc}\n\n{fallback}",
                    sources=sources,
                    mode="local_after_llm_error",
                    llm_error=str(exc),
                )

        return AssistantAnswer(
            answer=self._fallback_answer(question, lowered, report, candidates, selected_item, retrieved),
            sources=sources,
            mode="local_rag",
        )

    def _retrieve(
        self,
        question: str,
        report: dict[str, Any],
        candidates: list[ContentItem],
        selected_item_id: str | None,
    ) -> list[RagSearchResult]:
        index = RagIndex.from_report(report, [self._candidate_payload(item) for item in candidates])
        return index.search(question, selected_item_id=selected_item_id)

    def _fallback_answer(
        self,
        question: str,
        lowered: str,
        report: dict[str, Any],
        candidates: list[ContentItem],
        selected_item: ContentItem | None,
        retrieved: list[RagSearchResult],
    ) -> str:
        if selected_item and selected_item.content_type == ContentType.REPO and self._asks_repo(lowered):
            return RepoQAAgent().answer(selected_item, question)

        if self._asks_trend(lowered):
            return self._trend_answer(report, retrieved)
        if self._asks_action(lowered):
            return self._action_answer(report, retrieved)
        if self._asks_repo(lowered):
            return self._repo_answer(report, candidates, retrieved)
        if self._asks_paper(lowered):
            return self._paper_answer(report, retrieved)

        if not retrieved:
            return (
                "当前报告里没有检索到足够相关的证据。建议先运行一次 live/hybrid 检索，"
                "或者在论文、repo、趋势卡片上点 Ask 后带着具体上下文追问。"
            )

        lines = [
            "我基于当前报告中检索到的证据给出一个初步判断：",
            "",
        ]
        for result in retrieved[:3]:
            chunk = result.chunk
            lines.append(f"- {chunk.title}：{_trim(chunk.text, 180)}")
        lines.append("")
        lines.append("当前未启用千问 LLM，因此这是本地 RAG 摘要，不会生成超出证据的复杂推理。")
        return "\n".join(lines)

    def _identity_answer(self) -> str:
        return (
            "我是这个项目里的 Research Assistant Agent。我的职责是基于你的研究画像、"
            "每日检索结果、论文价值分析、repo 质量分析、趋势信号和反馈记录，回答你关于"
            "“今天读什么、哪个项目适合作 baseline、某个方向是否值得跟进、下一步怎么做”的问题。"
            "\n\n当前版本使用本地 RAG 检索证据；如果在 .env 中开启 ENABLE_LLM_ANALYSIS=true 并填写 DASHSCOPE_API_KEY，"
            "我会把检索到的证据交给千问模型生成更自然的答案。"
        )

    def _trend_answer(self, report: dict[str, Any], retrieved: list[RagSearchResult]) -> str:
        trends = [item for item in report.get("trends", []) if isinstance(item, dict)]
        if not trends:
            return "当前报告没有形成明确趋势信号。建议扩大 live/hybrid 检索源，并连续观察 7 天和 30 天窗口。"

        lines = ["当前最值得关注的趋势机会："]
        for trend in trends[:3]:
            lines.append(
                f"- {trend.get('topic', 'unknown')}（{trend.get('window_days', '?')}d）："
                f"{trend.get('user_implication') or trend.get('summary', '')}"
            )
        if retrieved:
            lines.append("")
            lines.append(f"主要证据来自：{', '.join(result.chunk.title for result in retrieved[:3])}")
        return "\n".join(lines)

    def _action_answer(self, report: dict[str, Any], retrieved: list[RagSearchResult]) -> str:
        actions = [str(action) for action in report.get("actions", []) if action]
        if not actions:
            return "当前报告还没有行动建议。建议先运行一次日报生成。"
        lines = ["建议按这个顺序推进："]
        lines.extend(f"{index + 1}. {action}" for index, action in enumerate(actions[:5]))
        if retrieved:
            lines.append("")
            lines.append(f"检索证据：{', '.join(result.chunk.title for result in retrieved[:3])}")
        return "\n".join(lines)

    def _repo_answer(
        self,
        report: dict[str, Any],
        candidates: list[ContentItem],
        retrieved: list[RagSearchResult],
    ) -> str:
        analysis = self._top_repo_analysis(report)
        if analysis:
            repo = self._find_item(candidates, str(analysis.get("item_id", "")))
        else:
            repo = self._best_repo_from_retrieval(candidates, retrieved) or self._best_repo(report, candidates)
        if repo is None and analysis is None:
            return "当前报告里没有可用于项目问答的 repo。建议先运行 live/hybrid 检索，或打开某个 repo 卡片后点 Ask。"

        if analysis is None and repo is not None:
            analysis = self._analysis_for_item(report, repo.item_id)
        title = repo.title if repo is not None else str(analysis.get("title", "top repo"))
        summary = repo.summary if repo is not None else ""
        lines = [f"`{title}` 可以作为候选 baseline，但需要先做可复现实验检查。"]
        if analysis:
            utility = float(analysis.get("practical_utility", 0) or 0)
            reproducibility = float(analysis.get("reproducibility", 0) or 0)
            lines.append(f"- 实用性评分：{utility:.1f}/10；可复现性评分：{reproducibility:.1f}/10。")
            if analysis.get("technical_core"):
                lines.append(f"- 核心价值：{analysis.get('technical_core')}")
            limitations = [str(item) for item in analysis.get("limitations", [])][:2]
            if limitations:
                lines.append(f"- 主要风险：{'; '.join(limitations)}")
        else:
            lines.append(f"- 当前只有候选元数据：{_trim(summary, 180)}")
        lines.append("- 建议先确认 README 运行路径、demo 是否可跑、模型/数据权重是否可获得，再决定是否纳入你的实验基线。")
        return "\n".join(lines)

    def _paper_answer(self, report: dict[str, Any], retrieved: list[RagSearchResult]) -> str:
        papers = [item for item in report.get("top_papers", []) if isinstance(item, dict)]
        if not papers:
            return "当前报告没有选出论文。建议先运行 live/hybrid 检索，或者在 Profile 中增加更明确的研究关键词。"
        paper = papers[0]
        lines = [
            f"建议优先看 `{paper.get('title', 'top paper')}`。",
            f"- 重要性：{paper.get('why_it_matters', '')}",
            f"- 和你方向的关系：{paper.get('relation_to_user', '')}",
        ]
        limitations = [str(item) for item in paper.get("limitations", [])][:2]
        if limitations:
            lines.append(f"- 阅读时重点质疑：{'; '.join(limitations)}")
        if retrieved:
            lines.append(f"- 检索证据：{', '.join(result.chunk.title for result in retrieved[:3])}")
        return "\n".join(lines)

    def _check_grounding(self, answer: str, chunks: list[RagSearchResult]) -> dict[str, Any]:
        """Check how well the LLM answer is grounded in the retrieved chunks.

        Extracts concrete claims (numbers, percentages, Title-Case sequences that
        could be model/paper/author names) from the answer and checks whether each
        appears in the combined chunk texts.

        Returns a dict with:
            grounding_score  – float in [0, 1]
            total_claims     – number of specific claims detected
            grounded_claims  – how many were found in chunks
            confidence       – "high" | "medium" | "low"
        """
        if not chunks:
            return {"grounding_score": 0.0, "total_claims": 0, "grounded_claims": 0, "confidence": "low"}

        chunk_text = " ".join(r.chunk.text for r in chunks).lower()

        # Numbers and percentages are often hallucinated
        numeric_claims = re.findall(r"\b\d+\.?\d*\s*%?|\b\d{4}\b", answer)
        # Title-Case sequences (likely model names, paper names, authors)
        title_claims = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", answer)

        all_claims = [c.strip() for c in numeric_claims + title_claims if c.strip()]

        if not all_claims:
            # No verifiable claims → moderate default (answer may be valid but unverifiable)
            return {"grounding_score": 0.6, "total_claims": 0, "grounded_claims": 0, "confidence": "medium"}

        grounded = sum(1 for c in all_claims if c.lower() in chunk_text)
        score = grounded / len(all_claims)

        return {
            "grounding_score": round(score, 2),
            "total_claims": len(all_claims),
            "grounded_claims": grounded,
            "confidence": "high" if score > 0.7 else "medium" if score > 0.4 else "low",
        }

    def _llm_context(self, result: RagSearchResult) -> dict[str, str]:
        chunk = result.chunk
        return {
            "id": chunk.chunk_id,
            "title": chunk.title,
            "kind": chunk.kind,
            "item_id": chunk.item_id,
            "url": chunk.url,
            "score": f"{result.score:.4f}",
            "text": chunk.text[:1800],
        }

    def _ensure_selected_context(
        self,
        report: dict[str, Any],
        selected_item: ContentItem,
        retrieved: list[RagSearchResult],
    ) -> list[RagSearchResult]:
        if any(result.chunk.item_id == selected_item.item_id for result in retrieved):
            return retrieved
        analysis = self._analysis_for_item(report, selected_item.item_id)
        text = self._selected_context_text(selected_item, analysis)
        chunk = RagChunk(
            chunk_id=f"selected:{selected_item.item_id}",
            item_id=selected_item.item_id,
            title=selected_item.title,
            kind=f"selected_{selected_item.content_type.value}",
            url=selected_item.url,
            source="selected_item",
            text=text,
        )
        return [RagSearchResult(chunk=chunk, score=1.5, boost_score=1.5), *retrieved]

    def _selected_context_text(self, selected_item: ContentItem, analysis: dict[str, Any] | None) -> str:
        if analysis:
            parts = [
                f"Content type: {analysis.get('content_type', selected_item.content_type.value)}",
                f"Why it matters: {analysis.get('why_it_matters', '')}",
                f"Relation to user: {analysis.get('relation_to_user', '')}",
                f"Technical core: {analysis.get('technical_core', '')}",
                "Strengths: " + "; ".join(str(item) for item in self._selected_list(analysis.get("strengths"))[:4]),
                "Limitations: " + "; ".join(str(item) for item in self._selected_list(analysis.get("limitations"))[:4]),
                "Possible actions: " + "; ".join(str(item) for item in self._selected_list(analysis.get("possible_actions"))[:4]),
                "Evidence: " + "; ".join(str(item) for item in self._selected_list(analysis.get("evidence"))[:4]),
            ]
            return " ".join(part for part in parts if part.strip(": "))
        return " ".join(
            part
            for part in (
                selected_item.summary,
                "Tags: " + ", ".join(selected_item.tags),
                "Authors: " + ", ".join(selected_item.authors),
                "Metrics: " + str(selected_item.metrics),
                "Technical signals: " + str(selected_item.technical_signals),
            )
            if part and part != "{}"
        )

    def _selected_list(self, value: Any) -> list[Any]:
        return value if isinstance(value, list) else []

    def _prioritize_report_top_repo(
        self,
        report: dict[str, Any],
        retrieved: list[RagSearchResult],
    ) -> list[RagSearchResult]:
        analysis = self._top_repo_analysis(report)
        if not analysis:
            return retrieved
        item_id = str(analysis.get("item_id", ""))
        if item_id:
            matching = [
                RagSearchResult(chunk=result.chunk, score=max(result.score, 1.25))
                for result in retrieved
                if result.chunk.item_id == item_id and result.chunk.kind == "repo_analysis"
            ]
            if matching:
                others = [
                    result
                    for result in retrieved
                    if not (result.chunk.item_id == item_id and result.chunk.kind == "repo_analysis")
                ]
                return [*matching, *others]
        chunk = RagChunk(
            chunk_id=f"analysis:top_repos:{item_id}",
            item_id=item_id,
            title=str(analysis.get("title", "Top repo")),
            kind="repo_analysis",
            url=str(analysis.get("url", "")),
            source="daily_report",
            text=" ".join(
                part
                for part in (
                    str(analysis.get("why_it_matters", "")),
                    str(analysis.get("relation_to_user", "")),
                    str(analysis.get("technical_core", "")),
                    "Limitations: " + "; ".join(str(item) for item in analysis.get("limitations", [])[:3]),
                    "Actions: " + "; ".join(str(item) for item in analysis.get("possible_actions", [])[:3]),
                )
                if part
            ),
        )
        return [RagSearchResult(chunk=chunk, score=1.25), *retrieved]

    def _candidate_payload(self, item: ContentItem) -> dict[str, Any]:
        return {
            "item_id": item.item_id,
            "content_type": item.content_type.value,
            "title": item.title,
            "url": item.url,
            "source": item.source,
            "summary": item.summary,
            "tags": item.tags,
            "authors": item.authors,
            "published_at": item.published_at,
            "metrics": item.metrics,
            "technical_signals": item.technical_signals,
            "links": item.links,
        }

    def _asks_identity(self, lowered: str) -> bool:
        return any(term in lowered for term in ("你是谁", "你是什么", "什么agent", "什么 agent", "你的作用", "who are you", "what are you"))

    def _asks_repo(self, lowered: str) -> bool:
        return any(term in lowered for term in ("repo", "github", "项目", "baseline", "基线", "运行", "安装", "代码"))

    def _asks_trend(self, lowered: str) -> bool:
        return any(term in lowered for term in ("trend", "趋势", "方向", "机会", "选题"))

    def _asks_paper(self, lowered: str) -> bool:
        return any(term in lowered for term in ("paper", "论文", "精读", "阅读", "方法", "实验", "evaluation"))

    def _asks_action(self, lowered: str) -> bool:
        return any(term in lowered for term in ("建议", "下一步", "action", "today", "今天", "优先", "做什么"))

    def _best_repo_from_retrieval(self, candidates: list[ContentItem], retrieved: list[RagSearchResult]) -> ContentItem | None:
        repos = {item.item_id: item for item in candidates if item.content_type == ContentType.REPO}
        for result in retrieved:
            if result.chunk.item_id in repos:
                return repos[result.chunk.item_id]
        return None

    def _best_repo(self, report: dict[str, Any], candidates: list[ContentItem]) -> ContentItem | None:
        analysis = self._top_repo_analysis(report)
        if analysis:
            repo = self._find_item(candidates, str(analysis.get("item_id", "")))
            if repo:
                return repo
        return next((item for item in candidates if item.content_type == ContentType.REPO), None)

    def _top_repo_analysis(self, report: dict[str, Any]) -> dict[str, Any] | None:
        top_repos = [item for item in report.get("top_repos", []) if isinstance(item, dict)]
        return top_repos[0] if top_repos else None

    def _analysis_for_item(self, report: dict[str, Any], item_id: str) -> dict[str, Any] | None:
        for section in ("top_papers", "top_repos", "top_tools"):
            for item in report.get(section, []):
                if isinstance(item, dict) and item.get("item_id") == item_id:
                    return item
        return None

    def _find_item(self, candidates: list[ContentItem], item_id: str | None) -> ContentItem | None:
        if not item_id:
            return None
        return next((item for item in candidates if item.item_id == item_id), None)


def _trim(text: str, limit: int) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "..."
