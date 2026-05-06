from __future__ import annotations

import re
from dataclasses import dataclass

from research_intel.agents.base import BaseAgent
from research_intel.agents.repo_qa_agent import RepoQAAgent
from research_intel.llm import LLMError, QwenChatClient
from research_intel.models import ContentItem, ContentType


TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_\-/.+]*|[\u4e00-\u9fff]")


@dataclass(slots=True)
class ContextChunk:
    chunk_id: str
    title: str
    kind: str
    text: str


class ResearchAssistantAgent(BaseAgent):
    name = "research-assistant-agent"

    def __init__(self, llm_client: QwenChatClient | None = None) -> None:
        self.llm_client = llm_client or QwenChatClient()

    def answer(
        self,
        question: str,
        report: dict[str, object],
        candidates: list[ContentItem],
        selected_item_id: str | None = None,
    ) -> str:
        question = question.strip()
        q = question.lower()
        selected_item = self._find_item(candidates, selected_item_id) if selected_item_id else None

        if self._asks_identity(q):
            return self._identity_answer()

        chunks = self._build_context(report, candidates, selected_item)
        retrieved = self._retrieve(question, chunks, selected_item_id=selected_item_id)

        if self.llm_client.enabled:
            try:
                return self.llm_client.answer_question(question, [self._chunk_payload(chunk) for chunk in retrieved])
            except LLMError as exc:
                return self._fallback_answer(question, q, report, candidates, selected_item, retrieved, llm_error=str(exc))

        return self._fallback_answer(question, q, report, candidates, selected_item, retrieved)

    def _fallback_answer(
        self,
        question: str,
        q: str,
        report: dict[str, object],
        candidates: list[ContentItem],
        selected_item: ContentItem | None,
        retrieved: list[ContextChunk],
        llm_error: str | None = None,
    ) -> str:
        prefix = ""
        if llm_error:
            prefix = f"LLM 调用失败，以下是基于本地检索规则的回答：{llm_error}\n\n"

        if selected_item and selected_item.content_type == ContentType.REPO and self._asks_repo(q):
            return prefix + RepoQAAgent().answer(selected_item, question)

        if self._asks_repo(q):
            repo = selected_item if selected_item and selected_item.content_type == ContentType.REPO else self._best_repo(report, candidates)
            if repo:
                return prefix + RepoQAAgent().answer(repo, question)
            return prefix + "当前报告里没有可用于项目问答的 repo。建议先运行 live/hybrid 检索，或点击某个 repo 卡片上的 Ask。"

        if self._asks_trend(q):
            return prefix + self._trend_answer(report)
        if self._asks_paper(q):
            return prefix + self._paper_answer(report)
        if self._asks_action(q):
            return prefix + self._action_answer(report)

        if retrieved:
            lines = [
                prefix + "当前未启用千问 LLM，我只能基于本地检索到的上下文给出有限回答。",
                "与你的问题最相关的上下文是：",
            ]
            for chunk in retrieved[:3]:
                lines.append(f"- {chunk.title}：{chunk.text[:220]}")
            lines.append("如果希望获得真正的自然语言推理回答，请在 `.env` 中设置 `ENABLE_LLM_ANALYSIS=true` 并填写 `DASHSCOPE_API_KEY`。")
            return "\n".join(lines)

        return prefix + "当前上下文不足以回答这个问题。建议先运行一次 live/hybrid 检索，或点击某篇论文/repo 的 Ask 后再追问。"

    def _identity_answer(self) -> str:
        return (
            "我是这个项目里的 Research Assistant Agent。"
            "我的目标是基于你的研究画像、每日检索结果、论文分析、repo 分析和趋势信号，"
            "帮助你判断今天该读什么、哪个项目适合作为 baseline、哪些方向值得继续跟踪。"
            "当前如果未开启千问 LLM，我会使用本地检索和规则回答；开启后会用检索到的上下文交给千问生成答案。"
        )

    def _build_context(
        self,
        report: dict[str, object],
        candidates: list[ContentItem],
        selected_item: ContentItem | None,
    ) -> list[ContextChunk]:
        chunks: list[ContextChunk] = []

        if selected_item:
            chunks.append(
                ContextChunk(
                    chunk_id=f"selected:{selected_item.item_id}",
                    title=selected_item.title,
                    kind=selected_item.content_type.value,
                    text=self._item_text(selected_item),
                )
            )

        for section, kind in (
            ("top_papers", "paper_analysis"),
            ("top_repos", "repo_analysis"),
            ("top_tools", "tool_analysis"),
        ):
            for item in report.get(section, []) if isinstance(report, dict) else []:
                if not isinstance(item, dict):
                    continue
                chunks.append(
                    ContextChunk(
                        chunk_id=str(item.get("item_id", item.get("title", ""))),
                        title=str(item.get("title", "")),
                        kind=kind,
                        text=self._analysis_text(item),
                    )
                )

        for idx, trend in enumerate(report.get("trends", []) if isinstance(report, dict) else []):
            if not isinstance(trend, dict):
                continue
            chunks.append(
                ContextChunk(
                    chunk_id=f"trend:{idx}",
                    title=str(trend.get("topic", "trend")),
                    kind="trend",
                    text=" ".join(
                        str(part)
                        for part in (
                            trend.get("summary", ""),
                            trend.get("user_implication", ""),
                            "; ".join(trend.get("signals", [])) if isinstance(trend.get("signals"), list) else "",
                        )
                    ),
                )
            )

        actions = report.get("actions", []) if isinstance(report, dict) else []
        if actions:
            chunks.append(
                ContextChunk(
                    chunk_id="actions",
                    title="Recommended actions",
                    kind="actions",
                    text="\n".join(str(action) for action in actions),
                )
            )

        for item in candidates[:50]:
            chunks.append(
                ContextChunk(
                    chunk_id=f"candidate:{item.item_id}",
                    title=item.title,
                    kind=item.content_type.value,
                    text=self._item_text(item),
                )
            )

        return chunks

    def _retrieve(
        self,
        question: str,
        chunks: list[ContextChunk],
        selected_item_id: str | None,
        limit: int = 8,
    ) -> list[ContextChunk]:
        query_tokens = self._tokens(question)
        scored: list[tuple[float, ContextChunk]] = []
        for chunk in chunks:
            text = f"{chunk.title} {chunk.kind} {chunk.text}".lower()
            chunk_tokens = self._tokens(text)
            overlap = len(query_tokens & chunk_tokens)
            phrase_bonus = sum(1 for token in query_tokens if len(token) > 1 and token in text)
            selected_bonus = 3.0 if selected_item_id and selected_item_id in chunk.chunk_id else 0.0
            score = overlap + phrase_bonus * 0.6 + selected_bonus
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored and selected_item_id:
            return [chunk for chunk in chunks if selected_item_id in chunk.chunk_id][:limit]
        return [chunk for _, chunk in scored[:limit]]

    def _tokens(self, text: str) -> set[str]:
        tokens = {token.lower() for token in TOKEN_RE.findall(text)}
        joined = "".join(token for token in tokens if "\u4e00" <= token <= "\u9fff")
        tokens.update(joined[index : index + 2] for index in range(max(0, len(joined) - 1)))
        return {token for token in tokens if token.strip()}

    def _chunk_payload(self, chunk: ContextChunk) -> dict[str, str]:
        return {
            "id": chunk.chunk_id,
            "title": chunk.title,
            "kind": chunk.kind,
            "text": chunk.text[:1800],
        }

    def _analysis_text(self, item: dict[str, object]) -> str:
        parts = [
            item.get("content_type", ""),
            item.get("why_it_matters", ""),
            item.get("relation_to_user", ""),
            item.get("technical_core", ""),
            "strengths: " + "; ".join(item.get("strengths", [])) if isinstance(item.get("strengths"), list) else "",
            "limitations: " + "; ".join(item.get("limitations", [])) if isinstance(item.get("limitations"), list) else "",
            "actions: " + "; ".join(item.get("possible_actions", [])) if isinstance(item.get("possible_actions"), list) else "",
            "evidence: " + "; ".join(item.get("evidence", [])) if isinstance(item.get("evidence"), list) else "",
        ]
        return " ".join(str(part) for part in parts if part)

    def _item_text(self, item: ContentItem) -> str:
        signals = "; ".join(f"{key}={value}" for key, value in item.technical_signals.items())
        metrics = "; ".join(f"{key}={value}" for key, value in item.metrics.items())
        return " ".join(
            part
            for part in (
                item.summary,
                "tags: " + ", ".join(item.tags),
                "signals: " + signals,
                "metrics: " + metrics,
                "url: " + item.url,
            )
            if part
        )

    def _asks_identity(self, q: str) -> bool:
        return any(term in q for term in ("你是什么", "什么agent", "什么 agent", "你的作用", "你是谁", "who are you", "what are you"))

    def _asks_repo(self, q: str) -> bool:
        return any(term in q for term in ("repo", "github", "项目", "baseline", "基线", "运行", "安装", "代码"))

    def _asks_trend(self, q: str) -> bool:
        return any(term in q for term in ("trend", "趋势", "方向", "机会", "选题"))

    def _asks_paper(self, q: str) -> bool:
        return any(term in q for term in ("paper", "论文", "精读", "阅读", "方法", "实验"))

    def _asks_action(self, q: str) -> bool:
        return any(term in q for term in ("建议", "下一步", "action", "today", "今天", "做什么"))

    def _trend_answer(self, report: dict[str, object]) -> str:
        trends = report.get("trends", []) if isinstance(report, dict) else []
        if not trends:
            return "当前报告没有形成明确趋势信号。建议扩大 live 检索源，或把 Source 切到 hybrid 以保留样例内容辅助分析。"
        lines = ["当前最值得关注的趋势："]
        for trend in trends[:3]:
            if not isinstance(trend, dict):
                continue
            lines.append(
                f"- {trend.get('topic', 'unknown')}（{trend.get('window_days', '?')}d）："
                f"{trend.get('user_implication', trend.get('summary', ''))}"
            )
        return "\n".join(lines)

    def _paper_answer(self, report: dict[str, object]) -> str:
        papers = report.get("top_papers", []) if isinstance(report, dict) else []
        if not papers:
            return "当前报告没有选出论文。建议先运行 live/hybrid 检索，或在 Profile 中增加更明确的研究关键词。"
        top = papers[0]
        if not isinstance(top, dict):
            return "当前论文数据结构异常，建议重新运行一次日报。"
        return (
            f"建议优先看 `{top.get('title', 'the top paper')}`。\n"
            f"原因：{top.get('why_it_matters', '')}\n"
            f"阅读重点：先看方法和 evaluation，再检查局限：{'; '.join(top.get('limitations', [])[:2])}"
        )

    def _action_answer(self, report: dict[str, object]) -> str:
        actions = report.get("actions", []) if isinstance(report, dict) else []
        if not actions:
            return "当前报告还没有行动建议。建议先运行一次日报。"
        return "建议按这个顺序推进：\n" + "\n".join(f"{idx + 1}. {action}" for idx, action in enumerate(actions[:5]))

    def _best_repo(self, report: dict[str, object], candidates: list[ContentItem]) -> ContentItem | None:
        top_repos = report.get("top_repos", []) if isinstance(report, dict) else []
        if top_repos and isinstance(top_repos[0], dict):
            repo_id = str(top_repos[0].get("item_id", ""))
            repo = self._find_item(candidates, repo_id)
            if repo:
                return repo
        return next((item for item in candidates if item.content_type == ContentType.REPO), None)

    def _find_item(self, candidates: list[ContentItem], item_id: str | None) -> ContentItem | None:
        if not item_id:
            return None
        return next((item for item in candidates if item.item_id == item_id), None)
