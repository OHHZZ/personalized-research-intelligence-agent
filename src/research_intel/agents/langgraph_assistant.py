from __future__ import annotations

import asyncio
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable, TypedDict

warnings.filterwarnings("ignore", message="The default value of `allowed_objects`.*", category=Warning)
warnings.filterwarnings("ignore", message="The default value of `allowed_objects`.*", category=PendingDeprecationWarning)

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from research_intel.agents.research_assistant_agent import ResearchAssistantAgent
from research_intel.assistant_context import (
    content_items_from_payloads,
    content_payloads,
    ensure_selected_result,
)
from research_intel.evaluation import evaluate_assistant_response
from research_intel.models import ContentItem
from research_intel.rag import RagIndex, RagSearchResult
from research_intel.storage import JsonStore


class AssistantGraphState(TypedDict, total=False):
    question: str
    selected_item_id: str
    report: dict[str, Any]
    candidate_payloads: list[dict[str, Any]]
    candidates: list[ContentItem]
    retrieved: list[RagSearchResult]
    sources: list[dict[str, Any]]
    plan: list[str]
    tool_results: list[dict[str, Any]]
    answer: str
    mode: str
    llm_error: str | None
    reflection: dict[str, Any]
    evaluation: dict[str, Any]


class LangGraphAssistant:
    """LangGraph-based assistant for streamed RAG + MCP tool grounded answers."""

    def __init__(self, project_root: Path | str) -> None:
        self.project_root = Path(project_root).resolve()
        self.store = JsonStore(self.project_root)
        self.graph = self._build_graph()

    def stream(self, question: str, selected_item_id: str | None = None) -> Iterable[dict[str, Any]]:
        latest_state: dict[str, Any] = {}
        initial: AssistantGraphState = {
            "question": question,
            "selected_item_id": selected_item_id or "",
        }
        for mode, payload in self.graph.stream(initial, stream_mode=["custom", "updates"]):
            if mode == "custom":
                yield dict(payload)
                continue
            if mode == "updates":
                for update in payload.values():
                    if isinstance(update, dict):
                        latest_state.update(update)

        answer = str(latest_state.get("answer", ""))
        sources = list(latest_state.get("sources", []))
        evaluation = latest_state.get("evaluation") or evaluate_assistant_response(
            question=question,
            answer=answer,
            sources=sources,
            llm_enabled=_llm_enabled(),
        )
        yield {
            "type": "final",
            "stage": "assistant",
            "status": "complete",
            "message": "Assistant answer ready",
            "answer": answer,
            "sources": sources,
            "mode": latest_state.get("mode", "langgraph"),
            "llm_error": latest_state.get("llm_error"),
            "evaluation": evaluation,
            "reflection": latest_state.get("reflection", {}),
        }

    def _build_graph(self):
        graph = StateGraph(AssistantGraphState)
        graph.add_node("load_context", self._load_context)
        graph.add_node("retrieve_rag", self._retrieve_rag)
        graph.add_node("plan", self._plan)
        graph.add_node("execute_tools", self._execute_tools)
        graph.add_node("generate_answer", self._generate_answer)
        graph.add_node("reflect", self._reflect)
        graph.add_edge(START, "load_context")
        graph.add_edge("load_context", "retrieve_rag")
        graph.add_edge("retrieve_rag", "plan")
        graph.add_edge("plan", "execute_tools")
        graph.add_edge("execute_tools", "generate_answer")
        graph.add_edge("generate_answer", "reflect")
        graph.add_edge("reflect", END)
        return graph.compile()

    def _load_context(self, state: AssistantGraphState) -> dict[str, Any]:
        writer = get_stream_writer()
        writer(_progress("context", "running", "Loading latest report and candidate metadata"))
        try:
            report = self.store.load_report_json("latest")
        except FileNotFoundError:
            report = {}
        payloads = content_payloads(self.project_root, report)
        candidates = content_items_from_payloads(payloads)
        writer(_progress("context", "complete", f"Loaded {len(candidates)} context records"))
        return {
            "report": report,
            "candidate_payloads": payloads,
            "candidates": candidates,
        }

    def _retrieve_rag(self, state: AssistantGraphState) -> dict[str, Any]:
        writer = get_stream_writer()
        writer(_progress("rag", "running", "Retrieving report evidence"))
        report = state.get("report", {})
        payloads = state.get("candidate_payloads", [])
        selected_item_id = state.get("selected_item_id") or None
        index = RagIndex.from_report(report, payloads)
        retrieved = index.search(state["question"], selected_item_id=selected_item_id)
        retrieved = ensure_selected_result(report, payloads, selected_item_id, retrieved)
        sources = [result.source_payload() for result in retrieved]
        writer(_progress("rag", "complete", f"Retrieved {len(retrieved)} evidence chunks"))
        return {"retrieved": retrieved, "sources": sources}

    def _plan(self, state: AssistantGraphState) -> dict[str, Any]:
        writer = get_stream_writer()
        question = state["question"].lower()
        plan = ["search_latest_report"]
        if state.get("selected_item_id"):
            plan.insert(0, "get_selected_item_context")
        if any(term in question for term in ("today", "action", "priority", "next", "今天", "优先", "下一步")):
            plan.append("list_recommended_actions")
        writer(
            {
                "type": "plan",
                "stage": "plan",
                "status": "complete",
                "message": "Planned MCP tool calls",
                "plan": plan,
            }
        )
        return {"plan": plan}

    def _execute_tools(self, state: AssistantGraphState) -> dict[str, Any]:
        writer = get_stream_writer()
        plan = state.get("plan", [])
        if not plan:
            return {"tool_results": []}
        writer(_progress("tools", "running", "Calling local MCP tools"))
        try:
            tool_results = asyncio.run(self._run_mcp_tools(state, plan))
            writer(_progress("tools", "complete", f"Collected {len(tool_results)} MCP tool result(s)"))
            return {"tool_results": tool_results}
        except Exception as exc:
            writer(_progress("tools", "warning", f"MCP tool call failed: {type(exc).__name__}: {exc}"))
            return {"tool_results": [{"tool": "mcp", "error": f"{type(exc).__name__}: {exc}"}]}

    async def _run_mcp_tools(self, state: AssistantGraphState, plan: list[str]) -> list[dict[str, Any]]:
        client = MultiServerMCPClient(
            {
                "local_research": {
                    "command": sys.executable,
                    "args": [
                        "-m",
                        "research_intel.mcp_server",
                        "--root",
                        str(self.project_root),
                    ],
                    "transport": "stdio",
                    "env": {
                        **os.environ,
                        "PYTHONPATH": str(self.project_root / "src"),
                        "RESEARCH_INTEL_ROOT": str(self.project_root),
                    },
                }
            }
        )
        tools = {tool.name: tool for tool in await client.get_tools()}
        results: list[dict[str, Any]] = []
        for tool_name in plan:
            tool = tools.get(tool_name)
            if tool is None:
                results.append({"tool": tool_name, "error": "tool_not_available"})
                continue
            args = self._tool_args(tool_name, state)
            value = await tool.ainvoke(args)
            results.append({"tool": tool_name, "args": args, "result": _tool_result_text(value)})
        return results

    def _tool_args(self, tool_name: str, state: AssistantGraphState) -> dict[str, Any]:
        if tool_name == "get_selected_item_context":
            return {"item_id": state.get("selected_item_id", "")}
        if tool_name == "search_latest_report":
            return {
                "query": state["question"],
                "selected_item_id": state.get("selected_item_id", ""),
                "limit": 5,
            }
        if tool_name == "list_recommended_actions":
            return {"limit": 5}
        return {}

    def _generate_answer(self, state: AssistantGraphState) -> dict[str, Any]:
        writer = get_stream_writer()
        writer(_progress("generation", "running", "Generating answer"))
        question = state["question"]
        sources = state.get("sources", [])
        retrieved = state.get("retrieved", [])
        candidates = state.get("candidates", [])
        selected_item_id = state.get("selected_item_id") or None

        if not _llm_enabled():
            fallback = ResearchAssistantAgent().answer(
                question,
                state.get("report", {}),
                candidates,
                selected_item_id=selected_item_id,
                retrieved=retrieved,
            )
            writer(_progress("generation", "complete", "Generated local fallback answer"))
            return {
                "answer": fallback.answer,
                "sources": fallback.sources or sources,
                "mode": "langgraph_local_rag",
                "llm_error": fallback.llm_error,
            }

        llm = _chat_model()
        messages = _answer_messages(question, retrieved, state.get("tool_results", []))
        tokens: list[str] = []
        try:
            for chunk in llm.stream(messages):
                text = _chunk_text(chunk.content)
                if not text:
                    continue
                tokens.append(text)
                writer({"type": "token", "stage": "generation", "status": "running", "text": text})
            answer = "".join(tokens).strip()
            writer(_progress("generation", "complete", "Generated streamed LLM answer"))
            return {"answer": answer, "sources": sources, "mode": "langgraph_qwen_stream", "llm_error": None}
        except Exception as exc:
            fallback = ResearchAssistantAgent().answer(
                question,
                state.get("report", {}),
                candidates,
                selected_item_id=selected_item_id,
                retrieved=retrieved,
            )
            writer(_progress("generation", "warning", f"LLM streaming failed: {type(exc).__name__}: {exc}"))
            return {
                "answer": fallback.answer,
                "sources": fallback.sources or sources,
                "mode": "langgraph_local_after_llm_error",
                "llm_error": str(exc),
            }

    def _reflect(self, state: AssistantGraphState) -> dict[str, Any]:
        writer = get_stream_writer()
        sources = state.get("sources", [])
        answer = state.get("answer", "")
        selected_item_id = state.get("selected_item_id", "")
        warnings: list[str] = []
        if selected_item_id and not any(source.get("item_id") == selected_item_id for source in sources):
            warnings.append("selected item was not present in returned sources")
        if not sources:
            warnings.append("answer has no retrieval sources")
        if len(answer.strip()) < 80:
            warnings.append("answer is short")
        reflection = {"status": "warning" if warnings else "ok", "warnings": warnings}
        evaluation = evaluate_assistant_response(
            question=state["question"],
            answer=answer,
            sources=sources,
            llm_enabled=_llm_enabled(),
        )
        writer(
            {
                "type": "reflection",
                "stage": "reflect",
                "status": reflection["status"],
                "message": "Reflection complete",
                "warnings": warnings,
            }
        )
        return {"reflection": reflection, "evaluation": evaluation}


def _progress(stage: str, status: str, message: str) -> dict[str, str]:
    return {"type": "progress", "stage": stage, "status": status, "message": message}


def _llm_enabled() -> bool:
    return bool(os.getenv("DASHSCOPE_API_KEY", "")) and os.getenv("ENABLE_LLM_ANALYSIS", "false").lower() in {
        "1",
        "true",
        "yes",
    }


def _chat_model() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "qwen-plus"),
        api_key=os.getenv("DASHSCOPE_API_KEY", ""),
        base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        temperature=0.2,
        streaming=True,
        timeout=float(os.getenv("LLM_STREAM_TIMEOUT_SECONDS", "60")),
        max_retries=0,
    )


def _answer_messages(
    question: str,
    retrieved: list[RagSearchResult],
    tool_results: list[dict[str, Any]],
) -> list[SystemMessage | HumanMessage]:
    context = [
        {
            "title": result.chunk.title,
            "item_id": result.chunk.item_id,
            "kind": result.chunk.kind,
            "url": result.chunk.url,
            "score": round(result.score, 4),
            "text": result.chunk.text[:1600],
        }
        for result in retrieved[:8]
    ]
    return [
        SystemMessage(
            content=(
                "You are a rigorous personalized research intelligence assistant. "
                "Answer in the user's language. Use only the provided RAG evidence and MCP tool results. "
                "If evidence is insufficient, say what is missing. Do not invent papers, repos, metrics, or experiments."
            )
        ),
        HumanMessage(
            content=json.dumps(
                {
                    "question": question,
                    "rag_evidence": context,
                    "mcp_tool_results": tool_results,
                    "answer_requirements": [
                        "Be specific and grounded.",
                        "Mention the selected paper or repo when the user selected one.",
                        "For repo or baseline questions, discuss suitability, risks, and next checks.",
                    ],
                },
                ensure_ascii=False,
            )
        ),
    ]


def _chunk_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return ""


def _tool_result_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)
