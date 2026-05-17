from __future__ import annotations

import json
import mimetypes
import os
from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import ParseResult, parse_qs, urlparse

from research_intel.agents.research_assistant_agent import ResearchAssistantAgent
from research_intel.agents.repo_qa_agent import RepoQAAgent
from research_intel.assistant_context import content_items_from_payloads, content_payloads
from research_intel.evaluation import evaluate_assistant_response
from research_intel.llm import QwenChatClient
from research_intel.models import ContentItem, ContentType, DailyReport, FeedbackEvent, to_plain_dict
from research_intel.pipeline import DailyResearchPipeline

try:
    from research_intel.langgraph_pipeline import LangGraphDailyPipeline as _LGPipeline
except ImportError:
    _LGPipeline = None  # type: ignore[assignment,misc]


def _make_pipeline(project_root: Path) -> DailyResearchPipeline:
    if _LGPipeline is not None and os.getenv("USE_LANGGRAPH_PIPELINE", "false").lower() == "true":
        return _LGPipeline(project_root)  # type: ignore[return-value]
    return DailyResearchPipeline(project_root)
from research_intel.rag import RagIndex, create_embedding_model, pgvector_health, sync_pgvector_from_env
from research_intel.storage import JsonStore

try:
    from research_intel.agents.langgraph_assistant import LangGraphAssistant
except ImportError:
    LangGraphAssistant = None  # type: ignore[assignment]


STATIC_DIR = Path(__file__).resolve().parent / "web" / "static"


class ResearchIntelHandler(BaseHTTPRequestHandler):
    server_version = "ResearchIntelWeb/0.1"
    protocol_version = "HTTP/1.1"

    def do_OPTIONS(self) -> None:
        self._send_empty(HTTPStatus.NO_CONTENT)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_static(STATIC_DIR / "index.html")
            return
        if parsed.path.startswith("/assets/"):
            self._send_static(STATIC_DIR / parsed.path.removeprefix("/assets/"))
            return
        if parsed.path == "/api/profile":
            profile_id = parse_qs(parsed.query).get("profile", ["default_user"])[0]
            self._send_json(to_plain_dict(self.store.load_profile(profile_id)))
            return
        if parsed.path == "/api/report":
            report_stem = parse_qs(parsed.query).get("report", ["latest"])[0]
            try:
                self._send_json(self._public_report_payload(self.store.load_report_json(report_stem)))
            except FileNotFoundError:
                self._send_json({"error": "report_not_found"}, HTTPStatus.NOT_FOUND)
            return
        if parsed.path == "/api/candidates":
            self._send_json(self._load_candidates_json())
            return
        if parsed.path == "/api/feedback":
            profile_id = parse_qs(parsed.query).get("profile", ["default_user"])[0]
            self._send_json(self.store.load_feedback(profile_id))
            return
        if parsed.path == "/api/health":
            self._send_json(self._health_payload())
            return
        if parsed.path == "/api/run/stream":
            self._stream_pipeline_run(parsed)
            return
        if parsed.path == "/api/assistant/stream":
            self._stream_assistant_answer(parsed)
            return
        self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/run":
            payload = self._read_json_body()
            profile_id = str(payload.get("profile", "default_user"))
            report_stem = str(payload.get("report", "latest"))
            source_mode = str(payload.get("source", "hybrid"))
            try:
                result = _make_pipeline(self.project_root).run(
                    profile_id=profile_id,
                    report_stem=report_stem,
                    source_mode=source_mode,
                )
                self._send_json(
                    {
                        "report": self._public_report_payload(result.report),
                        "json_path": str(result.json_path),
                        "markdown_path": str(result.markdown_path),
                    }
                )
            except Exception as exc:
                print(f"Pipeline failed: {type(exc).__name__}: {exc}")
                self._send_json(
                    {"error": "pipeline_failed", "detail": "Details are available in the backend console."},
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            return

        if parsed.path == "/api/profile":
            payload = self._read_json_body()
            try:
                profile = self.store.load_profile(str(payload.get("user_id", "default_user")))
                for key in (
                    "display_name",
                    "research_domains",
                    "methods",
                    "applications",
                    "preferred_content",
                    "excluded_topics",
                    "technical_level",
                    "current_goals",
                ):
                    if key in payload:
                        setattr(profile, key, payload[key])
                self.store.save_profile(profile)
                self._send_json(to_plain_dict(profile))
            except Exception as exc:
                self._send_json({"error": "profile_save_failed", "detail": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        if parsed.path == "/api/repo-qa":
            payload = self._read_json_body()
            repo_id = str(payload.get("repo_id", ""))
            question = str(payload.get("question", ""))
            repo = self._find_repo(repo_id)
            if repo is None:
                self._send_json({"error": "repo_not_found"}, HTTPStatus.NOT_FOUND)
                return
            self._send_json({"answer": RepoQAAgent(self.project_root).answer(repo, question)})
            return

        if parsed.path == "/api/assistant":
            payload = self._read_json_body()
            question = str(payload.get("question", ""))
            selected_item_id = str(payload.get("item_id", "")) or None
            if not question:
                self._send_json({"error": "missing_question"}, HTTPStatus.BAD_REQUEST)
                return
            try:
                self._send_json(self._assistant_response(question, selected_item_id))
            except Exception as exc:
                self._send_json({"error": "assistant_failed", "detail": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return

        if parsed.path == "/api/feedback":
            payload = self._read_json_body()
            profile_id = str(payload.get("profile_id", "default_user"))
            item_id = str(payload.get("item_id", ""))
            action = str(payload.get("action", ""))
            note = str(payload.get("note", ""))
            if not item_id or not action:
                self._send_json({"error": "missing_feedback_fields"}, HTTPStatus.BAD_REQUEST)
                return
            event = FeedbackEvent(
                feedback_id=uuid4().hex,
                profile_id=profile_id,
                item_id=item_id,
                action=action,
                note=note,
            )
            self.store.append_feedback(event)
            self._apply_feedback_to_profile(profile_id, item_id, action)
            self._send_json(to_plain_dict(event))
            return

        self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)

    @property
    def project_root(self) -> Path:
        return self.server.project_root  # type: ignore[attr-defined]

    @property
    def store(self) -> JsonStore:
        return self.server.store  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: object) -> None:
        return

    def _read_json_body(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def _send_empty(self, status: HTTPStatus) -> None:
        self.send_response(status)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def _stream_pipeline_run(self, parsed: ParseResult) -> None:
        query = parse_qs(parsed.query)
        profile_id = query.get("profile", ["default_user"])[0]
        report_stem = query.get("report", ["latest"])[0]
        source_mode = query.get("source", ["hybrid"])[0]
        run_id = uuid4().hex
        sequence = 0

        self._send_sse_headers()

        def emit(event: str, payload: dict[str, object]) -> None:
            nonlocal sequence
            sequence += 1
            self._send_sse_event(
                event,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    **payload,
                    "run_id": run_id,
                },
                event_id=f"{run_id}:{sequence}",
            )

        try:
            emit(
                "run.started",
                {
                    "stage": "run",
                    "status": "running",
                    "message": "Pipeline started",
                    "profile": profile_id,
                    "report": report_stem,
                    "source": source_mode,
                },
            )
            result = DailyResearchPipeline(self.project_root).run(
                profile_id=profile_id,
                report_stem=report_stem,
                source_mode=source_mode,
                progress=lambda payload: emit("run.progress", payload),
            )
            emit(
                "run.completed",
                {
                    "stage": "run",
                    "status": "complete",
                    "message": "Pipeline complete",
                    "report": self._public_report_payload(result.report),
                    "json_path": str(result.json_path),
                    "markdown_path": str(result.markdown_path),
                },
            )
        except BrokenPipeError:
            return
        except Exception as exc:
            print(f"Pipeline stream failed: {type(exc).__name__}: {exc}")
            try:
                emit(
                    "run.failed",
                    {
                        "stage": "run",
                        "status": "error",
                        "message": "Pipeline failed",
                        "error": "pipeline_failed",
                        "detail": "Details are available in the backend console.",
                    },
                )
            except BrokenPipeError:
                return

    def _stream_assistant_answer(self, parsed: ParseResult) -> None:
        query = parse_qs(parsed.query)
        question = query.get("question", [""])[0].strip()
        selected_item_id = query.get("item_id", [""])[0] or None
        stream_id = uuid4().hex
        sequence = 0
        emit_lock = Lock()

        self._send_sse_headers()

        def emit(event: str, payload: dict[str, object]) -> None:
            nonlocal sequence
            with emit_lock:
                sequence += 1
                self._send_sse_event(
                    event,
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        **payload,
                        "stream_id": stream_id,
                    },
                    event_id=f"{stream_id}:{sequence}",
                )

        try:
            if not question:
                emit(
                    "assistant.failed",
                    {
                        "stage": "assistant",
                        "status": "error",
                        "message": "Missing question",
                        "error": "missing_question",
                    },
                )
                return
            emit(
                "assistant.started",
                {
                    "stage": "assistant",
                    "status": "running",
                    "message": "Assistant started",
                    "selected_item_id": selected_item_id or "",
                },
            )
            if LangGraphAssistant is None:
                final_response = self._assistant_response(
                    question,
                    selected_item_id,
                    progress=lambda payload: emit("assistant.progress", payload),
                )
                emit("assistant.completed", final_response)
                return

            final_response: dict[str, Any] | None = None
            for event in LangGraphAssistant(self.project_root).stream(question, selected_item_id):
                event_type = str(event.get("type", "progress"))
                if event_type == "token":
                    emit("assistant.token", event)
                    continue
                if event_type == "final":
                    final_response = dict(event)
                    continue
                emit("assistant.progress", event)

            if final_response is None:
                raise RuntimeError("assistant graph did not produce a final response")
            trace_id = self._append_assistant_trace(
                {
                    "trace_id": uuid4().hex,
                    "question": question,
                    "selected_item_id": selected_item_id,
                    "mode": final_response.get("mode"),
                    "sources": final_response.get("sources", []),
                    "evaluation": final_response.get("evaluation", {}),
                    "reflection": final_response.get("reflection", {}),
                }
            )
            final_response["trace_id"] = trace_id
            emit("assistant.completed", final_response)
        except BrokenPipeError:
            return
        except Exception as exc:
            try:
                emit(
                    "assistant.failed",
                    {
                        "stage": "assistant",
                        "status": "error",
                        "message": "Assistant failed",
                        "error": "assistant_failed",
                        "detail": str(exc),
                    },
                )
            except BrokenPipeError:
                return

    def _assistant_response(
        self,
        question: str,
        selected_item_id: str | None = None,
        progress: Any | None = None,
    ) -> dict[str, Any]:
        def emit(stage: str, status: str, message: str, **extra: object) -> None:
            if progress:
                progress({"stage": stage, "status": status, "message": message, **extra})

        emit("context", "running", "Loading latest report")
        try:
            report = self.store.load_report_json("latest")
        except FileNotFoundError:
            report = {}
        emit("context", "complete", "Report context loaded")

        emit("context", "running", "Loading candidate and selected-item metadata")
        candidate_payloads = self._content_payloads(report)
        candidates = content_items_from_payloads(candidate_payloads)
        emit("context", "complete", f"Loaded {len(candidates)} candidate records")

        emit("rag", "running", "Building or loading RAG index")
        rag_index = self._load_or_build_rag_index(report, candidate_payloads, selected_item_id=selected_item_id)
        retrieved = rag_index.search(question, selected_item_id=selected_item_id)
        emit("rag", "complete", f"Retrieved {len(retrieved)} context chunks", source_count=len(retrieved))

        assistant = ResearchAssistantAgent()
        emit("generation", "running", "Generating grounded answer")
        result = assistant.answer(
            question,
            report,
            candidates,
            selected_item_id=selected_item_id,
            retrieved=retrieved,
        )
        emit("generation", "complete", "Answer generated", mode=result.mode)

        emit("evaluation", "running", "Evaluating answer grounding")
        evaluation = evaluate_assistant_response(
            question=question,
            answer=result.answer,
            sources=result.sources,
            llm_enabled=assistant.llm_client.enabled,
        )
        trace_id = self._append_assistant_trace(
            {
                "trace_id": uuid4().hex,
                "question": question,
                "selected_item_id": selected_item_id,
                "mode": result.mode,
                "sources": result.sources,
                "evaluation": evaluation,
            }
        )
        emit("evaluation", "complete", "Answer trace recorded", trace_id=trace_id)

        response = result.to_dict()
        response["evaluation"] = evaluation
        response["trace_id"] = trace_id
        return response

    def _send_sse_headers(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def _send_sse_event(self, event: str, payload: dict[str, object], event_id: str | None = None) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        lines = []
        if event_id:
            lines.append(f"id: {event_id}")
        lines.append(f"event: {event}")
        lines.extend(f"data: {line}" for line in body.splitlines() or [""])
        frame = "\n".join(lines) + "\n\n"
        self.wfile.write(frame.encode("utf-8"))
        self.wfile.flush()

    def _send_static(self, path: Path) -> None:
        if not path.exists() or not path.is_file() or STATIC_DIR not in path.resolve().parents:
            self._send_json({"error": "static_not_found"}, HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if path.suffix == ".js":
            content_type = "application/javascript"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.end_headers()
        self.wfile.write(body)

    def _load_candidates_json(self) -> list[dict[str, object]]:
        path = self.project_root / "data" / "runs" / "latest_candidates.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return [to_plain_dict(item) for item in self.store.load_content_items()]

    def _public_report_payload(self, report: DailyReport | dict[str, Any]) -> dict[str, Any]:
        payload = to_plain_dict(report)
        source_errors = payload.get("source_errors", [])
        error_count = len(source_errors) if isinstance(source_errors, list) else 0
        payload["source_error_count"] = error_count
        payload["source_status"] = "partial" if error_count else "ok"
        payload["source_errors"] = []
        return payload

    def _find_repo(self, repo_id: str) -> ContentItem | None:
        for item in self._content_payloads():
            if item.get("item_id") == repo_id and item.get("content_type") == ContentType.REPO.value:
                return ContentItem.from_dict(item)
        return None

    def _content_payloads(self, report: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        try:
            current_report = report if report is not None else self.store.load_report_json("latest")
        except FileNotFoundError:
            current_report = {}
        return content_payloads(self.project_root, current_report)

    def _load_or_build_rag_index(
        self,
        report: dict[str, Any],
        candidates: list[dict[str, Any]],
        selected_item_id: str | None = None,
    ) -> RagIndex:
        path = self.project_root / "data" / "runs" / "latest_rag_index.json"
        required_item_ids = self._required_rag_item_ids(report, selected_item_id)
        if self._rag_index_is_fresh(path):
            try:
                index = RagIndex.load(path)
                if index.is_compatible_with_current_embedding() and self._rag_index_has_items(index, required_item_ids):
                    return index
            except Exception:
                pass
        index = RagIndex.from_report(report, candidates)
        index.save(path)
        try:
            sync_pgvector_from_env(index)
        except Exception:
            pass
        return index

    def _required_rag_item_ids(self, report: dict[str, Any], selected_item_id: str | None) -> set[str]:
        required = {selected_item_id} if selected_item_id else set()
        for section in ("top_papers", "top_repos", "top_tools"):
            for item in report.get(section, []):
                if isinstance(item, dict) and item.get("item_id"):
                    required.add(str(item["item_id"]))
        return {item_id for item_id in required if item_id}

    def _rag_index_has_items(self, index: RagIndex, item_ids: set[str]) -> bool:
        if not item_ids:
            return True
        indexed_item_ids = {chunk.item_id for chunk in index.chunks if chunk.item_id}
        return item_ids.issubset(indexed_item_ids)

    def _rag_index_is_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        source_paths = [
            self.project_root / "reports" / "latest.json",
            self.project_root / "data" / "runs" / "latest_candidates.json",
        ]
        source_mtime = max((item.stat().st_mtime for item in source_paths if item.exists()), default=0)
        return path.stat().st_mtime >= source_mtime

    def _append_assistant_trace(self, event: dict[str, Any]) -> str:
        trace_id = str(event.get("trace_id", uuid4().hex))
        path = self.project_root / "data" / "runs" / "assistant_traces.json"
        events: list[dict[str, Any]] = []
        if path.exists():
            try:
                events = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                events = []
        events.append(event)
        limit = _assistant_trace_limit()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(events[-limit:], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return trace_id

    def _health_payload(self) -> dict[str, Any]:
        qwen_client = QwenChatClient()
        try:
            embedding_model = create_embedding_model()
            embedding = {
                "provider": embedding_model.provider,
                "model": embedding_model.model_name,
                "dimensions": embedding_model.dimensions,
                "status": "ok",
            }
        except Exception as exc:
            print(f"Embedding health check failed: {type(exc).__name__}: {exc}")
            embedding = {
                "provider": os.getenv("EMBEDDING_PROVIDER", "local_hash"),
                "model": os.getenv("EMBEDDING_MODEL", ""),
                "status": "error",
            }

        live_error_count = 0
        try:
            report = self.store.load_report_json("latest")
            source_errors = report.get("source_errors", [])
            live_error_count = len(source_errors) if isinstance(source_errors, list) else 0
        except FileNotFoundError:
            pass

        pgvector = pgvector_health()
        if pgvector.get("status") == "error":
            print(f"pgvector health check failed: {pgvector.get('detail', 'unknown error')}")

        return {
            "llm": {
                "enabled": qwen_client.enabled,
                "model": qwen_client.model,
            },
            "network": {
                "connector_timeout_seconds": os.getenv("CONNECTOR_TIMEOUT_SECONDS", "8"),
            },
            "embedding": embedding,
            "pgvector": _public_pgvector_health(pgvector),
            "latest_live_error_count": live_error_count,
            "latest_live_status": "partial" if live_error_count else "ok",
        }

    def _apply_feedback_to_profile(self, profile_id: str, item_id: str, action: str) -> None:
        weight_by_action = {
            "relevant": 0.5,
            "save": 0.4,
            "deeper": 0.35,
            "baseline": 0.45,
            "not_relevant": -0.8,
            "skip": -0.3,
        }
        weight = weight_by_action.get(action, 0.0)
        if weight == 0:
            return
        profile = self.store.load_profile(profile_id)
        item = next((item for item in self._load_candidates_json() if item.get("item_id") == item_id), None)
        if not item:
            return
        for tag in item.get("tags", []):
            normalized = str(tag).lower().strip()
            if normalized:
                profile.feedback_weights[normalized] = round(profile.feedback_weights.get(normalized, 0.0) + weight, 3)
        self.store.save_profile(profile)


def serve(project_root: Path | str, host: str = "127.0.0.1", port: int = 8765) -> None:
    root = Path(project_root).resolve()
    server = ThreadingHTTPServer((host, port), ResearchIntelHandler)
    server.project_root = root  # type: ignore[attr-defined]
    server.store = JsonStore(root)  # type: ignore[attr-defined]
    print(f"Research Intelligence web app: http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _assistant_trace_limit() -> int:
    raw = os.getenv("ASSISTANT_TRACE_LIMIT", "100")
    try:
        return max(10, int(raw))
    except ValueError:
        return 100


def _proxy_summary() -> dict[str, str]:
    keys = ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "GIT_HTTP_PROXY", "GIT_HTTPS_PROXY"]
    summary: dict[str, str] = {}
    for key in keys:
        value = os.getenv(key)
        if value is None:
            continue
        if value == "":
            summary[key] = ""
            continue
        parsed = urlparse(value)
        if parsed.scheme and parsed.hostname:
            netloc = parsed.hostname
            if parsed.port:
                netloc = f"{netloc}:{parsed.port}"
            summary[key] = f"{parsed.scheme}://{netloc}"
        else:
            summary[key] = value
    return summary


def _public_pgvector_health(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(payload.get("enabled")),
        "status": str(payload.get("status", "unknown")),
        "table": str(payload.get("table", "")),
    }
