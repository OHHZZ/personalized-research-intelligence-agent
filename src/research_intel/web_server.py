from __future__ import annotations

import asyncio
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator
from uuid import uuid4

from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from research_intel.agents.research_assistant_agent import ResearchAssistantAgent
from research_intel.agents.repo_qa_agent import RepoQAAgent
from research_intel.assistant_context import content_items_from_payloads, content_payloads
from research_intel.evaluation import evaluate_assistant_response
from research_intel.llm import QwenChatClient
from research_intel.models import ContentItem, ContentType, DailyReport, FeedbackEvent, to_plain_dict
from research_intel.langgraph_pipeline import LangGraphDailyPipeline
from research_intel.rag import RagIndex, create_embedding_model, pgvector_health, sync_pgvector_from_env
from research_intel.storage import JsonStore

try:
    from research_intel.agents.langgraph_assistant import LangGraphAssistant
except ImportError:
    LangGraphAssistant = None  # type: ignore[assignment]


STATIC_DIR = Path(__file__).resolve().parent / "web" / "static"

# Module-level state — set by serve() before uvicorn starts
_project_root: Path = Path(".")
_store: JsonStore | None = None


def _get_store() -> JsonStore:
    if _store is None:
        raise RuntimeError("Server not initialized — call serve() first")
    return _store


def _make_pipeline(project_root: Path) -> LangGraphDailyPipeline:
    return LangGraphDailyPipeline(project_root)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Research Intelligence", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(STATIC_DIR)), name="assets")


# ---------------------------------------------------------------------------
# Request body schemas
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    profile: str = "default_user"
    report: str = "latest"
    source: str = "hybrid"


class ProfileUpdateRequest(BaseModel):
    user_id: str = "default_user"
    display_name: str | None = None
    research_domains: list[str] | None = None
    methods: list[str] | None = None
    applications: list[str] | None = None
    preferred_content: list[str] | None = None
    excluded_topics: list[str] | None = None
    technical_level: str | None = None
    current_goals: list[str] | None = None


class RepoQARequest(BaseModel):
    repo_id: str = ""
    question: str = ""


class AssistantRequest(BaseModel):
    question: str = ""
    item_id: str = ""


class FeedbackRequest(BaseModel):
    profile_id: str = "default_user"
    item_id: str = ""
    action: str = ""
    note: str = ""


# ---------------------------------------------------------------------------
# Static / index
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


# ---------------------------------------------------------------------------
# GET endpoints
# ---------------------------------------------------------------------------

@app.get("/api/profile")
async def get_profile(profile: str = Query(default="default_user")) -> JSONResponse:
    return JSONResponse(to_plain_dict(_get_store().load_profile(profile)))


@app.get("/api/report")
async def get_report(report: str = Query(default="latest")) -> JSONResponse:
    try:
        payload = _public_report_payload(_get_store().load_report_json(report))
        return JSONResponse(payload)
    except FileNotFoundError:
        return JSONResponse({"error": "report_not_found"}, status_code=404)


@app.get("/api/candidates")
async def get_candidates() -> JSONResponse:
    return JSONResponse(_load_candidates_json(_project_root, _get_store()))


@app.get("/api/feedback")
async def get_feedback(profile: str = Query(default="default_user")) -> JSONResponse:
    return JSONResponse(_get_store().load_feedback(profile))


@app.get("/api/health")
async def get_health() -> JSONResponse:
    return JSONResponse(_health_payload(_project_root, _get_store()))


# ---------------------------------------------------------------------------
# SSE: pipeline run stream
# ---------------------------------------------------------------------------

@app.get("/api/run/stream")
async def stream_run(
    profile: str = Query(default="default_user"),
    report: str = Query(default="latest"),
    source: str = Query(default="hybrid"),
) -> StreamingResponse:
    async def event_gen() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        run_id = uuid4().hex
        seq = [0]
        loop = asyncio.get_running_loop()

        def make_frame(event: str, payload: dict[str, object]) -> str:
            seq[0] += 1
            body = json.dumps(
                {"timestamp": datetime.now(timezone.utc).isoformat(), **payload, "run_id": run_id},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            return f"id: {run_id}:{seq[0]}\nevent: {event}\ndata: {body}\n\n"

        def enqueue(frame: str | None) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, frame)

        def run() -> None:
            try:
                enqueue(make_frame(
                    "run.started",
                    {"stage": "run", "status": "running", "message": "Pipeline started",
                     "profile": profile, "report": report, "source": source},
                ))
                result = LangGraphDailyPipeline(_project_root).run(
                    profile_id=profile,
                    report_stem=report,
                    source_mode=source,
                    progress=lambda p: enqueue(make_frame("run.progress", p)),
                )
                enqueue(make_frame(
                    "run.completed",
                    {"stage": "run", "status": "complete", "message": "Pipeline complete",
                     "report": _public_report_payload(result.report),
                     "json_path": str(result.json_path),
                     "markdown_path": str(result.markdown_path)},
                ))
            except Exception as exc:
                print(f"Pipeline stream failed: {type(exc).__name__}: {exc}")
                enqueue(make_frame(
                    "run.failed",
                    {"stage": "run", "status": "error", "message": "Pipeline failed",
                     "error": "pipeline_failed",
                     "detail": "Details are available in the backend console."},
                ))
            finally:
                enqueue(None)

        threading.Thread(target=run, daemon=True).start()

        while True:
            frame = await queue.get()
            if frame is None:
                break
            yield frame

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


# ---------------------------------------------------------------------------
# SSE: assistant answer stream
# ---------------------------------------------------------------------------

@app.get("/api/assistant/stream")
async def stream_assistant(
    question: str = Query(default=""),
    item_id: str = Query(default=""),
) -> StreamingResponse:
    selected_item_id = item_id or None
    question = question.strip()

    async def event_gen() -> AsyncGenerator[str, None]:
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        stream_id = uuid4().hex
        seq = [0]
        loop = asyncio.get_running_loop()

        def make_frame(event: str, payload: dict[str, object]) -> str:
            seq[0] += 1
            body = json.dumps(
                {"timestamp": datetime.now(timezone.utc).isoformat(), **payload, "stream_id": stream_id},
                ensure_ascii=False,
                separators=(",", ":"),
            )
            return f"id: {stream_id}:{seq[0]}\nevent: {event}\ndata: {body}\n\n"

        def enqueue(frame: str | None) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, frame)

        def run() -> None:
            try:
                if not question:
                    enqueue(make_frame(
                        "assistant.failed",
                        {"stage": "assistant", "status": "error",
                         "message": "Missing question", "error": "missing_question"},
                    ))
                    return
                enqueue(make_frame(
                    "assistant.started",
                    {"stage": "assistant", "status": "running", "message": "Assistant started",
                     "selected_item_id": selected_item_id or ""},
                ))
                if LangGraphAssistant is None:
                    final_response = _assistant_response(
                        question, selected_item_id, _project_root, _get_store(),
                        progress=lambda p: enqueue(make_frame("assistant.progress", p)),
                    )
                    enqueue(make_frame("assistant.completed", final_response))
                    return

                final_response: dict[str, Any] | None = None
                for event in LangGraphAssistant(_project_root).stream(question, selected_item_id):
                    event_type = str(event.get("type", "progress"))
                    if event_type == "token":
                        enqueue(make_frame("assistant.token", event))
                        continue
                    if event_type == "final":
                        final_response = dict(event)
                        continue
                    enqueue(make_frame("assistant.progress", event))

                if final_response is None:
                    raise RuntimeError("assistant graph did not produce a final response")
                trace_id = _append_assistant_trace(
                    _project_root,
                    {"trace_id": uuid4().hex, "question": question,
                     "selected_item_id": selected_item_id,
                     "mode": final_response.get("mode"),
                     "sources": final_response.get("sources", []),
                     "evaluation": final_response.get("evaluation", {}),
                     "reflection": final_response.get("reflection", {})},
                )
                final_response["trace_id"] = trace_id
                enqueue(make_frame("assistant.completed", final_response))
            except Exception as exc:
                enqueue(make_frame(
                    "assistant.failed",
                    {"stage": "assistant", "status": "error",
                     "message": "Assistant failed", "error": "assistant_failed", "detail": str(exc)},
                ))
            finally:
                enqueue(None)

        threading.Thread(target=run, daemon=True).start()

        while True:
            frame = await queue.get()
            if frame is None:
                break
            yield frame

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


# ---------------------------------------------------------------------------
# POST endpoints
# ---------------------------------------------------------------------------

@app.post("/api/run")
async def post_run(body: RunRequest) -> JSONResponse:
    try:
        result = _make_pipeline(_project_root).run(
            profile_id=body.profile,
            report_stem=body.report,
            source_mode=body.source,
        )
        return JSONResponse({
            "report": _public_report_payload(result.report),
            "json_path": str(result.json_path),
            "markdown_path": str(result.markdown_path),
        })
    except Exception as exc:
        print(f"Pipeline failed: {type(exc).__name__}: {exc}")
        return JSONResponse(
            {"error": "pipeline_failed", "detail": "Details are available in the backend console."},
            status_code=500,
        )


@app.post("/api/profile")
async def post_profile(body: ProfileUpdateRequest) -> JSONResponse:
    try:
        store = _get_store()
        profile = store.load_profile(body.user_id)
        for key in (
            "display_name", "research_domains", "methods", "applications",
            "preferred_content", "excluded_topics", "technical_level", "current_goals",
        ):
            value = getattr(body, key, None)
            if value is not None:
                setattr(profile, key, value)
        store.save_profile(profile)
        return JSONResponse(to_plain_dict(profile))
    except Exception as exc:
        return JSONResponse({"error": "profile_save_failed", "detail": str(exc)}, status_code=400)


@app.post("/api/repo-qa")
async def post_repo_qa(body: RepoQARequest) -> JSONResponse:
    repo = _find_repo(body.repo_id, _project_root, _get_store())
    if repo is None:
        return JSONResponse({"error": "repo_not_found"}, status_code=404)
    return JSONResponse({"answer": RepoQAAgent(_project_root).answer(repo, body.question)})


@app.post("/api/assistant")
async def post_assistant(body: AssistantRequest) -> JSONResponse:
    if not body.question:
        return JSONResponse({"error": "missing_question"}, status_code=400)
    try:
        selected_item_id = body.item_id or None
        response = _assistant_response(body.question, selected_item_id, _project_root, _get_store())
        return JSONResponse(response)
    except Exception as exc:
        return JSONResponse({"error": "assistant_failed", "detail": str(exc)}, status_code=500)


@app.post("/api/feedback")
async def post_feedback(body: FeedbackRequest) -> JSONResponse:
    if not body.item_id or not body.action:
        return JSONResponse({"error": "missing_feedback_fields"}, status_code=400)
    store = _get_store()
    event = FeedbackEvent(
        feedback_id=uuid4().hex,
        profile_id=body.profile_id,
        item_id=body.item_id,
        action=body.action,
        note=body.note,
    )
    store.append_feedback(event)
    _apply_feedback_to_profile(body.profile_id, body.item_id, body.action, _project_root, store)
    return JSONResponse(to_plain_dict(event))


# ---------------------------------------------------------------------------
# Business logic helpers (pure functions — no handler state)
# ---------------------------------------------------------------------------

def _public_report_payload(report: DailyReport | dict[str, Any]) -> dict[str, Any]:
    payload = to_plain_dict(report)
    source_errors = payload.get("source_errors", [])
    error_count = len(source_errors) if isinstance(source_errors, list) else 0
    payload["source_error_count"] = error_count
    payload["source_status"] = "partial" if error_count else "ok"
    payload["source_errors"] = []
    return payload


def _load_candidates_json(project_root: Path, store: JsonStore) -> list[dict[str, object]]:
    path = project_root / "data" / "runs" / "latest_candidates.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return [to_plain_dict(item) for item in store.load_content_items()]


def _find_repo(repo_id: str, project_root: Path, store: JsonStore) -> ContentItem | None:
    for item in _content_payloads(project_root, store):
        if item.get("item_id") == repo_id and item.get("content_type") == ContentType.REPO.value:
            return ContentItem.from_dict(item)
    return None


def _content_payloads(
    project_root: Path, store: JsonStore, report: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    try:
        current_report = report if report is not None else store.load_report_json("latest")
    except FileNotFoundError:
        current_report = {}
    return content_payloads(project_root, current_report)


def _load_or_build_rag_index(
    project_root: Path,
    report: dict[str, Any],
    candidates: list[dict[str, Any]],
    selected_item_id: str | None = None,
) -> RagIndex:
    path = project_root / "data" / "runs" / "latest_rag_index.json"
    required_item_ids = _required_rag_item_ids(report, selected_item_id)
    if _rag_index_is_fresh(project_root, path):
        try:
            index = RagIndex.load(path)
            if index.is_compatible_with_current_embedding() and _rag_index_has_items(index, required_item_ids):
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


def _required_rag_item_ids(report: dict[str, Any], selected_item_id: str | None) -> set[str]:
    required = {selected_item_id} if selected_item_id else set()
    for section in ("top_papers", "top_repos", "top_tools"):
        for item in report.get(section, []):
            if isinstance(item, dict) and item.get("item_id"):
                required.add(str(item["item_id"]))
    return {item_id for item_id in required if item_id}


def _rag_index_has_items(index: RagIndex, item_ids: set[str]) -> bool:
    if not item_ids:
        return True
    indexed = {chunk.item_id for chunk in index.chunks if chunk.item_id}
    return item_ids.issubset(indexed)


def _rag_index_is_fresh(project_root: Path, path: Path) -> bool:
    if not path.exists():
        return False
    source_paths = [
        project_root / "reports" / "latest.json",
        project_root / "data" / "runs" / "latest_candidates.json",
    ]
    source_mtime = max((p.stat().st_mtime for p in source_paths if p.exists()), default=0)
    return path.stat().st_mtime >= source_mtime


def _append_assistant_trace(project_root: Path, event: dict[str, Any]) -> str:
    trace_id = str(event.get("trace_id", uuid4().hex))
    path = project_root / "data" / "runs" / "assistant_traces.json"
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


def _assistant_response(
    question: str,
    selected_item_id: str | None,
    project_root: Path,
    store: JsonStore,
    progress: Any | None = None,
) -> dict[str, Any]:
    def emit(stage: str, status: str, message: str, **extra: object) -> None:
        if progress:
            progress({"stage": stage, "status": status, "message": message, **extra})

    emit("context", "running", "Loading latest report")
    try:
        report = store.load_report_json("latest")
    except FileNotFoundError:
        report = {}
    emit("context", "complete", "Report context loaded")

    emit("context", "running", "Loading candidate and selected-item metadata")
    candidate_payloads = _content_payloads(project_root, store, report)
    candidates = content_items_from_payloads(candidate_payloads)
    emit("context", "complete", f"Loaded {len(candidates)} candidate records")

    emit("rag", "running", "Building or loading RAG index")
    rag_index = _load_or_build_rag_index(project_root, report, candidate_payloads, selected_item_id=selected_item_id)
    retrieved = rag_index.search(question, selected_item_id=selected_item_id)
    emit("rag", "complete", f"Retrieved {len(retrieved)} context chunks", source_count=len(retrieved))

    assistant = ResearchAssistantAgent()
    emit("generation", "running", "Generating grounded answer")
    result = assistant.answer(
        question, report, candidates, selected_item_id=selected_item_id, retrieved=retrieved
    )
    emit("generation", "complete", "Answer generated", mode=result.mode)

    emit("evaluation", "running", "Evaluating answer grounding")
    evaluation = evaluate_assistant_response(
        question=question,
        answer=result.answer,
        sources=result.sources,
        llm_enabled=assistant.llm_client.enabled,
    )
    trace_id = _append_assistant_trace(
        project_root,
        {"trace_id": uuid4().hex, "question": question, "selected_item_id": selected_item_id,
         "mode": result.mode, "sources": result.sources, "evaluation": evaluation},
    )
    emit("evaluation", "complete", "Answer trace recorded", trace_id=trace_id)

    response = result.to_dict()
    response["evaluation"] = evaluation
    response["trace_id"] = trace_id
    return response


def _health_payload(project_root: Path, store: JsonStore) -> dict[str, Any]:
    qwen_client = QwenChatClient()
    try:
        embedding_model = create_embedding_model()
        embedding: dict[str, Any] = {
            "provider": embedding_model.provider,
            "model": embedding_model.model_name,
            "dimensions": embedding_model.dimensions,
            "status": "ok",
        }
    except Exception as exc:
        print(f"Embedding health check failed: {type(exc).__name__}: {exc}")
        embedding = {
            "provider": os.getenv("EMBEDDING_PROVIDER", "sentence_transformers"),
            "model": os.getenv("EMBEDDING_MODEL", ""),
            "status": "error",
        }

    live_error_count = 0
    try:
        report = store.load_report_json("latest")
        source_errors = report.get("source_errors", [])
        live_error_count = len(source_errors) if isinstance(source_errors, list) else 0
    except FileNotFoundError:
        pass

    pgvector = pgvector_health()
    if pgvector.get("status") == "error":
        print(f"pgvector health check failed: {pgvector.get('detail', 'unknown error')}")

    return {
        "llm": {"enabled": qwen_client.enabled, "model": qwen_client.model},
        "network": {"connector_timeout_seconds": os.getenv("CONNECTOR_TIMEOUT_SECONDS", "8")},
        "embedding": embedding,
        "pgvector": _public_pgvector_health(pgvector),
        "latest_live_error_count": live_error_count,
        "latest_live_status": "partial" if live_error_count else "ok",
    }


def _apply_feedback_to_profile(
    profile_id: str, item_id: str, action: str, project_root: Path, store: JsonStore
) -> None:
    weight_by_action = {
        "relevant": 0.5, "save": 0.4, "deeper": 0.35, "baseline": 0.45,
        "not_relevant": -0.8, "skip": -0.3,
    }
    weight = weight_by_action.get(action, 0.0)
    if weight == 0:
        return
    profile = store.load_profile(profile_id)
    item = next((i for i in _load_candidates_json(project_root, store) if i.get("item_id") == item_id), None)
    if not item:
        return
    for tag in item.get("tags", []):
        normalized = str(tag).lower().strip()
        if normalized:
            profile.feedback_weights[normalized] = round(
                profile.feedback_weights.get(normalized, 0.0) + weight, 3
            )
    store.save_profile(profile)


def _assistant_trace_limit() -> int:
    raw = os.getenv("ASSISTANT_TRACE_LIMIT", "100")
    try:
        return max(10, int(raw))
    except ValueError:
        return 100


def _public_pgvector_health(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(payload.get("enabled")),
        "status": str(payload.get("status", "unknown")),
        "table": str(payload.get("table", "")),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def serve(project_root: Path | str, host: str = "127.0.0.1", port: int = 8765) -> None:
    global _project_root, _store
    _project_root = Path(project_root).resolve()
    _store = JsonStore(_project_root)

    import uvicorn
    print(f"Research Intelligence web app: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
