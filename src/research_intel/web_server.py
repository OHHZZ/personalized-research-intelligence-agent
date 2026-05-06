from __future__ import annotations

import json
import mimetypes
from uuid import uuid4
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from research_intel.agents.research_assistant_agent import ResearchAssistantAgent
from research_intel.agents.repo_qa_agent import RepoQAAgent
from research_intel.models import ContentItem, ContentType, FeedbackEvent, to_plain_dict
from research_intel.pipeline import DailyResearchPipeline
from research_intel.storage import JsonStore


STATIC_DIR = Path(__file__).resolve().parent / "web" / "static"


class ResearchIntelHandler(BaseHTTPRequestHandler):
    server_version = "ResearchIntelWeb/0.1"

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
                self._send_json(self.store.load_report_json(report_stem))
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
        self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/run":
            payload = self._read_json_body()
            profile_id = str(payload.get("profile", "default_user"))
            report_stem = str(payload.get("report", "latest"))
            source_mode = str(payload.get("source", "hybrid"))
            try:
                result = DailyResearchPipeline(self.project_root).run(
                    profile_id=profile_id,
                    report_stem=report_stem,
                    source_mode=source_mode,
                )
                self._send_json(
                    {
                        "report": to_plain_dict(result.report),
                        "json_path": str(result.json_path),
                        "markdown_path": str(result.markdown_path),
                    }
                )
            except Exception as exc:
                self._send_json({"error": "pipeline_failed", "detail": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
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
            self._send_json({"answer": RepoQAAgent().answer(repo, question)})
            return

        if parsed.path == "/api/assistant":
            payload = self._read_json_body()
            question = str(payload.get("question", ""))
            selected_item_id = str(payload.get("item_id", "")) or None
            if not question:
                self._send_json({"error": "missing_question"}, HTTPStatus.BAD_REQUEST)
                return
            try:
                report = self.store.load_report_json("latest")
            except FileNotFoundError:
                report = {}
            candidates = [ContentItem.from_dict(item) for item in self._repo_search_payloads() if item.get("item_id")]
            answer = ResearchAssistantAgent().answer(question, report, candidates, selected_item_id=selected_item_id)
            self._send_json({"answer": answer})
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
        self.end_headers()
        self.wfile.write(body)

    def _load_candidates_json(self) -> list[dict[str, object]]:
        path = self.project_root / "data" / "runs" / "latest_candidates.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return [to_plain_dict(item) for item in self.store.load_content_items()]

    def _find_repo(self, repo_id: str) -> ContentItem | None:
        for item in self._repo_search_payloads():
            if item.get("item_id") == repo_id and item.get("content_type") == ContentType.REPO.value:
                return ContentItem.from_dict(item)
        return None

    def _repo_search_payloads(self) -> list[dict[str, object]]:
        payloads: list[dict[str, object]] = []
        payloads.extend(self._load_candidates_json())
        try:
            report = self.store.load_report_json("latest")
        except FileNotFoundError:
            report = {}
        for item in report.get("candidates", []):
            if isinstance(item, dict):
                payloads.append(item)
        for item in report.get("top_repos", []):
            if isinstance(item, dict):
                payloads.append(
                    {
                        "item_id": item.get("item_id", ""),
                        "content_type": "repo",
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "source": "report",
                        "summary": item.get("technical_core", ""),
                        "tags": [],
                        "authors": [],
                        "metrics": {},
                        "technical_signals": {
                            "technical_core": item.get("technical_core", ""),
                            "baseline_ready": item.get("practical_utility", 0) >= 5,
                            "has_examples": "demo" in str(item.get("technical_core", "")).lower(),
                            "has_tests": False,
                            "has_license": False,
                            "last_commit_days": 999,
                        },
                        "links": {},
                        "raw": {},
                    }
                )
        return payloads

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
