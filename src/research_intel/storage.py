from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research_intel.models import ContentItem, DailyReport, FeedbackEvent, UserProfile, to_plain_dict


class JsonStore:
    """Small file-backed store for the local MVP."""

    def __init__(self, project_root: Path | str | None = None) -> None:
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.data_dir = self.project_root / "data"
        self.profile_dir = self.data_dir / "profiles"
        self.samples_dir = self.data_dir / "samples"
        self.runs_dir = self.data_dir / "runs"
        self.feedback_dir = self.data_dir / "feedback"
        self.reports_dir = self.project_root / "reports"

    def load_profile(self, profile_id: str) -> UserProfile:
        path = self.profile_dir / f"{profile_id}.json"
        payload = self._read_json(path)
        return UserProfile(**payload)

    def save_profile(self, profile: UserProfile) -> Path:
        path = self.profile_dir / f"{profile.user_id}.json"
        self._write_json(path, to_plain_dict(profile))
        return path

    def load_content_items(self, sample_name: str = "content_items") -> list[ContentItem]:
        path = self.samples_dir / f"{sample_name}.json"
        payload = self._read_json(path)
        return [ContentItem.from_dict(item) for item in payload]

    def save_content_items(self, items: list[ContentItem], stem: str = "latest_candidates") -> Path:
        path = self.runs_dir / f"{stem}.json"
        self._write_json(path, to_plain_dict(items))
        return path

    def save_run_json(self, stem: str, payload: Any) -> Path:
        path = self.runs_dir / f"{stem}.json"
        self._write_json(path, to_plain_dict(payload))
        return path

    def load_run_json(self, stem: str) -> Any:
        return self._read_json(self.runs_dir / f"{stem}.json")

    def append_feedback(self, event: FeedbackEvent) -> Path:
        path = self.feedback_dir / f"{event.profile_id}.json"
        events: list[dict[str, Any]] = []
        if path.exists():
            events = self._read_json(path)
        events.append(to_plain_dict(event))
        self._write_json(path, events)
        return path

    def load_feedback(self, profile_id: str) -> list[dict[str, Any]]:
        path = self.feedback_dir / f"{profile_id}.json"
        if not path.exists():
            return []
        return self._read_json(path)

    def load_report_json(self, stem: str = "latest") -> dict[str, Any]:
        return self._read_json(self.reports_dir / f"{stem}.json")

    def save_report(self, report: DailyReport, stem: str = "latest") -> tuple[Path, Path]:
        json_path = self.reports_dir / f"{stem}.json"
        markdown_path = self.reports_dir / f"{stem}.md"
        self._write_json(json_path, to_plain_dict(report))
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(report.markdown, encoding="utf-8")
        return json_path, markdown_path

    def _read_json(self, path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
