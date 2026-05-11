from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from research_intel.assistant_context import content_payloads, selected_item_result
from research_intel.rag import RagIndex
from research_intel.storage import JsonStore


def create_server(project_root: Path | str | None = None) -> FastMCP:
    root = Path(project_root or os.getenv("RESEARCH_INTEL_ROOT", ".")).resolve()
    store = JsonStore(root)
    app = FastMCP("research-intel-local")

    @app.tool()
    def get_selected_item_context(item_id: str) -> str:
        """Return detailed context for a selected paper/repo/tool from the latest report."""

        report = _load_report(store)
        payloads = content_payloads(root, report)
        selected = selected_item_result(report, payloads, item_id)
        if selected is None:
            return json.dumps({"found": False, "item_id": item_id}, ensure_ascii=False)
        return json.dumps(
            {
                "found": True,
                "item_id": item_id,
                "title": selected.chunk.title,
                "kind": selected.chunk.kind,
                "url": selected.chunk.url,
                "context": selected.chunk.text,
            },
            ensure_ascii=False,
        )

    @app.tool()
    def search_latest_report(query: str, selected_item_id: str = "", limit: int = 5) -> str:
        """Search the latest report and candidates with the local RAG index."""

        report = _load_report(store)
        payloads = content_payloads(root, report)
        index = RagIndex.from_report(report, payloads)
        results = index.search(query, selected_item_id=selected_item_id or None, limit=limit)
        return json.dumps(
            [
                {
                    "item_id": result.chunk.item_id,
                    "title": result.chunk.title,
                    "kind": result.chunk.kind,
                    "url": result.chunk.url,
                    "score": round(result.score, 4),
                    "preview": result.chunk.text[:500],
                }
                for result in results
            ],
            ensure_ascii=False,
        )

    @app.tool()
    def list_recommended_actions(limit: int = 5) -> str:
        """Return recommended actions from the latest daily research report."""

        report = _load_report(store)
        actions = [str(action) for action in report.get("actions", []) if action][:limit]
        return json.dumps({"actions": actions}, ensure_ascii=False)

    return app


def _load_report(store: JsonStore) -> dict[str, Any]:
    try:
        return store.load_report_json("latest")
    except FileNotFoundError:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Research Intelligence MCP tool server")
    parser.add_argument("--root", default=os.getenv("RESEARCH_INTEL_ROOT", "."), help="Project root")
    args = parser.parse_args()
    create_server(args.root).run(transport="stdio")


if __name__ == "__main__":
    main()
