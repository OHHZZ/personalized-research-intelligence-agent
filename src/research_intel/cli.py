from __future__ import annotations

import argparse
import json
from pathlib import Path

from research_intel.agents.repo_qa_agent import RepoQAAgent
from research_intel.config import load_dotenv
from research_intel.models import ContentItem, ContentType
from research_intel.pipeline import DailyResearchPipeline
from research_intel.rag import PgVectorStore
from research_intel.storage import JsonStore
from research_intel.web_server import serve


def main() -> None:
    parser = argparse.ArgumentParser(description="Personalized Research Intelligence Agent")
    parser.add_argument("--root", default=".", help="Project root path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    daily = subparsers.add_parser("run-daily", help="Run the daily research intelligence pipeline")
    daily.add_argument("--profile", default="default_user", help="Profile id")
    daily.add_argument("--report", default="latest", help="Report file stem")
    daily.add_argument(
        "--source",
        choices=["sample", "live", "hybrid"],
        default="hybrid",
        help="Candidate source mode. hybrid uses live sources and falls back to sample data.",
    )

    ask = subparsers.add_parser("ask-repo", help="Ask a question about a discovered repo")
    ask.add_argument("--repo-id", required=True, help="Repository item id from sample data")
    ask.add_argument("--question", required=True, help="Question to ask")

    web = subparsers.add_parser("serve-web", help="Start the local web app")
    web.add_argument("--host", default="127.0.0.1", help="Host to bind")
    web.add_argument("--port", type=int, default=8765, help="Port to bind")

    pgvector = subparsers.add_parser("init-pgvector", help="Initialize PostgreSQL + pgvector table from environment")
    pgvector.add_argument("--dimensions", type=int, default=384, help="Fallback embedding dimensions if index is not built yet")

    args = parser.parse_args()
    root = Path(args.root).resolve()
    load_dotenv(root)

    if args.command == "run-daily":
        result = DailyResearchPipeline(root).run(
            profile_id=args.profile,
            report_stem=args.report,
            source_mode=args.source,
        )
        print(f"Report JSON: {result.json_path}")
        print(f"Report Markdown: {result.markdown_path}")
        print("")
        print(result.report.markdown)
        return

    if args.command == "ask-repo":
        store = JsonStore(root)
        candidate_path = root / "data" / "runs" / "latest_candidates.json"
        if candidate_path.exists():
            items = [
                ContentItem.from_dict(item)
                for item in json.loads(candidate_path.read_text(encoding="utf-8"))
            ]
        else:
            items = store.load_content_items()
        repos = {
            item.item_id: item
            for item in items
            if item.content_type == ContentType.REPO
        }
        repo = repos.get(args.repo_id)
        if repo is None:
            known = ", ".join(sorted(repos))
            raise SystemExit(f"Unknown repo id `{args.repo_id}`. Known repo ids: {known}")
        print(RepoQAAgent().answer(repo, args.question))
        return

    if args.command == "serve-web":
        serve(root, host=args.host, port=args.port)
        return

    if args.command == "init-pgvector":
        store = PgVectorStore.from_env()
        if store is None:
            raise SystemExit("PGVECTOR_DSN is not configured in .env.")
        store.initialize(args.dimensions)
        print(f"Initialized pgvector table `{store.config.table}`.")
        return


if __name__ == "__main__":
    main()
