from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_intel.assistant_context import (
    content_items_from_payloads,
    content_payloads,
    ensure_selected_result,
    selected_item_result,
)
from research_intel.agents.research_assistant_agent import ResearchAssistantAgent


class AssistantContextTest(unittest.TestCase):
    def test_selected_top_paper_context_is_available_without_candidate_payload(self) -> None:
        report = {
            "top_papers": [
                {
                    "item_id": "paper_selected",
                    "content_type": "paper",
                    "title": "Selected Paper",
                    "url": "https://doi.org/10.0000/test",
                    "technical_core": "A controlled video editing method.",
                    "why_it_matters": "It has a relevant evaluation protocol.",
                    "relation_to_user": "Strong match to the configured profile.",
                    "limitations": ["no public code"],
                    "possible_actions": ["read the evaluation section"],
                }
            ],
            "candidates": [],
        }
        payloads = content_payloads(ROOT, report)
        selected = selected_item_result(report, payloads, "paper_selected")

        self.assertIsNotNone(selected)
        self.assertEqual(selected.chunk.item_id, "paper_selected")
        self.assertEqual(selected.chunk.kind, "selected_paper")
        self.assertIn("Selected Paper", selected.chunk.title)
        self.assertIn("controlled video editing", selected.chunk.text)

    def test_selected_top_repo_context_is_available_without_candidate_payload(self) -> None:
        report = {
            "top_repos": [
                {
                    "item_id": "repo_selected",
                    "content_type": "repo",
                    "title": "selected/repo",
                    "url": "https://github.com/selected/repo",
                    "technical_core": "Runnable baseline with examples.",
                    "practical_utility": 7,
                    "reproducibility": 6,
                }
            ],
            "candidates": [],
        }
        payloads = content_payloads(ROOT, report)
        selected = ensure_selected_result(report, payloads, "repo_selected", [])
        items = content_items_from_payloads(payloads)

        self.assertEqual(selected[0].chunk.item_id, "repo_selected")
        self.assertEqual(selected[0].chunk.kind, "selected_repo")
        self.assertTrue(any(item.item_id == "repo_selected" for item in items))

    def test_legacy_assistant_keeps_selected_report_context_without_candidate(self) -> None:
        report = {
            "top_papers": [
                {
                    "item_id": "paper_selected",
                    "content_type": "paper",
                    "title": "Selected Paper",
                    "url": "https://doi.org/10.0000/test",
                    "technical_core": "A controlled video editing method.",
                    "why_it_matters": "It has a relevant evaluation protocol.",
                    "relation_to_user": "Strong match to the configured profile.",
                    "limitations": ["no public code"],
                }
            ],
        }
        answer = ResearchAssistantAgent().answer(
            "分析一下这篇论文",
            report,
            [],
            selected_item_id="paper_selected",
            retrieved=[],
        )

        self.assertGreater(len(answer.sources), 0)
        self.assertEqual(answer.sources[0]["item_id"], "paper_selected")


if __name__ == "__main__":
    unittest.main()
