from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_intel.rag import RagIndex


class RagIndexTest(unittest.TestCase):
    def test_search_returns_repo_for_baseline_question(self) -> None:
        report = {
            "top_repos": [
                {
                    "item_id": "repo_video_baseline",
                    "title": "VideoEditFlow",
                    "content_type": "repo",
                    "url": "https://github.com/example/videoeditflow",
                    "technical_core": "A controllable video editing pipeline with reproducible demo scripts.",
                    "limitations": ["small test coverage"],
                    "possible_actions": ["run the demo as a baseline"],
                    "practical_utility": 7,
                    "reproducibility": 6,
                }
            ]
        }
        candidates = [
            {
                "item_id": "repo_video_baseline",
                "content_type": "repo",
                "title": "VideoEditFlow",
                "url": "https://github.com/example/videoeditflow",
                "source": "github",
                "summary": "Controllable video editing baseline with demo and install instructions.",
                "tags": ["video editing", "baseline"],
                "metrics": {"stars": 500},
                "technical_signals": {"has_examples": True},
            }
        ]

        index = RagIndex.from_report(report, candidates, dimensions=128)
        results = index.search("which repo is suitable as a video editing baseline?", limit=3)

        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].chunk.item_id, "repo_video_baseline")
        self.assertGreater(results[0].dense_score, 0)
        self.assertGreater(results[0].keyword_score, 0)

    def test_selected_item_boosts_matching_chunk(self) -> None:
        report = {}
        candidates = [
            {
                "item_id": "paper_a",
                "content_type": "paper",
                "title": "Temporal Consistency Benchmark",
                "url": "https://arxiv.org/abs/0000.00001",
                "source": "arxiv",
                "summary": "Evaluation for temporal consistency in generated video.",
            },
            {
                "item_id": "repo_b",
                "content_type": "repo",
                "title": "Image Utility Repo",
                "url": "https://github.com/example/image",
                "source": "github",
                "summary": "Image processing utilities.",
            },
        ]

        index = RagIndex.from_report(report, candidates, dimensions=128)
        results = index.search("tell me about this item", selected_item_id="paper_a", limit=1)

        self.assertEqual(results[0].chunk.item_id, "paper_a")


if __name__ == "__main__":
    unittest.main()
