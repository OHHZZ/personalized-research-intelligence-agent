from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_intel.agents.filtering_agent import FilteringAgent
from research_intel.models import FilterStatus
from research_intel.langgraph_pipeline import LangGraphDailyPipeline
from research_intel.storage import JsonStore


class PipelineTest(unittest.TestCase):
    def test_daily_pipeline_generates_report(self) -> None:
        result = LangGraphDailyPipeline(ROOT).run(
            profile_id="default_user",
            report_stem="test_latest",
            source_mode="sample",
        )

        self.assertTrue(result.markdown_path.exists())
        self.assertTrue(result.json_path.exists())
        self.assertIn("Top Papers", result.report.markdown)
        self.assertGreaterEqual(len(result.report.actions), 3)
        self.assertGreater(len(result.report.filter_decisions), 0)
        self.assertGreater(len(result.report.candidates), 0)

    def test_filtering_rejects_low_quality_prompt_collection(self) -> None:
        store = JsonStore(ROOT)
        profile = store.load_profile("default_user")
        items = store.load_content_items()
        decisions = {decision.item_id: decision for decision in FilteringAgent().filter(profile, items)}

        self.assertEqual(decisions["repo_prompt_pack_video_ai"].status, FilterStatus.REJECT)
        self.assertEqual(decisions["paper_weak_video_generation_note"].status, FilterStatus.REJECT)
        self.assertIn(
            decisions["paper_temporal_control_diffusion"].status,
            {FilterStatus.CANDIDATE, FilterStatus.HIGH_PRIORITY},
        )


if __name__ == "__main__":
    unittest.main()
