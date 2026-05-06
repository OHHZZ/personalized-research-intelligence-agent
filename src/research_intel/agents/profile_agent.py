from __future__ import annotations

from pathlib import Path

from research_intel.agents.base import BaseAgent
from research_intel.models import UserProfile, utc_now_iso
from research_intel.storage import JsonStore


class ProfileAgent(BaseAgent):
    name = "profile-agent"

    def __init__(self, store: JsonStore) -> None:
        self.store = store

    def load_or_create(self, profile_id: str = "default_user") -> UserProfile:
        path = Path(self.store.profile_dir / f"{profile_id}.json")
        if path.exists():
            return self.store.load_profile(profile_id)

        profile = UserProfile(
            user_id=profile_id,
            display_name="Researcher",
            research_domains=[
                "video generation",
                "controllable video editing",
                "multimodal agents",
            ],
            methods=[
                "diffusion models",
                "evaluation benchmark",
                "retrieval augmented generation",
            ],
            applications=[
                "AI drawing",
                "AI video editing",
                "academic writing",
            ],
            preferred_content=[
                "paper",
                "repo",
                "benchmark",
                "tool",
                "article",
            ],
            excluded_topics=[
                "prompt collection",
                "marketing only",
                "thin readme",
            ],
            current_goals=[
                "find publishable research gaps",
                "find baseline projects",
                "track open-source tools",
            ],
        )
        self.store.save_profile(profile)
        return profile

    def update_from_feedback(
        self,
        profile: UserProfile,
        liked_tags: list[str] | None = None,
        disliked_tags: list[str] | None = None,
    ) -> UserProfile:
        liked_tags = liked_tags or []
        disliked_tags = disliked_tags or []

        for tag in liked_tags:
            normalized = tag.lower().strip()
            profile.feedback_weights[normalized] = profile.feedback_weights.get(normalized, 0.0) + 0.5

        for tag in disliked_tags:
            normalized = tag.lower().strip()
            profile.feedback_weights[normalized] = profile.feedback_weights.get(normalized, 0.0) - 0.7

        profile.updated_at = utc_now_iso()
        self.store.save_profile(profile)
        return profile
