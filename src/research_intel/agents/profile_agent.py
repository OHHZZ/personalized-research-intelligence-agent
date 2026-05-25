from __future__ import annotations

import re
from pathlib import Path

from research_intel.agents.base import BaseAgent
from research_intel.models import UserProfile, utc_now_iso
from research_intel.storage import JsonStore

_PROFILE_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


def validate_profile_id(profile_id: str) -> str:
    if not _PROFILE_ID_RE.match(profile_id):
        raise ValueError(f"Invalid profile id {profile_id!r}. Use letters, digits, _ or - (max 64 chars).")
    return profile_id


class ProfileAgent(BaseAgent):
    name = "profile-agent"

    def __init__(self, store: JsonStore) -> None:
        self.store = store

    def load_or_create(self, profile_id: str) -> UserProfile:
        validate_profile_id(profile_id)
        path = Path(self.store.profile_dir / f"{profile_id}.json")
        if path.exists():
            return self.store.load_profile(profile_id)

        profile = UserProfile(
            user_id=profile_id,
            display_name="",
            research_domains=[],
            methods=[],
            applications=[],
            preferred_content=["paper", "repo", "benchmark", "tool", "article"],
            excluded_topics=[],
            current_goals=[],
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
