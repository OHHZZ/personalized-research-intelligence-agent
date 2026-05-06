from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AgentContext:
    profile_id: str
    run_id: str


class BaseAgent:
    name = "base-agent"

    def log_prefix(self, context: AgentContext | None = None) -> str:
        if context is None:
            return f"[{self.name}]"
        return f"[{self.name}:{context.run_id}]"

