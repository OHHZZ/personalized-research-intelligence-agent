"""Tool registry and built-in tools for pipeline agents.

Import this package to ensure all built-in tools are registered:

    from research_intel.tools import paper_tools, repo_tools  # noqa: F401
"""
from research_intel.tools.tool_registry import ToolNotFoundError, ToolRegistry

__all__ = ["ToolRegistry", "ToolNotFoundError"]
