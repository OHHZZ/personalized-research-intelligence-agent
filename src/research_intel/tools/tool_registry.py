"""Simple function-based tool registry for pipeline agents.

Tools are registered with ``@ToolRegistry.register("name")`` and called via
``ToolRegistry.call("name", **kwargs)``.  All calls are best-effort: any
exception is caught and ``None`` is returned so agents are never blocked by a
failing tool.
"""
from __future__ import annotations

from typing import Any, Callable


class ToolNotFoundError(KeyError):
    pass


class ToolRegistry:
    """Class-level registry mapping tool names to callable functions."""

    _tools: dict[str, Callable[..., Any]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator that registers a function under *name*."""

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            cls._tools[name] = fn
            return fn

        return decorator

    @classmethod
    def call(cls, name: str, **kwargs: Any) -> Any:
        """Call a registered tool by name.

        Returns ``None`` on any error (not found or runtime exception) so that
        callers treat tools as optional enrichment, not hard requirements.
        """
        if name not in cls._tools:
            return None
        try:
            return cls._tools[name](**kwargs)
        except Exception:
            return None

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name in cls._tools

    @classmethod
    def list_tools(cls) -> list[str]:
        return sorted(cls._tools)
