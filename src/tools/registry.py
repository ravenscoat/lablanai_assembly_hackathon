"""
Tool registry: maps tool names to factory functions.
Tools are registered by name and resolved at agent creation time.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Type: A factory function that takes (config, dependencies) and returns a tool callable
ToolFactory = Callable[..., Any]


class ToolRegistry:
    """
    Registry of all available tools, both platform-level and tenant-specific.

    Tools are registered by name. Each registration is a factory function
    that takes (TenantConfig, **deps) and returns a callable suitable
    for use as a function_tool on an Agent.
    """

    def __init__(self):
        self._tools: dict[str, ToolFactory] = {}
        self._register_defaults()

    def register(self, name: str, factory: ToolFactory) -> None:
        """Register a tool factory by name."""
        self._tools[name] = factory
        logger.debug(f"Registered tool: {name}")

    def resolve(self, name: str) -> ToolFactory | None:
        """Look up a tool factory by name."""
        factory = self._tools.get(name)
        if not factory:
            logger.warning(f"Unknown tool: {name}")
        return factory

    def has(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return sorted(self._tools.keys())

    def _register_defaults(self) -> None:
        """Register built-in platform tools."""
        try:
            from tools.platform.search_kb import create_search_kb_tool

            self.register("search_knowledge_base", create_search_kb_tool)
        except ImportError:
            pass

        try:
            from tools.platform.escalate import (
                create_confirm_escalation_tool,
                create_request_escalation_tool,
            )

            self.register("request_escalation", create_request_escalation_tool)
            self.register("confirm_escalation", create_confirm_escalation_tool)
        except ImportError:
            pass

        try:
            from tools.platform.get_time import create_time_tool

            self.register("get_current_time", create_time_tool)
        except ImportError:
            pass

        try:
            from tools.platform.goodbye import create_goodbye_tool

            self.register("end_call", create_goodbye_tool)
        except ImportError:
            pass

        try:
            from tools.platform.start_flow import create_start_flow_tool

            self.register("start_flow", create_start_flow_tool)
        except ImportError:
            pass

        try:
            from tools.platform.select_language import create_select_language_tool

            self.register("select_language", create_select_language_tool)
        except ImportError:
            pass

        try:
            from tools.platform.collect_csat import create_collect_csat_tool

            self.register("collect_csat", create_collect_csat_tool)
        except ImportError:
            pass

    def register_tenant_tools(self) -> None:
        """Register tenant-specific tools (call after platform tools)."""
        try:
            from tools.tenant.example_tools import register_example_tools

            register_example_tools(self)
        except ImportError:
            pass


_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """Get singleton tool registry instance."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
        _registry.register_tenant_tools()
    return _registry
