"""
Example tenant-specific tool factories.

This module shows the PATTERN for per-tenant custom tools: register named
factories in the ToolRegistry so they can be referenced from a tenant YAML's
`tools.tenant` list. The factory signature is
``def create_tool(config: TenantConfig, **deps) -> function_tool | None``.

By default no tenant tools are registered (the list below is empty) — add your
own tenant's tools here, or create a new module and wire it in
``ToolRegistry.register_tenant_tools``.
"""

from __future__ import annotations

import logging

from config.schema import TenantConfig  # noqa: F401  (imported for typing in real tools)

logger = logging.getLogger(__name__)


def register_example_tools(registry) -> None:
    """Register example tenant-specific tools in the ToolRegistry.

    No-op by default. Example of how you would register a tool:

        registry.register("my_tenant_tool", _my_tool_factory)
    """
    # No tenant tools registered by default. Add registry.register(...) calls
    # here for your tenant, following the _noop_tool_factory pattern below.
    logger.debug("No example tenant tools registered (template module)")


def _noop_tool_factory(config: "TenantConfig", **kwargs):
    """
    Placeholder factory for tools that are handled via sub-agent handoff.

    A tool name may exist in a YAML config's tools.tenant list while its actual
    behavior is implemented via sub_agents config + AgentFactory's
    _build_handoff_tools(). Returning None signals AgentFactory to skip building
    a standalone tool for this name (the handoff tool takes priority).
    """
    return None
