"""
start_flow tool: LLM calls this to activate a guided conversation flow.
Returns text that LLM MUST speak verbatim (no rephrasing).
"""

from __future__ import annotations

import logging

from livekit.agents import RunContext, function_tool

from config.schema import TenantConfig

logger = logging.getLogger(__name__)


def create_start_flow_tool(config: TenantConfig, **kwargs):
    """Factory: creates a start_flow function_tool."""

    @function_tool(name="start_flow")
    async def start_flow(context: RunContext, flow_id: str) -> str:
        """Start a guided flow. Speak the returned text to the caller exactly as-is.

        Args:
            flow_id: The flow to start. Valid ids come from the tenant's
                .flow.md files under knowledge/<kb-dir>/flows/.
        """
        # TODO(urdu): the user-facing strings returned by this tool below are in
        # Uzbek — add localized ('ur'/'en') variants for an Urdu tenant.
        agent = context.session.current_agent

        if not hasattr(agent, "_flow_engine") or not agent._flow_engine:
            return "Jarayonlar tizimi mavjud emas."

        engine = agent._flow_engine

        # If already in a flow, deactivate it and switch to new one
        if engine.is_active():
            engine.deactivate()

        flow = engine.get_flow(flow_id)
        if not flow:
            available = list(engine._flows.keys())
            return f"'{flow_id}' topilmadi. Mavjud: {', '.join(available)}"

        result = engine.activate(flow)

        logger.info(f"Flow started: {flow_id}")
        if result.step_context:
            return result.step_context

        engine.deactivate()
        return "Jarayonni boshlashda xatolik."

    return start_flow
