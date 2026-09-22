"""
end_call tool: ends the call gracefully.
"""

from __future__ import annotations

import logging

from livekit.agents import RunContext, function_tool

from config.schema import TenantConfig

logger = logging.getLogger(__name__)


def create_goodbye_tool(config: TenantConfig, **kwargs):
    """Factory: creates an end_call function_tool."""

    @function_tool(name="end_call")
    async def end_call(context: RunContext) -> str:
        """Qo'ng'iroqni tugatish. End the current phone call and say goodbye."""
        logger.info("Call ended via end_call tool")

        agent = context.session.current_agent
        if hasattr(agent, "do_end_call"):
            return await agent.do_end_call()

        return config.personality.goodbye

    return end_call
