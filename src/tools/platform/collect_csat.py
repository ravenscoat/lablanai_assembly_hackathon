"""collect_csat platform tool (NAV-156).

Captures a 1-5 satisfaction rating from the caller before the call ends.
The rating is stashed on the agent as `_pending_csat` and flushed into
`metadata.csat` by `lifecycle/shutdown.py` on the call-record PATCH.

Out-of-range values, refusals, and non-numeric answers are handled by the
system-prompt rule, not the tool: if the LLM never calls `collect_csat`,
the agent simply has no `_pending_csat` attribute and the call record
omits `csat` — graceful null on the dashboard side.
"""

from __future__ import annotations

import logging

from livekit.agents import RunContext, function_tool

from config.schema import TenantConfig

logger = logging.getLogger(__name__)


_THANKS: dict[str, str] = {
    "uz": "Bahoyingiz uchun rahmat!",
    "ru": "Спасибо за вашу оценку!",
}

_OUT_OF_RANGE: dict[str, str] = {
    "uz": "Iltimos, 1 dan 5 gacha bo'lgan raqam ayting.",
    "ru": "Пожалуйста, назовите число от 1 до 5.",
}


def create_collect_csat_tool(config: TenantConfig, **kwargs):
    """Factory: creates a collect_csat function_tool."""

    @function_tool(name="collect_csat")
    async def collect_csat(context: RunContext, rating: int) -> str:
        """Foydalanuvchi suhbatni 1 dan 5 gacha baholaganida chaqirish.

        Call ONLY when the caller has explicitly given a 1-5 rating for the
        conversation. Do NOT call when the caller refuses to rate, hangs up,
        or answers with a non-numeric word — in those cases proceed directly
        to end_call without invoking this tool.

        Args:
            rating: integer score, must be in [1, 5].
        """
        agent = context.session.current_agent
        lang = getattr(agent, "language", None) or "uz"

        if not isinstance(rating, int) or not (1 <= rating <= 5):
            logger.info("collect_csat rejected: rating=%r", rating)
            return _OUT_OF_RANGE.get(lang, _OUT_OF_RANGE["uz"])

        agent._pending_csat = rating
        logger.info("collect_csat: rating=%d stashed on agent", rating)
        return _THANKS.get(lang, _THANKS["uz"])

    return collect_csat
