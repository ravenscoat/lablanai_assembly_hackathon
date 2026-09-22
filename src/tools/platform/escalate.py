"""
Two-stage escalation: request_escalation → reflection → confirm_escalation.

The LLM proposes a handoff via `request_escalation(reason_for_transfer)`. For
strict reasons (`kb_miss`, `user_dissatisfied`), the tool returns a reflection
prompt asking the LLM to enumerate what it tried and quote the caller's
dissatisfaction. The LLM then either calls `confirm_escalation(reasoning=...)`
to fire the transfer, or goes back to helping the caller.

Deterministic safety net: after `_LOOP_BREAK_THRESHOLD` request_escalation
calls without a successful confirm, the next request short-circuits and fires
the transfer with `loop_break=True`. Caller never gets trapped by a stuck LLM.

Lenient reasons (`out_of_scope`, `psychological_confirmed`) bypass the
reflection step — their reason IS the justification.

The deterministic OOH check is still applied first: outside office hours, all
escalations route to the appeal sub-agent regardless of reason.

The decision core (validation, counter, reflection prompt, metadata buffer)
lives in `transfer_guardian.py` so the equivalent tools on `BaseSubAgent` can
reuse the same logic (NAV-162). What stays here: the office-hours clock
check, the OOH redirect, the `_resolve_main_agent` walker, and the tool
factories themselves.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Literal
from zoneinfo import ZoneInfo

from livekit.agents import RunContext, function_tool

from config.schema import TenantConfig
from tools.platform.transfer_guardian import (
    _LOOP_BREAK_THRESHOLD,
    _STRICT_REASONS,
    REFLECTION_PROMPT_TEMPLATE,
    FireDecision,
    buffer_metadata,
    evaluate_confirm,
    evaluate_request,
)

logger = logging.getLogger(__name__)

# Re-exported so existing call sites (and tests) that import these from
# `tools.platform.escalate` keep working. The canonical home is
# `transfer_guardian`.
__all__ = [
    "TransferReason",
    "_LOOP_BREAK_THRESHOLD",
    "_STRICT_REASONS",
    "REFLECTION_PROMPT_TEMPLATE",
    "buffer_metadata",
    "create_confirm_escalation_tool",
    "create_request_escalation_tool",
    "try_ooh_redirect",
]

TransferReason = Literal[
    "kb_miss",
    "user_dissatisfied",
    "out_of_scope",
    "psychological_confirmed",
]

# Default office timezone when a tenant does not set transfer.timezone.
# Asia/Karachi for Pakistan (PK); override per-tenant in the YAML transfer block.
DEFAULT_TZ = "Asia/Karachi"

# Generic fallback used only when a tenant has not provided a localized
# out_of_hours_message for the active language.
_DEFAULT_OUT_OF_HOURS = (
    "Our operators are currently outside working hours. Let me take down your "
    "request — an operator will call you back during the next business day."
)


def _office_tz(transfer_cfg) -> ZoneInfo:
    """Resolve the office timezone from tenant config, defaulting to DEFAULT_TZ.

    Falls back to DEFAULT_TZ if the configured value is missing or invalid.
    """
    tz_name = (getattr(transfer_cfg, "timezone", None) or DEFAULT_TZ).strip() or DEFAULT_TZ
    try:
        return ZoneInfo(tz_name)
    except Exception:
        logger.warning("Invalid transfer.timezone %r; falling back to %s", tz_name, DEFAULT_TZ)
        return ZoneInfo(DEFAULT_TZ)


def _is_within_office_hours(transfer_cfg, now: datetime | None = None) -> bool:
    """Office-local clock check against the tenant's office_days +
    office_hours_start/end, using transfer.timezone (default Asia/Karachi).
    Hour is half-open: [start, end)."""
    dt = now or datetime.now(_office_tz(transfer_cfg))
    if dt.weekday() not in transfer_cfg.office_days:
        return False
    return transfer_cfg.office_hours_start <= dt.hour < transfer_cfg.office_hours_end


def _resolve_main_agent(context: RunContext) -> tuple[Any, Any]:
    """Return (current_agent, main_agent_or_None).

    Main agent owns the escalation state (counter, proposed reason, metadata
    buffer). Sub-agents delegate via `_parent_agent`."""
    agent = context.session.current_agent
    if hasattr(agent, "do_forward_to_operator"):
        return agent, agent
    parent = getattr(agent, "_parent_agent", None)
    if parent is not None and hasattr(parent, "do_forward_to_operator"):
        return agent, parent
    return agent, None


def try_ooh_redirect(context: RunContext, config: TenantConfig, *, log_extra: dict) -> Any | None:
    """If we are out of hours, hand off to AppealAgent. Returns the LiveKit
    handoff value when redirected, None otherwise."""
    if not config.transfer.enabled or _is_within_office_hours(config.transfer):
        return None
    from agents.factory import build_appeal_handoff_from_agent

    current = context.session.current_agent
    agent_lang = getattr(current, "language", None) or config.languages.default
    ooh_msg = config.transfer.out_of_hours_message.get(agent_lang) or _DEFAULT_OUT_OF_HOURS

    if getattr(current, "_parent_agent", None) is not None:
        logger.info(
            "Out-of-hours escalation requested from inside a sub-agent - returning message string"
        )
        return ooh_msg

    result = build_appeal_handoff_from_agent(current, config, override_message=ooh_msg)
    if result is not None:
        logger.info("Out-of-hours escalation routed to AppealAgent", extra=log_extra)
        return result
    logger.warning(
        "Out-of-hours escalate fired but tenant has no appeal sub-agent; "
        "falling through to operator transfer."
    )
    return None


def create_request_escalation_tool(config: TenantConfig, **kwargs):
    """Factory for the main agent's `request_escalation` tool.

    Sub-agents define their own version on `BaseSubAgent` because they must
    return a LiveKit handoff tuple on the fire path. Both delegate the
    decision to `transfer_guardian.evaluate_request`."""

    @function_tool(name="request_escalation")
    async def request_escalation(
        context: RunContext,
        reason_for_transfer: TransferReason,
    ) -> Any:
        """Propose a handoff to a human operator.

        For strict reasons (kb_miss, user_dissatisfied) the tool returns a
        reflection prompt. Read it, decide if your attempt was real, then
        either call confirm_escalation(reasoning=...) to fire the transfer
        or attempt the missing step first.

        For lenient reasons (out_of_scope, psychological_confirmed) the
        transfer fires immediately — these reasons are self-justifying.

        Args:
            reason_for_transfer:
              - kb_miss: you searched the KB but found no relevant answer.
              - user_dissatisfied: you answered but the caller insisted on a human.
              - out_of_scope: the request is categorically beyond the AI.
              - psychological_confirmed: caller confirmed they want a psychologist.
        """
        agent, main_agent = _resolve_main_agent(context)
        agent_lang = getattr(agent, "language", None) or config.languages.default
        log_extra = {"reason_for_transfer": reason_for_transfer, "language": agent_lang}

        decision = evaluate_request(
            main_agent,
            config,
            reason_for_transfer=reason_for_transfer,
            context=context,
            log_extra=log_extra,
        )
        if isinstance(decision, FireDecision):
            buffer_metadata(
                main_agent,
                reasoning=decision.reasoning,
                reason_for_transfer=decision.reason_for_transfer,
                loop_break=decision.loop_break,
            )
            return await main_agent.do_forward_to_operator()
        return decision

    return request_escalation


def create_confirm_escalation_tool(config: TenantConfig, **kwargs):
    """Factory for the main agent's `confirm_escalation` tool.

    Called after `request_escalation` returned a reflection prompt. The LLM
    supplies its own reasoning; the helper does the non-empty check and
    re-runs the OOH gate (which may have flipped during the LLM's thinking
    time near the boundary), then this wrapper fires the forward."""

    @function_tool(name="confirm_escalation")
    async def confirm_escalation(context: RunContext, reasoning: str) -> Any:
        """Confirm an escalation previously proposed via request_escalation.

        Args:
            reasoning: One sentence summarizing the concrete attempts you
                made (referencing real tool calls from your conversation
                history) and why the caller still needs a human.
                Example: "Searched KB for 'Japan grants', no match;
                caller insisted on operator."
        """
        agent, main_agent = _resolve_main_agent(context)
        agent_lang = getattr(agent, "language", None) or config.languages.default
        log_extra = {"language": agent_lang}

        decision = evaluate_confirm(
            main_agent,
            config,
            reasoning=reasoning,
            context=context,
            log_extra=log_extra,
        )
        if isinstance(decision, FireDecision):
            buffer_metadata(
                main_agent,
                reasoning=decision.reasoning,
                reason_for_transfer=decision.reason_for_transfer,
                loop_break=decision.loop_break,
            )
            return await main_agent.do_forward_to_operator()
        return decision

    return confirm_escalation
