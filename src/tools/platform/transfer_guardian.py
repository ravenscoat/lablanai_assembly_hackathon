"""
Transfer-guardian decision core (NAV-162).

Both the main-agent escalation tools (`src/tools/platform/escalate.py`) and the
sub-agent ones (`BaseSubAgent`) need the *same* logic — strict-vs-lenient
reason validation, the loop-break counter, OOH redirect, the reflection prompt
on first request. The only difference between the two call sites is the *fire*
step: the main agent forwards in place; a sub-agent has to build a fresh main
agent and hand the LiveKit session back to it.

This module owns the decision logic plus the shared constants. Each call site
asks for a decision via `evaluate_request` / `evaluate_confirm`, then executes
the fire in its own way when the helper returns a `FireDecision`.

The helpers MUTATE the passed-in agent (`_transfer_rejection_count`,
`_proposed_escalation_reason`). That mirrors the pre-refactor code and matches
where the state has always lived. `try_ooh_redirect` is imported lazily from
`escalate.py` to avoid a top-level circular import — that module still owns
the office-hours clock check because tests patch its `datetime` attribute.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from livekit.agents import RunContext

from config.schema import TenantConfig

logger = logging.getLogger(__name__)


# Source of truth for these — escalate.py and base_sub_agent.py import from
# here to avoid drift.
#
# NOTE: "user_dissatisfied" was moved to lenient (NAV-219). When the caller
# explicitly asks for an operator the guardian instructions already gate the
# request (agent must ask "why" first). The two-step reflection caused a
# deadlock: the LLM spoke "kuting" but never called confirm_escalation,
# leaving the caller in silence until hangup.
_STRICT_REASONS: set[str] = {"kb_miss"}
_LOOP_BREAK_THRESHOLD: int = 2

REFLECTION_PROMPT_TEMPLATE: str = (
    "[INTERNAL CHECK — do NOT speak this to the caller.]\n\n"
    "You proposed escalation (reason={reason}). Before it fires, silently "
    "verify:\n"
    "- Did you actually call a tool to help (e.g. search_knowledge_base)?\n"
    "- Did the caller insist on a human AFTER you attempted to help?\n\n"
    "Respond with EXACTLY ONE of the following, and nothing else:\n"
    'A) If both are true → call confirm_escalation(reasoning="<one sentence '
    'summarizing your real attempt>"). Do NOT speak before, after, or instead '
    "of this tool call.\n"
    "B) If something is missing → take that action now (search KB, ask a "
    "clarifying question, offer an alternative). Speak naturally to the caller "
    "about what you are doing, but do NOT narrate this check or mention the "
    "tools by name.\n\n"
    'Forbidden: do NOT say things like "men bilimlar bazasidan qidirdim", '
    '"men urinib ko\'rdim", "keling tekshirib ko\'ray". The caller must not '
    "hear this reflection at all.\n\n"
    "Unconfirmed requests: {count}/{threshold}. After {threshold}, escalation "
    "fires regardless."
)


@dataclass
class FireDecision:
    """Caller should buffer metadata and fire the transfer with these args.

    `reasoning` lands in `metadata.transfer.ai_attempt_summary` on the call
    record (legacy schema; renaming requires a dashboard migration).
    """

    reasoning: str
    reason_for_transfer: str
    loop_break: bool


def evaluate_request(
    main_agent: Any,
    config: TenantConfig,
    *,
    reason_for_transfer: str,
    context: RunContext,
    log_extra: dict[str, Any],
) -> Any:
    """Decide what to do with a `request_escalation` tool invocation.

    Returns either:
    - `FireDecision` — caller fires the transfer (after buffering metadata).
    - LiveKit handoff tuple — OOH redirect to AppealAgent. Caller returns
      verbatim.
    - `str` — reflection prompt or stop message. Caller returns verbatim.

    Mutates `main_agent`: increments `_transfer_rejection_count`, sets
    `_proposed_escalation_reason` for the matching `confirm_escalation` call.

    `main_agent` may be `None` if a sub-agent has no `_parent_agent`
    (defensive — should not happen in production).
    """
    ooh = _try_ooh_redirect(context, config, log_extra=log_extra)
    if ooh is not None:
        return ooh

    logger.info("Escalation requested", extra=log_extra)

    if main_agent is None:
        return _no_main_agent_message(config)

    if reason_for_transfer not in _STRICT_REASONS:
        # Lenient reasons (out_of_scope, psychological_confirmed) self-justify.
        # Reset counter + proposal in case the LLM was mid-loop.
        main_agent._transfer_rejection_count = 0
        main_agent._proposed_escalation_reason = None
        return FireDecision(
            reasoning=f"Lenient reason: {reason_for_transfer}",
            reason_for_transfer=reason_for_transfer,
            loop_break=False,
        )

    count = getattr(main_agent, "_transfer_rejection_count", 0) + 1
    main_agent._transfer_rejection_count = count

    if count >= _LOOP_BREAK_THRESHOLD:
        logger.warning(
            "Escalation loop-break triggered (%d unconfirmed requests)",
            count,
            extra=log_extra,
        )
        main_agent._proposed_escalation_reason = None
        return FireDecision(
            reasoning=f"loop_break — {count} unconfirmed requests",
            reason_for_transfer=reason_for_transfer,
            loop_break=True,
        )

    # Strict reason, below threshold → record the proposal so the matching
    # confirm_escalation can validate it, and return the reflection prompt.
    main_agent._proposed_escalation_reason = reason_for_transfer
    logger.info("Reflection prompt returned (count=%d)", count, extra=log_extra)
    return REFLECTION_PROMPT_TEMPLATE.format(
        reason=reason_for_transfer,
        count=count,
        threshold=_LOOP_BREAK_THRESHOLD,
    )


def evaluate_confirm(
    main_agent: Any,
    config: TenantConfig,
    *,
    reasoning: str,
    context: RunContext,
    log_extra: dict[str, Any],
) -> Any:
    """Decide what to do with a `confirm_escalation` tool invocation.

    Returns either:
    - `FireDecision` — caller fires the transfer with the user-supplied
      reasoning.
    - LiveKit handoff tuple — OOH redirect (the hour can flip between
      request and confirm). Caller returns verbatim.
    - `str` — coaching string (no prior request, or empty reasoning).
      Caller returns verbatim.

    Mutates `main_agent`: clears `_proposed_escalation_reason` and
    `_transfer_rejection_count` on every non-error path.
    """
    reasoning_len = len(reasoning.strip()) if reasoning else 0
    log_extra = {**log_extra, "reasoning_length": reasoning_len}

    if main_agent is None:
        return _no_main_agent_message(config)

    proposed = getattr(main_agent, "_proposed_escalation_reason", None)
    if not proposed:
        logger.warning(
            "confirm_escalation called without prior request_escalation",
            extra=log_extra,
        )
        return (
            "Call request_escalation first with a reason_for_transfer. "
            "confirm_escalation can only be used after a request that was "
            "returned for reflection."
        )

    if reasoning_len == 0:
        logger.info("confirm_escalation rejected: empty reasoning", extra=log_extra)
        return (
            "Provide a one-sentence reasoning describing what you tried "
            "and why the caller still needs a human."
        )

    ooh = _try_ooh_redirect(
        context,
        config,
        log_extra={**log_extra, "reason_for_transfer": proposed},
    )
    if ooh is not None:
        main_agent._proposed_escalation_reason = None
        main_agent._transfer_rejection_count = 0
        return ooh

    main_agent._proposed_escalation_reason = None
    main_agent._transfer_rejection_count = 0
    logger.info(
        "Escalation confirmed",
        extra={**log_extra, "reason_for_transfer": proposed},
    )
    return FireDecision(
        reasoning=reasoning.strip(),
        reason_for_transfer=proposed,
        loop_break=False,
    )


def buffer_metadata(
    target_agent: Any,
    *,
    reasoning: str,
    reason_for_transfer: str,
    loop_break: bool,
) -> None:
    """Write transfer metadata onto `target_agent` for the shutdown PATCH.

    `target_agent` is whichever agent will be alive when shutdown fires:
    - Main-agent path: the current main agent.
    - Sub-agent path: the freshly-built post-handoff main agent.

    Field names match the legacy dashboard schema (`ai_attempt_summary`
    rather than `reasoning`). `caller_request` is left empty — its data is
    captured in the conversation-derived `ai_summary` written by
    `do_forward_to_operator`.
    """
    if target_agent is None:
        return
    buf = getattr(target_agent, "_pending_transfer_metadata", None)
    if buf is None:
        return
    buf["caller_request"] = ""
    buf["ai_attempt_summary"] = reasoning
    buf["reason_for_transfer"] = reason_for_transfer
    buf["loop_break"] = loop_break


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _no_main_agent_message(config: TenantConfig) -> str:
    """Fallback when the helper is asked to evaluate without a main agent.
    Defensive — should not happen in production, but the legacy code had a
    graceful fallback and the existing tests cover it."""
    if not config.transfer.enabled:
        return "Hozirda operator xizmati mavjud emas. Boshqa savol berishingiz mumkin."
    return config.transfer.transfer_message


def _try_ooh_redirect(
    context: RunContext,
    config: TenantConfig,
    *,
    log_extra: dict[str, Any],
) -> Any | None:
    """Lazy import: the OOH clock check still lives in `escalate.py` because
    the OOH test suite patches its `datetime` attribute (`tools.platform.
    escalate.datetime.now`). Keeping the helper there preserves that patch
    path without a test rewrite."""
    from tools.platform.escalate import try_ooh_redirect

    return try_ooh_redirect(context, config, log_extra=log_extra)
