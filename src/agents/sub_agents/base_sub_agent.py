"""
BaseSubAgent: common base for all sub-agents.
Provides cancel/return-to-main functionality.
"""

from __future__ import annotations

import logging
from typing import Any

from livekit.agents import Agent, RunContext, function_tool

from config.schema import TenantConfig
from tools.platform.escalate import TransferReason
from tools.platform.transfer_guardian import (
    FireDecision,
    buffer_metadata,
    evaluate_confirm,
    evaluate_request,
)

logger = logging.getLogger(__name__)


class BaseSubAgent(Agent):
    """
    Base class for all sub-agents (currently: AppealAgent).

    Provides the cancel / return-to-main pattern and the safety-net
    tools (escalate_to_human, end_call) that every sub-agent should
    expose so the user can always route out of a collection flow.
    """

    SUB_AGENT_NAME: str = "base"

    def __init__(
        self,
        *,
        instructions: str,
        parent_config: TenantConfig,
        chat_ctx: Any | None = None,
        **kwargs,
    ):
        super().__init__(instructions=instructions, chat_ctx=chat_ctx, **kwargs)
        self._parent_config = parent_config

    @function_tool()
    async def cancel_and_return(self, context: RunContext):
        """Joriy jarayonni bekor qilish va asosiy agentga qaytish. Cancel and return to main agent."""
        logger.info(f"Sub-agent {self.SUB_AGENT_NAME} cancelled, returning to main")
        main_agent = self._create_main_agent()
        return main_agent, "Yaxshi, bekor qildik. Boshqa savolingiz bormi?"

    # --- Safety-net tools available in every sub-agent ---
    #
    # Sub-agents must give the LLM a way out when the user changes
    # their mind mid-flow. Without these, an LLM inside AppealAgent
    # that hears "operatorga ulab ber" (connect to operator) or
    # "qo'ng'iroqni tugat" (hang up) has no tool to route the request
    # and will either ignore it or try to force-complete the form —
    # exactly the failure mode seen in the call log.
    #
    # Each tool hands back to a fresh main agent (is_post_handoff=True
    # so its on_enter doesn't trigger an LLM loop) and then delegates
    # to the main agent's `do_*` method, which has the call_db_id /
    # platform_client / job_context needed for the real backend work.

    @function_tool(name="request_escalation")
    async def request_escalation(
        self,
        context: RunContext,
        reason_for_transfer: TransferReason,
    ) -> Any:
        """Propose a handoff to a human operator from inside a sub-agent.

        Mirrors the main-agent tool: strict reasons return a reflection
        prompt; lenient reasons forward immediately; the 3-strike ceiling
        forces a forward. Counter and proposed-reason state are stored on
        the parent main agent so request/confirm pair across this sub-agent
        and its parent.

        Decision logic is delegated to `transfer_guardian.evaluate_request`
        (NAV-162). When the helper returns a `FireDecision`, this wrapper
        builds a fresh main agent, buffers the metadata on it (the new
        agent is what shutdown will read), runs the forward, and returns
        the LiveKit handoff tuple.
        """
        parent = getattr(self, "_parent_agent", None)
        cfg = self._parent_config
        agent_lang = getattr(self, "_language", None) or cfg.languages.default
        log_extra = {
            "sub_agent": self.SUB_AGENT_NAME,
            "reason_for_transfer": reason_for_transfer,
            "language": agent_lang,
        }

        decision = evaluate_request(
            parent,
            cfg,
            reason_for_transfer=reason_for_transfer,
            context=context,
            log_extra=log_extra,
        )
        if isinstance(decision, FireDecision):
            return await self._do_forward(
                reasoning=decision.reasoning,
                reason_for_transfer=decision.reason_for_transfer,
                loop_break=decision.loop_break,
            )
        return decision

    @function_tool(name="confirm_escalation")
    async def confirm_escalation(
        self,
        context: RunContext,
        reasoning: str,
    ) -> Any:
        """Confirm a previously-proposed escalation from inside a sub-agent.

        Args:
            reasoning: One-sentence summary of concrete attempts you made
                and why the caller still needs a human.
        """
        parent = getattr(self, "_parent_agent", None)
        cfg = self._parent_config
        agent_lang = getattr(self, "_language", None) or cfg.languages.default
        log_extra = {
            "sub_agent": self.SUB_AGENT_NAME,
            "language": agent_lang,
        }

        decision = evaluate_confirm(
            parent,
            cfg,
            reasoning=reasoning,
            context=context,
            log_extra=log_extra,
        )
        if isinstance(decision, FireDecision):
            return await self._do_forward(
                reasoning=decision.reasoning,
                reason_for_transfer=decision.reason_for_transfer,
                loop_break=decision.loop_break,
            )
        return decision

    async def _do_forward(
        self,
        *,
        reasoning: str,
        reason_for_transfer: str,
        loop_break: bool,
    ) -> Any:
        """Build a fresh main agent, run its `do_forward_to_operator`, and
        return the LiveKit handoff tuple so the session swaps to the main
        agent for post-transfer state. Mirrors the legacy sub-agent
        escalation behavior."""
        main_agent = self._create_main_agent()
        buffer_metadata(
            main_agent,
            reasoning=reasoning,
            reason_for_transfer=reason_for_transfer,
            loop_break=loop_break,
        )
        msg = self._parent_config.transfer.transfer_message
        if hasattr(main_agent, "do_forward_to_operator"):
            try:
                msg = await main_agent.do_forward_to_operator() or msg
            except Exception as e:
                logger.error(f"Operator handoff failed from sub-agent: {e}")
        return main_agent, msg

    @function_tool(name="end_call")
    async def end_call(self, context: RunContext) -> Any:
        """Qo'ng'iroqni tugatish. Call when the user has finished and
        wants to hang up (says xayr/rahmat/kerak emas and similar).
        """
        logger.info(f"Sub-agent {self.SUB_AGENT_NAME} ending call")
        main_agent = self._create_main_agent()
        msg = self._parent_config.personality.goodbye or "Xayr!"
        if hasattr(main_agent, "do_end_call"):
            try:
                result = await main_agent.do_end_call()
                if result:
                    msg = result
            except Exception as e:
                logger.error(f"end_call failed from sub-agent: {e}")
        return main_agent, msg

    def _create_main_agent(self) -> Agent:
        """Create a new main TenantAgent to return to.

        Forwards all call state (platform_client, job_context, call_db_id,
        language, caller history, conversation_history) from the parent
        so the reconstructed main agent can continue operator handoff,
        transcript sync, etc. The `is_post_handoff=True` flag tells the
        new agent's on_enter to wait for the user instead of triggering
        an LLM reply — without it, the LLM hallucinates another sub-agent
        handoff from chat_ctx and creates a duplicate-submission loop.
        """
        from agents.factory import get_factory

        parent = getattr(self, "_parent_agent", None)

        # Language: prefer the parent's runtime language (may differ from
        # tenant default after select_language).
        parent_lang = getattr(parent, "language", None) if parent else None

        factory = get_factory()
        agent = factory.create_agent(
            self._parent_config,
            chat_ctx=self.chat_ctx,
            platform_client=getattr(self, "_platform_client", None)
            or (getattr(parent, "_platform_client", None) if parent else None),
            job_context=getattr(self, "_parent_job_context", None)
            or (getattr(parent, "_job_context", None) if parent else None),
            current_language=parent_lang,
            is_post_handoff=True,
        )

        # Copy live call state from the parent so operator handoff,
        # transcript sync, and the final shutdown PATCH all have what
        # they need. Without this, escalate_to_human etc. silently
        # fall back to no-op because call_db_id is None.
        if parent is not None:
            for attr in (
                "call_db_id",
                "caller_phone",
                "agent_phone",
                "agent_identity",
                "_caller_history",
                "murojaat_id",
                "transferred",
            ):
                if hasattr(parent, attr):
                    try:
                        setattr(agent, attr, getattr(parent, attr))
                    except Exception as e:
                        logger.debug(f"_create_main_agent: could not forward {attr}: {e}")
            # conversation_history is a list — copy by extension to avoid
            # sharing the underlying list with the old (dead) agent.
            parent_history = getattr(parent, "conversation_history", None)
            if isinstance(parent_history, list) and parent_history:
                agent.conversation_history = list(parent_history)

        return agent
