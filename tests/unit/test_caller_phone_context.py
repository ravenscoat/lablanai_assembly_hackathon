"""Tests for caller phone context injection in youth tenant."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agents.tenant_agent import TenantAgent
from config.schema import TenantConfig


class _TurnCtx:
    def __init__(self):
        self.messages: list[dict] = []

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})


def _make_stub_agent(slug: str = "yoshlar", caller_phone: str = "+923990526804") -> TenantAgent:
    agent = TenantAgent.__new__(TenantAgent)
    agent.config = TenantConfig.model_validate(
        {
            "tenant": {"id": "t1", "slug": slug, "name": "Test"},
            "transfer": {"enabled": True},
        }
    )
    agent.caller_phone = caller_phone
    agent._context_window = MagicMock()
    agent._session_state = MagicMock()
    agent._session_state.get_state_summary.return_value = ""
    agent._session_state.should_suggest_wrap_up.return_value = False
    agent._session_state.record_turn.return_value = None
    agent.conversation_history = []
    agent._turns_since_last_sync = 0
    agent.call_db_id = None
    agent._error_handler = None
    agent._flow_engine = None
    agent._history_injected = True
    agent._caller_history = None
    return agent


@pytest.mark.asyncio
async def test_does_not_inject_phone_guidance_per_turn_for_youth():
    """Phone-number guidance lives in the static system prompt, not per-turn.
    Re-injecting it every turn just duplicates a static rule and burns tokens."""
    agent = _make_stub_agent(slug="yoshlar", caller_phone="+923990526804")
    turn_ctx = _TurnCtx()
    msg = SimpleNamespace(text_content="salom")

    await agent.on_user_turn_completed(turn_ctx, msg)

    combined = "\n".join(m["content"] for m in turn_ctx.messages)
    assert "+923990526804" not in combined
    assert "PHONE NUMBER POLICY" not in combined
    assert "MUHIM QOIDA - YOSHLAR CALL CONTEXT" not in combined


@pytest.mark.asyncio
async def test_does_not_inject_phone_guidance_per_turn_for_non_youth():
    agent = _make_stub_agent(slug="paynet", caller_phone="+923990526804")
    turn_ctx = _TurnCtx()
    msg = SimpleNamespace(text_content="raqamimni yozib oling")

    await agent.on_user_turn_completed(turn_ctx, msg)

    combined = "\n".join(m["content"] for m in turn_ctx.messages)
    assert "PHONE NUMBER POLICY" not in combined
    assert "HECH QACHON 'saqlay olmayman'" not in combined


@pytest.mark.asyncio
async def test_does_not_inject_psych_escalation_guidance_per_turn():
    """NAV-153 Transfer Guardian: the psych-escalation directive is handled
    once by the PSYCHOLOGICAL TRANSFER GUARDIAN block in the system prompt
    (rendered by factory.py), NOT re-injected per-turn."""
    agent = _make_stub_agent(slug="yoshlar", caller_phone="+923990526804")
    turn_ctx = _TurnCtx()
    msg = SimpleNamespace(text_content="menga psixologik yordam kerak")

    await agent.on_user_turn_completed(turn_ctx, msg)

    combined = "\n".join(m["content"] for m in turn_ctx.messages)
    assert "operators_available_now=" not in combined
    assert "psixologik yordam so'rasa" not in combined
    assert "darhol" not in combined
    assert "escalate_to_human toolini ishlating" not in combined
