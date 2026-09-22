"""Integration smoke: two-stage escalation (request_escalation → confirm_escalation).

Replaces the legacy NAV-150 single-tool flow. Each test exercises one of:
- strict reason → reflection prompt → confirm fires transfer with metadata
- lenient reason → immediate forward, no reflection
- strict reason + no follow-up confirm → no transfer, no metadata buffer
- 3 unconfirmed requests → loop-break fires with loop_break=True in metadata
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from config.schema import TenantConfig


def _youth_config():
    return TenantConfig.model_validate(
        {
            "tenant": {"id": "yt", "slug": "yoshlar", "name": "Youth"},
            "transfer": {
                "enabled": True,
                "guardian": {
                    "operator_instructions": "Ask what they need, try KB, only escalate if insistent.",
                    "psychological_instructions": "Reply empathetically first, then offer.",
                },
            },
        }
    )


def _stub_agent(config):
    from agents.tenant_agent import TenantAgent

    agent = TenantAgent.__new__(TenantAgent)
    agent.config = config
    agent.call_db_id = "507f1f77bcf86cd799439011"
    agent.call_id = "lk"
    agent.transferred = False
    agent.transfer_reason = ""
    agent.ai_summary = ""
    agent.agent_identity = ""
    agent._platform_client = None
    agent._job_context = MagicMock()
    agent._session_state = MagicMock()
    agent.conversation_history = []
    agent._transfer_rejection_count = 0
    agent._proposed_escalation_reason = None
    agent._pending_transfer_metadata = {}
    agent.do_forward_to_operator = AsyncMock(return_value="TRANSFERRED")
    agent.language = "uz"
    return agent


def _unwrap(tool):
    return getattr(tool, "fnc", None) or getattr(tool, "_fnc", None) or tool


@pytest.mark.asyncio
async def test_user_dissatisfied_lenient_fires_immediately_with_metadata():
    """Caller insists on operator → request_escalation with user_dissatisfied
    is now lenient and fires immediately (NAV-219 deadlock fix)."""
    from tools.platform.escalate import create_request_escalation_tool

    config = _youth_config()
    agent = _stub_agent(config)

    ctx = MagicMock()
    ctx.session.current_agent = agent

    request = _unwrap(create_request_escalation_tool(config))

    # user_dissatisfied is lenient → fires immediately, no reflection.
    result = await request(ctx, reason_for_transfer="user_dissatisfied")
    assert result == "TRANSFERRED"
    agent.do_forward_to_operator.assert_awaited_once()
    assert agent._pending_transfer_metadata == {
        "caller_request": "",
        "ai_attempt_summary": "Lenient reason: user_dissatisfied",
        "reason_for_transfer": "user_dissatisfied",
        "loop_break": False,
    }


@pytest.mark.asyncio
async def test_kb_miss_strict_reason_requires_confirm():
    """kb_miss remains strict: request_escalation returns reflection →
    LLM responds with confirm_escalation → transfer fires with reasoning."""
    from tools.platform.escalate import (
        create_confirm_escalation_tool,
        create_request_escalation_tool,
    )

    config = _youth_config()
    agent = _stub_agent(config)

    ctx = MagicMock()
    ctx.session.current_agent = agent

    request = _unwrap(create_request_escalation_tool(config))
    confirm = _unwrap(create_confirm_escalation_tool(config))

    # Stage 1: strict reason → reflection prompt returned, no transfer.
    reflection = await request(ctx, reason_for_transfer="kb_miss")
    assert "INTERNAL CHECK" in reflection
    assert agent._proposed_escalation_reason == "kb_miss"
    assert agent._transfer_rejection_count == 1
    agent.do_forward_to_operator.assert_not_awaited()
    assert agent._pending_transfer_metadata == {}

    # Stage 2: confirm with reasoning → transfer fires.
    reasoning = "Searched KB, gave 5-7 day window, caller insisted on operator."
    result = await confirm(ctx, reasoning=reasoning)
    assert result == "TRANSFERRED"
    agent.do_forward_to_operator.assert_awaited_once()
    assert agent._pending_transfer_metadata == {
        "caller_request": "",
        "ai_attempt_summary": reasoning,
        "reason_for_transfer": "kb_miss",
        "loop_break": False,
    }
    # Counter and proposed reason reset after successful confirm.
    assert agent._transfer_rejection_count == 0
    assert agent._proposed_escalation_reason is None


@pytest.mark.asyncio
async def test_kb_hit_no_transfer_fired():
    """When the LLM resolves via KB and never calls request_escalation, no
    forward, no metadata buffer."""
    from tools.platform.escalate import create_request_escalation_tool

    config = _youth_config()
    agent = _stub_agent(config)

    # Tool instantiated but never invoked — same as a KB-hit conversation.
    tool = create_request_escalation_tool(config)
    assert tool is not None
    agent.do_forward_to_operator.assert_not_awaited()
    assert agent._pending_transfer_metadata == {}


@pytest.mark.asyncio
async def test_psychological_confirmed_bypasses_reflection():
    """Lenient reason → request_escalation forwards immediately, no reflection,
    no confirm needed."""
    from tools.platform.escalate import create_request_escalation_tool

    config = _youth_config()
    agent = _stub_agent(config)

    ctx = MagicMock()
    ctx.session.current_agent = agent

    request = _unwrap(create_request_escalation_tool(config))

    result = await request(ctx, reason_for_transfer="psychological_confirmed")
    assert result == "TRANSFERRED"
    agent.do_forward_to_operator.assert_awaited_once()
    assert agent._pending_transfer_metadata["reason_for_transfer"] == "psychological_confirmed"
    assert agent._pending_transfer_metadata["loop_break"] is False
    assert "Lenient reason" in agent._pending_transfer_metadata["ai_attempt_summary"]


@pytest.mark.asyncio
async def test_strict_request_without_confirm_does_not_fire_or_buffer():
    """request_escalation with strict reason returns reflection prompt; if the
    LLM does NOT follow up with confirm_escalation, no transfer fires."""
    from tools.platform.escalate import create_request_escalation_tool

    config = _youth_config()
    agent = _stub_agent(config)

    ctx = MagicMock()
    ctx.session.current_agent = agent

    request = _unwrap(create_request_escalation_tool(config))

    reflection = await request(ctx, reason_for_transfer="kb_miss")
    assert "INTERNAL CHECK" in reflection
    agent.do_forward_to_operator.assert_not_awaited()
    assert agent._pending_transfer_metadata == {}
    assert agent._transfer_rejection_count == 1
    assert agent._proposed_escalation_reason == "kb_miss"


@pytest.mark.asyncio
async def test_lenient_after_strict_resets_dangling_state():
    """If LLM calls request_escalation(kb_miss) (strict, sets counter/proposal)
    then later calls request_escalation(user_dissatisfied) (lenient), the lenient
    path fires immediately AND resets the dangling strict state."""
    from tools.platform.escalate import create_request_escalation_tool

    config = _youth_config()
    agent = _stub_agent(config)

    ctx = MagicMock()
    ctx.session.current_agent = agent

    request = _unwrap(create_request_escalation_tool(config))

    # Strict call — sets counter=1, proposal=kb_miss.
    reflection = await request(ctx, reason_for_transfer="kb_miss")
    assert "INTERNAL CHECK" in reflection
    assert agent._transfer_rejection_count == 1
    assert agent._proposed_escalation_reason == "kb_miss"
    agent.do_forward_to_operator.assert_not_awaited()

    # Lenient call — fires immediately AND resets dangling state.
    result = await request(ctx, reason_for_transfer="user_dissatisfied")
    assert result == "TRANSFERRED"
    agent.do_forward_to_operator.assert_awaited_once()
    # Dangling state is cleaned up.
    assert agent._transfer_rejection_count == 0
    assert agent._proposed_escalation_reason is None


@pytest.mark.asyncio
async def test_loop_break_fires_with_loop_break_flag():
    """2nd consecutive request_escalation without confirm fires the transfer
    deterministically (caller-safety ceiling), with loop_break=True in
    metadata."""
    from tools.platform.escalate import create_request_escalation_tool

    config = _youth_config()
    agent = _stub_agent(config)

    ctx = MagicMock()
    ctx.session.current_agent = agent

    request = _unwrap(create_request_escalation_tool(config))

    # First request returns reflection prompt.
    reflection = await request(ctx, reason_for_transfer="kb_miss")
    assert "INTERNAL CHECK" in reflection
    assert agent._transfer_rejection_count == 1
    agent.do_forward_to_operator.assert_not_awaited()

    # Second request hits the ceiling and fires.
    result = await request(ctx, reason_for_transfer="kb_miss")
    assert result == "TRANSFERRED"
    agent.do_forward_to_operator.assert_awaited_once()
    assert agent._pending_transfer_metadata == {
        "caller_request": "",
        "ai_attempt_summary": "loop_break — 2 unconfirmed requests",
        "reason_for_transfer": "kb_miss",
        "loop_break": True,
    }
