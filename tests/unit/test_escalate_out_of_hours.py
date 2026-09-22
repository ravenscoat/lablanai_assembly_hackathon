"""Out-of-hours routing in the two-stage escalation flow.

Both `request_escalation` and `confirm_escalation` re-check office hours via
`try_ooh_redirect` so a boundary crossing during the LLM's reflection turn
still routes the caller to AppealAgent instead of a no-op operator forward.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest

from config.schema import TenantConfig

TASHKENT = ZoneInfo("Asia/Tashkent")


def _config():
    return TenantConfig.model_validate(
        {
            "tenant": {"id": "ya-test", "slug": "yoshlar", "name": "YA"},
            "languages": {"default": "uz", "available": ["uz"]},
            "transfer": {
                "enabled": True,
                "transfer_mode": "sip",
                "office_hours_start": 9,
                "office_hours_end": 18,
                "office_days": [0, 1, 2, 3, 4, 5],
                "out_of_hours_message": {"uz": "TEST_OOH_UZ"},
            },
            "sub_agents": {
                "appeal": {
                    "type": "appeal",
                    "instructions": "Sen murojaat yig'uvchisan.",
                    "fields": [{"name": "full_name", "prompt": "Ism?"}],
                    "goals": [{"key": "problem_specifics", "description": {"uz": "x"}}],
                }
            },
        }
    )


def _fake_main_agent(config):
    agent = MagicMock()
    agent.do_forward_to_operator = AsyncMock(return_value="FORWARDED")
    agent._transfer_rejection_count = 0
    agent._proposed_escalation_reason = None
    agent._pending_transfer_metadata = {}
    agent.language = "uz"
    agent.caller_phone = "+923901234567"
    agent.chat_ctx = None
    agent._parent_agent = None
    return agent


def _unwrap(tool):
    return getattr(tool, "fnc", None) or getattr(tool, "_fnc", None) or tool


def _freeze_clock(monkeypatch, dt):
    """Pin tools.platform.escalate.datetime.now(tz) to return `dt`."""
    from tools.platform import escalate as mod

    monkeypatch.setattr(
        mod,
        "datetime",
        MagicMock(now=MagicMock(return_value=dt)),
    )


@pytest.mark.asyncio
async def test_in_hours_request_escalation_lenient_fires_forward(monkeypatch):
    """Lenient reason in-hours → request_escalation forwards immediately
    via do_forward_to_operator."""
    from tools.platform.escalate import create_request_escalation_tool

    cfg = _config()
    main_agent = _fake_main_agent(cfg)
    ctx = MagicMock()
    ctx.session.current_agent = main_agent

    # Monday 2026-04-20 10:00 Tashkent (within 9-18)
    _freeze_clock(monkeypatch, datetime(2026, 4, 20, 10, 0, tzinfo=TASHKENT))

    request = _unwrap(create_request_escalation_tool(cfg))
    result = await request(context=ctx, reason_for_transfer="out_of_scope")

    main_agent.do_forward_to_operator.assert_awaited_once()
    assert result == "FORWARDED"


@pytest.mark.asyncio
async def test_in_hours_strict_request_then_confirm_forwards(monkeypatch):
    """Strict reason in-hours: request returns reflection, confirm fires."""
    from tools.platform.escalate import (
        create_confirm_escalation_tool,
        create_request_escalation_tool,
    )

    cfg = _config()
    main_agent = _fake_main_agent(cfg)
    ctx = MagicMock()
    ctx.session.current_agent = main_agent

    _freeze_clock(monkeypatch, datetime(2026, 4, 20, 10, 0, tzinfo=TASHKENT))

    request = _unwrap(create_request_escalation_tool(cfg))
    confirm = _unwrap(create_confirm_escalation_tool(cfg))

    reflection = await request(context=ctx, reason_for_transfer="kb_miss")
    assert "INTERNAL CHECK" in reflection
    main_agent.do_forward_to_operator.assert_not_awaited()

    result = await confirm(
        context=ctx,
        reasoning="Searched KB; no match; caller insists on operator.",
    )
    assert result == "FORWARDED"
    main_agent.do_forward_to_operator.assert_awaited_once()


@pytest.mark.asyncio
async def test_out_of_hours_request_returns_appeal_handoff_with_ooh_msg(monkeypatch):
    """Out-of-hours request_escalation → AppealAgent handoff with the
    configured OOH message, regardless of strict/lenient reason."""
    from tools.platform.escalate import create_request_escalation_tool

    cfg = _config()
    main_agent = _fake_main_agent(cfg)
    ctx = MagicMock()
    ctx.session.current_agent = main_agent

    # Monday 22:00 Tashkent (after 18:00 close)
    _freeze_clock(monkeypatch, datetime(2026, 4, 20, 22, 0, tzinfo=TASHKENT))

    request = _unwrap(create_request_escalation_tool(cfg))
    result = await request(context=ctx, reason_for_transfer="kb_miss")

    main_agent.do_forward_to_operator.assert_not_called()
    assert isinstance(result, tuple)
    appeal, msg = result
    assert appeal.__class__.__name__ == "AppealAgent"
    assert msg == "TEST_OOH_UZ"


@pytest.mark.asyncio
async def test_out_of_hours_confirm_re_routes_to_appeal(monkeypatch):
    """If the LLM proposes in-hours but the OOH boundary crosses before
    confirm fires, the confirm path also re-checks and routes to AppealAgent."""
    from tools.platform.escalate import (
        create_confirm_escalation_tool,
        create_request_escalation_tool,
    )

    cfg = _config()
    main_agent = _fake_main_agent(cfg)
    ctx = MagicMock()
    ctx.session.current_agent = main_agent

    # 1) request_escalation while in-hours (kb_miss is strict → reflection).
    _freeze_clock(monkeypatch, datetime(2026, 4, 20, 17, 59, tzinfo=TASHKENT))
    request = _unwrap(create_request_escalation_tool(cfg))
    reflection = await request(context=ctx, reason_for_transfer="kb_miss")
    assert "INTERNAL CHECK" in reflection
    assert main_agent._proposed_escalation_reason == "kb_miss"

    # 2) confirm_escalation after OOH boundary crossed.
    _freeze_clock(monkeypatch, datetime(2026, 4, 20, 18, 1, tzinfo=TASHKENT))
    confirm = _unwrap(create_confirm_escalation_tool(cfg))
    result = await confirm(
        context=ctx,
        reasoning="Tried KB, gave answer, caller insisted on operator.",
    )

    main_agent.do_forward_to_operator.assert_not_called()
    assert isinstance(result, tuple)
    appeal, msg = result
    assert appeal.__class__.__name__ == "AppealAgent"
    assert msg == "TEST_OOH_UZ"
    # State cleared on the redirect.
    assert main_agent._proposed_escalation_reason is None
    assert main_agent._transfer_rejection_count == 0


@pytest.mark.asyncio
async def test_out_of_hours_falls_back_to_default_when_no_configured_message(monkeypatch):
    from tools.platform.escalate import create_request_escalation_tool

    cfg = _config()
    cfg.transfer.out_of_hours_message = {}  # clear the configured override
    main_agent = _fake_main_agent(cfg)
    ctx = MagicMock()
    ctx.session.current_agent = main_agent

    _freeze_clock(monkeypatch, datetime(2026, 4, 20, 22, 0, tzinfo=TASHKENT))

    request = _unwrap(create_request_escalation_tool(cfg))
    result = await request(context=ctx, reason_for_transfer="kb_miss")

    assert isinstance(result, tuple)
    _, msg = result
    assert "outside working hours" in msg.lower()


@pytest.mark.asyncio
async def test_sub_agent_ooh_returns_string_when_already_in_appeal(monkeypatch):
    """Caller mid-appeal who says 'operator!' at 22:00 must NOT get a real
    operator. Sub-agent's request_escalation re-runs the OOH gate."""
    from agents.factory import build_appeal_handoff_from_agent
    from agents.sub_agents.base_sub_agent import BaseSubAgent
    from tools.platform import escalate as mod  # noqa: F401

    cfg = _config()
    main_agent = _fake_main_agent(cfg)

    built = build_appeal_handoff_from_agent(main_agent, cfg)
    assert built is not None
    sub_agent, _ = built

    ctx = MagicMock()
    ctx.session.current_agent = sub_agent

    _freeze_clock(monkeypatch, datetime(2026, 4, 20, 22, 0, tzinfo=TASHKENT))

    tool = BaseSubAgent.request_escalation
    fn = _unwrap(tool)
    result = await fn(sub_agent, context=ctx, reason_for_transfer="user_dissatisfied")

    assert isinstance(result, str)
    assert result == "TEST_OOH_UZ"
    # The parent main agent's do_forward_to_operator must NOT have run.
    main_agent.do_forward_to_operator.assert_not_called()
