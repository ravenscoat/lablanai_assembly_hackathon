"""Tests for operator handoff flow."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from api.models import OperatorStatus
from config.schema import TenantConfig


def _make_config(**overrides):
    # Tests in this file were written for the SIP-only handoff path
    # (request_operator/dashboard handoff is a separate web-mode flow added
    # in NAV-145/javlonbek's work). Default transfer_mode to "sip" here so
    # existing assertions about SIP fallback semantics keep holding.
    base = {"tenant": {"id": "test", "slug": "test", "name": "Test Agent"}}
    if "transfer" in overrides:
        overrides["transfer"].setdefault("transfer_mode", "sip")
    base.update(overrides)
    return TenantConfig.model_validate(base)


def _make_agent(config=None, call_db_id="call-123", platform_client=None):
    """Create a minimal TenantAgent for testing without LiveKit deps."""
    from agents.tenant_agent import TenantAgent

    agent = TenantAgent.__new__(TenantAgent)
    agent.config = config or _make_config(
        transfer={
            "enabled": True,
            "transfer_mode": "sip",
            "transfer_number": "+923712345678",
        }
    )
    agent.call_db_id = call_db_id
    agent.call_id = "lk-abc"
    agent.transferred = False
    agent._transfer_rejection_count = 0
    agent._pending_transfer_metadata = {}
    agent.transfer_reason = ""
    agent.ai_summary = ""
    agent.agent_identity = ""
    agent._platform_client = platform_client
    agent._job_context = MagicMock()
    agent._job_context.room.name = "room-abc"
    agent._session_state = MagicMock()
    agent._silence_monitor = None
    agent.conversation_history = [
        {"role": "user", "content": "operator bering"},
    ]
    # Mock background tasks started via asyncio.create_task inside
    # do_forward_to_operator. Without these, pytest-asyncio hangs on
    # teardown waiting for the never-completing SIP dispatch or status
    # poll. Tests that need the real _poll_operator_status (the two
    # polling tests below) override this on the agent instance.
    agent._sip_transfer = AsyncMock()
    agent._poll_operator_status = AsyncMock()
    return agent


def _mock_office_hours():
    """Return a context manager that mocks pytz to return Monday noon."""
    mock_now = MagicMock()
    mock_now.weekday.return_value = 0  # Monday
    mock_now.hour = 12  # Noon

    mock_tz = MagicMock()
    # When datetime.now(tz) is called, return our mock
    return patch("pytz.timezone", return_value=mock_tz), patch(
        "datetime.datetime", wraps=datetime, now=MagicMock(return_value=mock_now)
    )


class TestDashboardHandoff:
    @pytest.mark.asyncio
    async def test_requests_operator_via_api(self):
        mock_client = AsyncMock()
        mock_client.request_operator.return_value = True

        agent = _make_agent(platform_client=mock_client)

        # Mock time to be within office hours (Monday noon)
        mock_now = MagicMock()
        mock_now.weekday.return_value = 0
        mock_now.hour = 12
        mock_tz = MagicMock()

        with patch("pytz.timezone", return_value=mock_tz):
            # Patch datetime.now to return our mock when called with tz
            def fake_now(tz=None):
                return mock_now

            with patch("datetime.datetime") as mock_dt_cls:
                mock_dt_cls.now = fake_now
                result = await agent.do_forward_to_operator()

        assert result == "Sizni operatorga ulayman. Iltimos, kuting."
        mock_client.request_operator.assert_not_called()
        assert agent.transferred is True

    @pytest.mark.asyncio
    async def test_transfer_disabled_returns_message(self):
        config = _make_config(transfer={"enabled": False})
        agent = _make_agent(config=config)
        result = await agent.do_forward_to_operator()
        assert "mavjud emas" in result

    @pytest.mark.asyncio
    async def test_no_call_db_id_falls_to_sip(self):
        config = _make_config(transfer={"enabled": True, "transfer_number": "+923712345678"})
        agent = _make_agent(config=config, call_db_id=None, platform_client=None)

        mock_now = MagicMock()
        mock_now.weekday.return_value = 0
        mock_now.hour = 12
        mock_tz = MagicMock()

        with patch("pytz.timezone", return_value=mock_tz):

            def fake_now(tz=None):
                return mock_now

            with patch("datetime.datetime") as mock_dt_cls:
                mock_dt_cls.now = fake_now
                result = await agent.do_forward_to_operator()

        assert result == config.transfer.transfer_message
        assert agent.transferred is True

    @pytest.mark.asyncio
    async def test_build_conversation_summary(self):
        agent = _make_agent()
        agent.conversation_history = [
            {"role": "user", "content": "yoshlar daftari haqida"},
            {"role": "assistant", "content": "Albatta..."},
            {"role": "user", "content": "operator bering"},
        ]
        summary = agent._build_conversation_summary()
        assert "yoshlar daftari" in summary
        assert "operator bering" in summary

    @pytest.mark.asyncio
    async def test_build_conversation_summary_empty(self):
        agent = _make_agent()
        agent.conversation_history = []
        assert agent._build_conversation_summary() == ""

    @pytest.mark.asyncio
    async def test_poll_operator_transferred(self):
        """Test that polling removes AI when operator is transferred."""
        mock_client = AsyncMock()
        mock_client.get_operator_status.return_value = OperatorStatus(status="transferred")

        agent = _make_agent(platform_client=mock_client)
        agent._job_context.room.disconnect = AsyncMock()
        # Restore the real polling method — _make_agent mocks it by default
        # to prevent background-task hangs, but this test exercises it
        # directly. Also mock _wait_for_operator_in_room to return True
        # immediately instead of its real 15-second wall-clock wait.
        from agents.tenant_agent import TenantAgent

        agent._poll_operator_status = TenantAgent._poll_operator_status.__get__(agent)
        agent._wait_for_operator_in_room = AsyncMock(return_value=True)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await agent._poll_operator_status()

        mock_client.get_operator_status.assert_called_once()
        agent._job_context.room.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_poll_operator_failed_stops(self):
        """Test that polling stops on failed status."""
        from agents.tenant_agent import TenantAgent

        mock_client = AsyncMock()
        mock_client.get_operator_status.return_value = OperatorStatus(status="failed")

        agent = _make_agent(platform_client=mock_client)
        agent._job_context.room.disconnect = AsyncMock()
        agent._poll_operator_status = TenantAgent._poll_operator_status.__get__(agent)
        agent._wait_for_operator_in_room = AsyncMock(return_value=False)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            await agent._poll_operator_status()

        mock_client.get_operator_status.assert_called_once()
        # Should NOT disconnect on failure
        agent._job_context.room.disconnect.assert_not_called()

    @pytest.mark.asyncio
    async def test_escalate_from_subagent_delegates_to_parent(self, monkeypatch):
        """When request_escalation is called from a sub-agent's context, the
        counter and proposed-reason live on the PARENT main agent (so the
        request/confirm pair sees consistent state), and the eventual confirm
        path forwards via the parent's do_forward_to_operator.

        Regression: pre-fix, the sub-agent's escalate tool routed through
        the parent inconsistently and lost call_db_id / platform_client.
        """
        from tools.platform.escalate import (
            create_request_escalation_tool,
        )

        parent = _make_agent(platform_client=AsyncMock())
        parent.do_forward_to_operator = AsyncMock(return_value="OPERATOR_REQUESTED")
        parent._proposed_escalation_reason = None

        # Simulated sub-agent: no do_forward_to_operator, but has a
        # _parent_agent pointing to the real TenantAgent.
        class _Sub:
            pass

        sub = _Sub()
        sub._parent_agent = parent
        sub._platform_client = parent._platform_client

        ctx = MagicMock()
        ctx.session.current_agent = sub

        request = create_request_escalation_tool(parent.config)
        request_fn = getattr(request, "fnc", None) or getattr(request, "_fnc", None) or request

        # Ensure we are "in hours" to bypass the OOH guard
        from datetime import datetime
        from zoneinfo import ZoneInfo

        from tools.platform import escalate as mod

        monkeypatch.setattr(
            mod,
            "datetime",
            MagicMock(
                now=MagicMock(
                    return_value=datetime(2026, 4, 20, 10, 0, tzinfo=ZoneInfo("Asia/Tashkent"))
                )
            ),
        )

        # user_dissatisfied is now lenient — fires immediately, no reflection.
        result = await request_fn(ctx, reason_for_transfer="user_dissatisfied")
        assert result == "OPERATOR_REQUESTED"
        parent.do_forward_to_operator.assert_awaited_once()
        assert parent._proposed_escalation_reason is None
        assert parent._transfer_rejection_count == 0

    @pytest.mark.asyncio
    async def test_dashboard_failure_falls_back_to_sip(self):
        """When dashboard handoff fails, SIP fallback should be attempted."""
        mock_client = AsyncMock()
        mock_client.request_operator.return_value = False  # Dashboard fails

        config = _make_config(transfer={"enabled": True, "transfer_number": "+923712345678"})
        agent = _make_agent(config=config, platform_client=mock_client)

        mock_now = MagicMock()
        mock_now.weekday.return_value = 0
        mock_now.hour = 12
        mock_tz = MagicMock()

        with patch("pytz.timezone", return_value=mock_tz):

            def fake_now(tz=None):
                return mock_now

            with patch("datetime.datetime") as mock_dt_cls:
                mock_dt_cls.now = fake_now
                result = await agent.do_forward_to_operator()

        assert result == config.transfer.transfer_message
        assert agent.transferred is True

    @pytest.mark.asyncio
    async def test_out_of_hours_returns_next_shift_and_callback_offer(self):
        config = _make_config(
            tenant={"id": "test", "slug": "yoshlar", "name": "Yoshlar"},
            transfer={
                "enabled": True,
                "office_days": [0, 1, 2, 3, 4],  # Mon-Fri
                "office_hours_start": 9,
                "office_hours_end": 18,
            },
        )
        mock_client = AsyncMock()
        agent = _make_agent(config=config, platform_client=mock_client)
        agent._get_tashkent_now = MagicMock(return_value=datetime(2026, 4, 13, 20, 30))  # Monday

        result = await agent.do_forward_to_operator()

        assert "mavjud emas" in result
        assert "Navbatdagi operator smenasi: Seshanba soat 09:00" in result
        assert "callback" in result.lower()
        mock_client.request_operator.assert_not_called()
        assert agent.transferred is False

    @pytest.mark.asyncio
    async def test_closed_day_returns_next_shift_and_callback_offer(self):
        config = _make_config(
            tenant={"id": "test", "slug": "yoshlar", "name": "Yoshlar"},
            transfer={
                "enabled": True,
                "office_days": [0, 1, 2, 3, 4],  # Mon-Fri
                "office_hours_start": 9,
                "office_hours_end": 18,
            },
        )
        mock_client = AsyncMock()
        agent = _make_agent(config=config, platform_client=mock_client)
        agent._get_tashkent_now = MagicMock(return_value=datetime(2026, 4, 12, 10, 0))  # Sunday

        result = await agent.do_forward_to_operator()

        assert "mavjud emas" in result
        assert "Navbatdagi operator smenasi: Dushanba soat 09:00" in result
        assert "callback" in result.lower()
        mock_client.request_operator.assert_not_called()
        assert agent.transferred is False

    @pytest.mark.asyncio
    async def test_105_bypass_ignores_schedule(self):
        config = _make_config(
            tenant={"id": "test", "slug": "yoshlar", "name": "Yoshlar"},
            transfer={
                "enabled": True,
                "transfer_number": "sip:105@192.0.2.10",
                "office_days": [0, 1, 2, 3, 4],  # Mon-Fri
                "office_hours_start": 9,
                "office_hours_end": 18,
            },
        )
        agent = _make_agent(config=config, call_db_id=None, platform_client=None)
        agent._get_tashkent_now = MagicMock(return_value=datetime(2026, 4, 12, 23, 15))  # Sunday

        result = await agent.do_forward_to_operator()

        assert result == config.transfer.transfer_message
        assert agent.transferred is True

    @pytest.mark.asyncio
    async def test_out_of_hours_returns_schedule_aware_message(self):
        # The slug-specific out-of-hours fork was removed during genericization;
        # all tenants now get the schedule-aware message with a callback offer.
        config = _make_config(
            tenant={"id": "test", "slug": "example-tenant", "name": "Example"},
            transfer={"enabled": True, "office_days": [0, 1, 2, 3, 4]},
        )
        agent = _make_agent(config=config, platform_client=AsyncMock())
        agent._get_tashkent_now = MagicMock(return_value=datetime(2026, 4, 12, 10, 0))  # Sunday

        result = await agent.do_forward_to_operator()

        assert "operatorlar mavjud emas" in result.lower()
        assert "callback" in result.lower()

    # --- Two-stage escalation tool tests (request_escalation + confirm_escalation) ---
    #
    # These exercise the tool API directly, replacing the legacy
    # `create_escalate_tool` single-tool tests. State that used to live on
    # `_transfer_rejection_count` alone now spans that counter plus
    # `_proposed_escalation_reason`. Buffer assertions match the new
    # `_pending_transfer_metadata` shape (caller_request always empty,
    # ai_attempt_summary holds the LLM-supplied reasoning).

    @pytest.mark.asyncio
    async def test_request_lenient_reason_fires_transfer_immediately(self):
        """Lenient reason (out_of_scope) → request_escalation forwards
        immediately, no reflection step required."""
        from tools.platform.escalate import create_request_escalation_tool

        agent = _make_agent()
        agent._proposed_escalation_reason = None
        agent.do_forward_to_operator = AsyncMock(return_value="OPERATOR_REQUESTED")

        ctx = MagicMock()
        ctx.session.current_agent = agent

        tool = create_request_escalation_tool(agent.config)
        fn = getattr(tool, "fnc", None) or getattr(tool, "_fnc", None) or tool

        result = await fn(ctx, reason_for_transfer="out_of_scope")

        assert result == "OPERATOR_REQUESTED"
        agent.do_forward_to_operator.assert_awaited_once()
        assert agent._transfer_rejection_count == 0
        assert agent._proposed_escalation_reason is None

    @pytest.mark.asyncio
    async def test_request_strict_reason_returns_reflection_prompt(self):
        """Strict reason → reflection prompt returned, no transfer fired."""
        from tools.platform.escalate import create_request_escalation_tool

        agent = _make_agent()
        agent._proposed_escalation_reason = None
        agent.do_forward_to_operator = AsyncMock(return_value="OPERATOR_REQUESTED")

        ctx = MagicMock()
        ctx.session.current_agent = agent

        tool = create_request_escalation_tool(agent.config)
        fn = getattr(tool, "fnc", None) or getattr(tool, "_fnc", None) or tool

        result = await fn(ctx, reason_for_transfer="kb_miss")

        assert "INTERNAL CHECK" in result
        agent.do_forward_to_operator.assert_not_awaited()
        assert agent._transfer_rejection_count == 1
        assert agent._proposed_escalation_reason == "kb_miss"

    @pytest.mark.asyncio
    async def test_request_psychological_confirmed_fires_transfer(self):
        """psychological_confirmed is lenient → immediate forward."""
        from tools.platform.escalate import create_request_escalation_tool

        agent = _make_agent()
        agent._proposed_escalation_reason = None
        agent.do_forward_to_operator = AsyncMock(return_value="OPERATOR_REQUESTED")

        ctx = MagicMock()
        ctx.session.current_agent = agent

        tool = create_request_escalation_tool(agent.config)
        fn = getattr(tool, "fnc", None) or getattr(tool, "_fnc", None) or tool

        result = await fn(ctx, reason_for_transfer="psychological_confirmed")

        assert result == "OPERATOR_REQUESTED"
        agent.do_forward_to_operator.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_confirm_without_prior_request_returns_coaching(self):
        """confirm_escalation called out of order → coaching string, no
        transfer fired."""
        from tools.platform.escalate import create_confirm_escalation_tool

        agent = _make_agent()
        agent._proposed_escalation_reason = None
        agent.do_forward_to_operator = AsyncMock(return_value="OPERATOR_REQUESTED")

        ctx = MagicMock()
        ctx.session.current_agent = agent

        tool = create_confirm_escalation_tool(agent.config)
        fn = getattr(tool, "fnc", None) or getattr(tool, "_fnc", None) or tool

        result = await fn(ctx, reasoning="Tried KB.")

        assert "request_escalation" in result.lower()
        agent.do_forward_to_operator.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_confirm_with_empty_reasoning_returns_coaching(self):
        """confirm_escalation with whitespace-only reasoning → coaching
        string, no transfer fired (proposed_reason preserved for retry)."""
        from tools.platform.escalate import (
            create_confirm_escalation_tool,
            create_request_escalation_tool,
        )

        agent = _make_agent()
        agent._proposed_escalation_reason = None
        agent.do_forward_to_operator = AsyncMock(return_value="OPERATOR_REQUESTED")

        ctx = MagicMock()
        ctx.session.current_agent = agent

        request = create_request_escalation_tool(agent.config)
        confirm = create_confirm_escalation_tool(agent.config)
        request_fn = getattr(request, "fnc", None) or getattr(request, "_fnc", None) or request
        confirm_fn = getattr(confirm, "fnc", None) or getattr(confirm, "_fnc", None) or confirm

        await request_fn(ctx, reason_for_transfer="kb_miss")
        result = await confirm_fn(ctx, reasoning="   ")

        assert "reasoning" in result.lower()
        agent.do_forward_to_operator.assert_not_awaited()
        # Proposed reason kept so the LLM can retry confirm with real reasoning.
        assert agent._proposed_escalation_reason == "kb_miss"

    @pytest.mark.asyncio
    async def test_request_then_confirm_fires_transfer_and_resets_state(self):
        """Strict reason (kb_miss): request → reflection → confirm → forward.
        Counter and proposed_reason reset to 0/None after successful confirm."""
        from tools.platform.escalate import (
            create_confirm_escalation_tool,
            create_request_escalation_tool,
        )

        agent = _make_agent()
        agent._proposed_escalation_reason = None
        agent.do_forward_to_operator = AsyncMock(return_value="OPERATOR_REQUESTED")

        ctx = MagicMock()
        ctx.session.current_agent = agent

        request = create_request_escalation_tool(agent.config)
        confirm = create_confirm_escalation_tool(agent.config)
        request_fn = getattr(request, "fnc", None) or getattr(request, "_fnc", None) or request
        confirm_fn = getattr(confirm, "fnc", None) or getattr(confirm, "_fnc", None) or confirm

        await request_fn(ctx, reason_for_transfer="kb_miss")
        assert agent._transfer_rejection_count == 1
        assert agent._proposed_escalation_reason == "kb_miss"

        reasoning = "searched KB, gave 5-7 day window, caller insisted on operator"
        result = await confirm_fn(ctx, reasoning=reasoning)

        assert result == "OPERATOR_REQUESTED"
        agent.do_forward_to_operator.assert_awaited_once()
        assert agent._transfer_rejection_count == 0
        assert agent._proposed_escalation_reason is None
        assert agent._pending_transfer_metadata == {
            "caller_request": "",
            "ai_attempt_summary": reasoning,
            "reason_for_transfer": "kb_miss",
            "loop_break": False,
        }

    @pytest.mark.asyncio
    async def test_loop_break_after_two_unconfirmed_requests(self):
        """2nd request_escalation without an intervening confirm fires the
        transfer with loop_break=True (caller-safety ceiling)."""
        from tools.platform.escalate import create_request_escalation_tool

        agent = _make_agent()
        agent._proposed_escalation_reason = None
        agent.do_forward_to_operator = AsyncMock(return_value="OPERATOR_REQUESTED")

        ctx = MagicMock()
        ctx.session.current_agent = agent

        tool = create_request_escalation_tool(agent.config)
        fn = getattr(tool, "fnc", None) or getattr(tool, "_fnc", None) or tool

        # First request: reflection returned, no transfer.
        result = await fn(ctx, reason_for_transfer="kb_miss")
        assert "INTERNAL CHECK" in result
        assert agent._transfer_rejection_count == 1
        agent.do_forward_to_operator.assert_not_awaited()

        # Second request: hits threshold → loop-break fires transfer.
        result = await fn(ctx, reason_for_transfer="kb_miss")
        assert result == "OPERATOR_REQUESTED"
        agent.do_forward_to_operator.assert_awaited_once()
        assert agent._pending_transfer_metadata == {
            "caller_request": "",
            "ai_attempt_summary": "loop_break — 2 unconfirmed requests",
            "reason_for_transfer": "kb_miss",
            "loop_break": True,
        }

    @pytest.mark.asyncio
    async def test_lenient_request_resets_counter(self):
        """A successful lenient-reason request clears any built-up rejection
        streak from prior strict-reason requests."""
        from tools.platform.escalate import create_request_escalation_tool

        agent = _make_agent()
        agent._proposed_escalation_reason = None
        agent.do_forward_to_operator = AsyncMock(return_value="OPERATOR_REQUESTED")

        ctx = MagicMock()
        ctx.session.current_agent = agent

        tool = create_request_escalation_tool(agent.config)
        fn = getattr(tool, "fnc", None) or getattr(tool, "_fnc", None) or tool

        await fn(ctx, reason_for_transfer="kb_miss")
        assert agent._transfer_rejection_count == 1

        await fn(ctx, reason_for_transfer="out_of_scope")
        assert agent._transfer_rejection_count == 0
        assert agent._proposed_escalation_reason is None

    @pytest.mark.asyncio
    async def test_reflection_does_not_populate_metadata_buffer(self):
        """request_escalation returning a reflection prompt MUST NOT touch
        the metadata buffer — no transfer happened."""
        from tools.platform.escalate import create_request_escalation_tool

        agent = _make_agent()
        agent._proposed_escalation_reason = None
        agent.do_forward_to_operator = AsyncMock(return_value="OPERATOR_REQUESTED")

        ctx = MagicMock()
        ctx.session.current_agent = agent

        tool = create_request_escalation_tool(agent.config)
        fn = getattr(tool, "fnc", None) or getattr(tool, "_fnc", None) or tool

        await fn(ctx, reason_for_transfer="kb_miss")

        assert agent._pending_transfer_metadata == {}

    @pytest.mark.asyncio
    async def test_orphan_call_without_main_agent_returns_static_message(self):
        """No main_agent reachable (no do_forward_to_operator, no parent) →
        the tool returns the configured transfer_message and never touches a
        metadata buffer."""
        from tools.platform.escalate import create_request_escalation_tool

        class _Orphan:
            pass

        orphan = _Orphan()
        orphan._pending_transfer_metadata = {}

        config = _make_config(transfer={"enabled": True})

        ctx = MagicMock()
        ctx.session.current_agent = orphan

        tool = create_request_escalation_tool(config)
        fn = getattr(tool, "fnc", None) or getattr(tool, "_fnc", None) or tool

        result = await fn(ctx, reason_for_transfer="out_of_scope")

        assert result == config.transfer.transfer_message
        assert orphan._pending_transfer_metadata == {}
