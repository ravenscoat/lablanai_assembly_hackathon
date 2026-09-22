"""Tests for the shared guardian decision helpers (NAV-162).

Direct tests on `transfer_guardian.evaluate_request` / `evaluate_confirm`.
The main-agent and sub-agent call sites both delegate to these functions,
so coverage here is the most important regression guard.

The end-to-end sub-agent path (build new main agent + buffer metadata on it
+ return LiveKit handoff tuple) is exercised in `test_sub_agent_escalation.py`.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from config.schema import TenantConfig
from tools.platform.transfer_guardian import (
    _LOOP_BREAK_THRESHOLD,
    FireDecision,
    buffer_metadata,
    evaluate_confirm,
    evaluate_request,
)


def _make_config(transfer_enabled: bool = True):
    return TenantConfig.model_validate(
        {
            "tenant": {"id": "t1", "slug": "test", "name": "Test"},
            "transfer": {
                "enabled": transfer_enabled,
                # Force in-hours so OOH redirect never fires in unit tests
                # (the OOH path is covered separately).
                "office_hours_start": 0,
                "office_hours_end": 24,
                "office_days": [0, 1, 2, 3, 4, 5, 6],
            },
        }
    )


def _make_main_agent():
    """Minimal stand-in: a plain object so attribute writes work and
    `do_forward_to_operator` exists for `_resolve_main_agent` checks."""
    agent = MagicMock()
    agent._pending_transfer_metadata = {}
    agent._transfer_rejection_count = 0
    agent._proposed_escalation_reason = None
    return agent


def _make_context(current_agent=None):
    ctx = MagicMock()
    ctx.session.current_agent = current_agent or MagicMock()
    return ctx


class TestEvaluateRequest:
    def test_lenient_reason_returns_fire_decision_and_resets_counters(self):
        agent = _make_main_agent()
        agent._transfer_rejection_count = 2  # mid-loop
        agent._proposed_escalation_reason = "kb_miss"  # mid-loop

        decision = evaluate_request(
            agent,
            _make_config(),
            reason_for_transfer="out_of_scope",
            context=_make_context(),
            log_extra={},
        )

        assert isinstance(decision, FireDecision)
        assert decision.reason_for_transfer == "out_of_scope"
        assert decision.loop_break is False
        assert "Lenient" in decision.reasoning
        # Mid-loop state cleared.
        assert agent._transfer_rejection_count == 0
        assert agent._proposed_escalation_reason is None

    def test_psychological_confirmed_is_lenient(self):
        agent = _make_main_agent()
        decision = evaluate_request(
            agent,
            _make_config(),
            reason_for_transfer="psychological_confirmed",
            context=_make_context(),
            log_extra={},
        )
        assert isinstance(decision, FireDecision)

    def test_strict_reason_first_request_returns_reflection_prompt(self):
        agent = _make_main_agent()

        result = evaluate_request(
            agent,
            _make_config(),
            reason_for_transfer="kb_miss",
            context=_make_context(),
            log_extra={},
        )

        assert isinstance(result, str)
        assert "kb_miss" in result  # reflection prompt mentions the reason
        assert "confirm_escalation" in result
        # State recorded for the matching confirm_escalation call.
        assert agent._transfer_rejection_count == 1
        assert agent._proposed_escalation_reason == "kb_miss"

    def test_strict_reason_counter_increments_per_request(self):
        agent = _make_main_agent()

        # kb_miss is the only strict reason now (user_dissatisfied is lenient)
        result = evaluate_request(
            agent,
            _make_config(),
            reason_for_transfer="kb_miss",
            context=_make_context(),
            log_extra={},
        )
        assert isinstance(result, str)
        assert agent._transfer_rejection_count == 1

    def test_user_dissatisfied_is_lenient_fires_immediately(self):
        agent = _make_main_agent()

        decision = evaluate_request(
            agent,
            _make_config(),
            reason_for_transfer="user_dissatisfied",
            context=_make_context(),
            log_extra={},
        )
        assert isinstance(decision, FireDecision)
        assert decision.reason_for_transfer == "user_dissatisfied"
        assert decision.loop_break is False

    def test_loop_break_at_threshold_returns_fire_decision(self):
        agent = _make_main_agent()
        # Pre-load to one below threshold; next request hits threshold.
        agent._transfer_rejection_count = _LOOP_BREAK_THRESHOLD - 1

        decision = evaluate_request(
            agent,
            _make_config(),
            reason_for_transfer="kb_miss",
            context=_make_context(),
            log_extra={},
        )

        assert isinstance(decision, FireDecision)
        assert decision.loop_break is True
        assert "loop_break" in decision.reasoning
        assert decision.reason_for_transfer == "kb_miss"
        assert agent._proposed_escalation_reason is None

    def test_no_main_agent_returns_transfer_message(self):
        config = _make_config()

        result = evaluate_request(
            None,
            config,
            reason_for_transfer="kb_miss",
            context=_make_context(),
            log_extra={},
        )

        assert result == config.transfer.transfer_message

    def test_no_main_agent_with_transfer_disabled_returns_unavailable(self):
        result = evaluate_request(
            None,
            _make_config(transfer_enabled=False),
            reason_for_transfer="out_of_scope",
            context=_make_context(),
            log_extra={},
        )

        assert "mavjud emas" in result


class TestEvaluateConfirm:
    def test_confirm_after_request_returns_fire_decision(self):
        agent = _make_main_agent()
        # Simulate a previous request_escalation that set up the proposal.
        agent._proposed_escalation_reason = "user_dissatisfied"
        agent._transfer_rejection_count = 1

        decision = evaluate_confirm(
            agent,
            _make_config(),
            reasoning="Searched KB for grants, no match; caller insisted on operator.",
            context=_make_context(),
            log_extra={},
        )

        assert isinstance(decision, FireDecision)
        assert decision.reason_for_transfer == "user_dissatisfied"
        assert decision.loop_break is False
        assert decision.reasoning.startswith("Searched KB")
        # Both state vars cleared once the proposal is consumed.
        assert agent._proposed_escalation_reason is None
        assert agent._transfer_rejection_count == 0

    def test_confirm_without_prior_request_returns_coaching_string(self):
        agent = _make_main_agent()  # no _proposed_escalation_reason set

        result = evaluate_confirm(
            agent,
            _make_config(),
            reasoning="caller insisted",
            context=_make_context(),
            log_extra={},
        )

        assert isinstance(result, str)
        assert "request_escalation first" in result

    def test_confirm_with_empty_reasoning_returns_coaching_string(self):
        agent = _make_main_agent()
        agent._proposed_escalation_reason = "kb_miss"

        result = evaluate_confirm(
            agent,
            _make_config(),
            reasoning="   ",  # whitespace only
            context=_make_context(),
            log_extra={},
        )

        assert isinstance(result, str)
        assert "one-sentence reasoning" in result
        # State preserved so the LLM can retry the confirm.
        assert agent._proposed_escalation_reason == "kb_miss"

    def test_confirm_strips_whitespace_from_reasoning(self):
        agent = _make_main_agent()
        agent._proposed_escalation_reason = "kb_miss"

        decision = evaluate_confirm(
            agent,
            _make_config(),
            reasoning="  Caller frustrated.  ",
            context=_make_context(),
            log_extra={},
        )

        assert isinstance(decision, FireDecision)
        assert decision.reasoning == "Caller frustrated."

    def test_confirm_without_main_agent_returns_transfer_message(self):
        config = _make_config()

        result = evaluate_confirm(
            None,
            config,
            reasoning="anything",
            context=_make_context(),
            log_extra={},
        )

        assert result == config.transfer.transfer_message


class TestBufferMetadata:
    def test_buffer_writes_all_fields_on_agent(self):
        agent = _make_main_agent()

        buffer_metadata(
            agent,
            reasoning="Tried KB, caller insisted.",
            reason_for_transfer="user_dissatisfied",
            loop_break=False,
        )

        assert agent._pending_transfer_metadata == {
            "caller_request": "",
            "ai_attempt_summary": "Tried KB, caller insisted.",
            "reason_for_transfer": "user_dissatisfied",
            "loop_break": False,
        }

    def test_buffer_silently_skips_when_agent_is_none(self):
        # Defensive: helper must not raise on a missing main agent.
        buffer_metadata(
            None,
            reasoning="x",
            reason_for_transfer="kb_miss",
            loop_break=False,
        )

    def test_buffer_silently_skips_when_buffer_missing(self):
        agent = _make_main_agent()
        del agent._pending_transfer_metadata

        # No exception, no attribute created.
        buffer_metadata(
            agent,
            reasoning="x",
            reason_for_transfer="kb_miss",
            loop_break=False,
        )
        assert not hasattr(agent, "_pending_transfer_metadata")
