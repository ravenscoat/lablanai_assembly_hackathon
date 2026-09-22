"""End-to-end-ish tests for BaseSubAgent escalation (NAV-162).

These cover the contract from the ticket's acceptance criteria:

1. Sub-agent escalation with valid args → metadata buffer populated on the
   freshly-built main agent (the one that will be alive at shutdown).
2. Sub-agent escalation with bad strict-reason summary → coaching string
   returned, no transfer attempted.
3. Lenient reasons forward immediately and still buffer metadata.

The sub-agent's fire path builds a NEW main agent (LiveKit handoff target)
and buffers metadata on THAT agent, not the existing parent. Shutdown then
reads the buffer off the new agent. Stubbing `_create_main_agent` lets us
verify the buffer ends up where shutdown expects to find it.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from config.schema import TenantConfig
from tools.platform.transfer_guardian import _LOOP_BREAK_THRESHOLD


def _make_config():
    return TenantConfig.model_validate(
        {
            "tenant": {"id": "t1", "slug": "test", "name": "Test"},
            "transfer": {
                "enabled": True,
                "office_hours_start": 0,
                "office_hours_end": 24,
                "office_days": [0, 1, 2, 3, 4, 5, 6],
                "transfer_message": "Operatorga ulamoqdaman.",
            },
        }
    )


def _make_parent():
    """A parent main agent that already exists from before the sub-agent
    handoff. The sub-agent reads/writes its counter + proposed-reason."""
    parent = MagicMock()
    parent._transfer_rejection_count = 0
    parent._proposed_escalation_reason = None
    # Parent's own buffer — not what we care about; the NEW main agent's
    # buffer is what shutdown reads.
    parent._pending_transfer_metadata = {}
    parent.language = "uz"
    parent.call_db_id = "db-abc"
    parent.caller_phone = "+923901234567"
    parent.agent_phone = "+923781225397"
    parent.agent_identity = "agent-id"
    parent.transferred = False
    parent.murojaat_id = None
    parent.conversation_history = []
    parent._platform_client = AsyncMock()
    parent._job_context = MagicMock()
    return parent


def _make_new_main_agent_factory():
    """Stub `_create_main_agent` to return a fresh agent each time and
    return the agent so the test can inspect its buffer."""
    new_agent = MagicMock()
    new_agent._pending_transfer_metadata = {}
    # Treat the forward call as successful with a known confirmation message
    # so the sub-agent's `_do_forward` returns the (agent, msg) tuple.
    new_agent.do_forward_to_operator = AsyncMock(return_value="Sizni ulayman, kuting.")
    return new_agent


def _make_sub_agent(cfg, parent, new_main_agent):
    """Bare BaseSubAgent-ish object wired up for the tests. We skip Agent.__init__
    (which needs LiveKit pipeline plumbing) by using `__new__` and patch
    `_create_main_agent` to return the agent we want to inspect."""
    from agents.sub_agents.appeal import AppealAgent

    sub = AppealAgent.__new__(AppealAgent)
    sub._parent_config = cfg
    sub._parent_agent = parent
    sub._platform_client = parent._platform_client
    sub._parent_job_context = parent._job_context
    sub._create_main_agent = MagicMock(return_value=new_main_agent)
    return sub


def _ctx(agent):
    ctx = MagicMock()
    ctx.session.current_agent = agent
    return ctx


def _unwrap(tool):
    """Pull the underlying coroutine function out of a `@function_tool` wrapper.

    LiveKit's `FunctionTool` exposes the original function as `_func` on
    current versions. Older versions used `fnc` / `_fnc`; the existing
    test suite falls through those as a portability shim.
    """
    return (
        getattr(tool, "_func", None)
        or getattr(tool, "fnc", None)
        or getattr(tool, "_fnc", None)
        or tool
    )


@pytest.mark.asyncio
async def test_lenient_reason_buffers_on_new_main_agent_and_returns_handoff_tuple():
    """Acceptance: sub-agent escalation with valid args → metadata.transfer
    populated on the agent shutdown will read (the freshly-built main)."""
    cfg = _make_config()
    parent = _make_parent()
    new_main = _make_new_main_agent_factory()
    sub = _make_sub_agent(cfg, parent, new_main)

    # Bypass Agent's descriptor by using __get__ to extract bound method.
    result = await sub.request_escalation._func(sub, _ctx(sub), reason_for_transfer="out_of_scope")

    assert isinstance(result, tuple) and len(result) == 2
    returned_agent, msg = result
    assert returned_agent is new_main
    assert msg == "Sizni ulayman, kuting."
    # Buffer written on the NEW main, NOT the parent — shutdown reads the
    # new agent post-handoff.
    assert new_main._pending_transfer_metadata == {
        "caller_request": "",
        "ai_attempt_summary": "Lenient reason: out_of_scope",
        "reason_for_transfer": "out_of_scope",
        "loop_break": False,
    }
    # Parent buffer untouched (the parent is being replaced).
    assert parent._pending_transfer_metadata == {}
    new_main.do_forward_to_operator.assert_awaited_once()


@pytest.mark.asyncio
async def test_strict_reason_first_request_returns_reflection_no_handoff():
    """Acceptance: sub-agent escalation with strict reason on first call
    must return the reflection prompt (string), not fire a transfer."""
    cfg = _make_config()
    parent = _make_parent()
    new_main = _make_new_main_agent_factory()
    sub = _make_sub_agent(cfg, parent, new_main)

    result = await sub.request_escalation._func(sub, _ctx(sub), reason_for_transfer="kb_miss")

    assert isinstance(result, str)
    assert "kb_miss" in result
    assert "confirm_escalation" in result
    # No new agent created, no forward fired.
    sub._create_main_agent.assert_not_called()
    new_main.do_forward_to_operator.assert_not_called()
    # State recorded on the PARENT (the active main agent at this point).
    assert parent._transfer_rejection_count == 1
    assert parent._proposed_escalation_reason == "kb_miss"


@pytest.mark.asyncio
async def test_strict_reason_loop_break_threshold_fires_transfer():
    """Loop-break must trigger a forward from sub-agent context too."""
    cfg = _make_config()
    parent = _make_parent()
    parent._transfer_rejection_count = _LOOP_BREAK_THRESHOLD - 1
    new_main = _make_new_main_agent_factory()
    sub = _make_sub_agent(cfg, parent, new_main)

    result = await sub.request_escalation._func(sub, _ctx(sub), reason_for_transfer="kb_miss")

    assert isinstance(result, tuple) and len(result) == 2
    returned_agent, _msg = result
    assert returned_agent is new_main
    # Loop-break recorded with loop_break=True so the dashboard can flag it.
    assert new_main._pending_transfer_metadata["loop_break"] is True
    assert new_main._pending_transfer_metadata["reason_for_transfer"] == "kb_miss"
    new_main.do_forward_to_operator.assert_awaited_once()


@pytest.mark.asyncio
async def test_confirm_with_valid_reasoning_buffers_and_fires():
    """After a strict request that returned reflection, a valid confirm
    must populate `metadata.transfer.ai_attempt_summary` with the LLM's
    reasoning string and fire the forward."""
    cfg = _make_config()
    parent = _make_parent()
    # Simulate having already gone through `request_escalation` once.
    parent._proposed_escalation_reason = "kb_miss"
    parent._transfer_rejection_count = 1
    new_main = _make_new_main_agent_factory()
    sub = _make_sub_agent(cfg, parent, new_main)

    result = await sub.confirm_escalation._func(
        sub,
        _ctx(sub),
        reasoning="Searched KB for grants, no match; caller insisted on operator.",
    )

    assert isinstance(result, tuple)
    returned_agent, _msg = result
    assert returned_agent is new_main
    assert new_main._pending_transfer_metadata == {
        "caller_request": "",
        "ai_attempt_summary": "Searched KB for grants, no match; caller insisted on operator.",
        "reason_for_transfer": "kb_miss",
        "loop_break": False,
    }
    # Proposal consumed, counter reset on the parent.
    assert parent._proposed_escalation_reason is None
    assert parent._transfer_rejection_count == 0


@pytest.mark.asyncio
async def test_confirm_with_empty_reasoning_returns_coaching_string_no_handoff():
    """Acceptance: sub-agent confirm with bad reasoning → coaching string,
    no transfer."""
    cfg = _make_config()
    parent = _make_parent()
    parent._proposed_escalation_reason = "kb_miss"
    new_main = _make_new_main_agent_factory()
    sub = _make_sub_agent(cfg, parent, new_main)

    result = await sub.confirm_escalation._func(sub, _ctx(sub), reasoning="   ")

    assert isinstance(result, str)
    assert "one-sentence reasoning" in result
    sub._create_main_agent.assert_not_called()
    new_main.do_forward_to_operator.assert_not_called()
    # Proposal preserved so the LLM gets another chance to confirm properly.
    assert parent._proposed_escalation_reason == "kb_miss"


@pytest.mark.asyncio
async def test_confirm_without_prior_request_returns_coaching_string():
    cfg = _make_config()
    parent = _make_parent()  # no _proposed_escalation_reason
    new_main = _make_new_main_agent_factory()
    sub = _make_sub_agent(cfg, parent, new_main)

    result = await sub.confirm_escalation._func(sub, _ctx(sub), reasoning="anything goes here")

    assert isinstance(result, str)
    assert "request_escalation first" in result
    sub._create_main_agent.assert_not_called()
