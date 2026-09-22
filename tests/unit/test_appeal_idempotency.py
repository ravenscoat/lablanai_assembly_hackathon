"""Idempotency guards for appeal submission.

Three layers protect against duplicate submissions in the same call:
  1. AppealAgent._submit_started — blocks concurrent submits while one is in
     flight (must be released even on exception).
  2. AppealAgent._submitted_once — blocks resubmits within this AppealAgent
     after a successful submit.
  3. parent.murojaat_id — blocks a fresh transfer_to_appeal handoff after
     the previous appeal succeeded (the new main agent inherits the id from
     the parent via BaseSubAgent._create_main_agent).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from api.models import MurojaatResult
from config.schema import FieldConfig, TenantConfig


def _config():
    return TenantConfig.model_validate(
        {
            "tenant": {"id": "t1", "slug": "yoshlar", "name": "Test"},
            "languages": {"default": "uz", "available": ["uz"]},
        }
    )


def _bare_appeal_agent(*, parent_murojaat_id: str | None = None):
    """Build an AppealAgent stripped to the state confirm_and_submit needs.
    Avoids LiveKit Agent.__init__ which requires a session/STT/TTS stack."""
    from agents.sub_agents.appeal import AppealAgent

    agent = AppealAgent.__new__(AppealAgent)
    agent._parent_config = _config()
    agent._fields = [FieldConfig(name="content", prompt="?")]
    agent._data = {"content": "shikoyat matni"}
    agent._goals = []
    agent._summary_instructions = {}
    agent._messages = {}
    agent._language = "uz"
    agent._parent_call_db_id = None
    agent._parent_caller_phone = None
    agent._parent_agent = SimpleNamespace(murojaat_id=parent_murojaat_id)
    agent._submit_started = False
    agent._submitted_once = False
    agent._platform_client = AsyncMock()
    agent._platform_client.submit_murojaat.return_value = MurojaatResult(
        success=True, murojaat_id="mur-1"
    )
    agent._create_main_agent = lambda: MagicMock()
    return agent


def _unwrap(tool):
    """LiveKit @function_tool wraps the coroutine; reach the inner fn."""
    return getattr(tool, "fnc", None) or getattr(tool, "_fnc", None) or tool


@pytest.mark.asyncio
async def test_rejects_when_already_submitted_once():
    from agents.sub_agents.appeal import AppealAgent

    agent = _bare_appeal_agent()
    agent._submitted_once = True

    fn = _unwrap(AppealAgent.confirm_and_submit)
    result = await fn(agent, context=None, summary="")

    assert result == AppealAgent._DUPLICATE_REJECT_MESSAGE
    agent._platform_client.submit_murojaat.assert_not_called()


@pytest.mark.asyncio
async def test_rejects_when_parent_has_murojaat_id():
    """A second transfer_to_appeal in the same call: the parent main agent
    inherited the previously-submitted murojaat_id, so even if the LLM tries
    to confirm_and_submit again, the guard fires."""
    from agents.sub_agents.appeal import AppealAgent

    agent = _bare_appeal_agent(parent_murojaat_id="mur-existing-1")

    fn = _unwrap(AppealAgent.confirm_and_submit)
    result = await fn(agent, context=None, summary="")

    assert result == AppealAgent._DUPLICATE_REJECT_MESSAGE
    agent._platform_client.submit_murojaat.assert_not_called()


@pytest.mark.asyncio
async def test_rejects_concurrent_submit_attempt():
    from agents.sub_agents.appeal import AppealAgent

    agent = _bare_appeal_agent()
    agent._submit_started = True

    fn = _unwrap(AppealAgent.confirm_and_submit)
    result = await fn(agent, context=None, summary="")

    assert result == AppealAgent._DUPLICATE_REJECT_MESSAGE
    agent._platform_client.submit_murojaat.assert_not_called()


@pytest.mark.asyncio
async def test_submit_started_resets_on_exception():
    """Without try/finally, a raising _submit() would leave _submit_started=True
    and every subsequent confirm_and_submit on this agent would falsely reject."""
    from agents.sub_agents.appeal import AppealAgent

    agent = _bare_appeal_agent()

    async def raising_submit():
        raise RuntimeError("network blew up")

    agent._submit = raising_submit

    fn = _unwrap(AppealAgent.confirm_and_submit)
    with pytest.raises(RuntimeError):
        await fn(agent, context=None, summary="")

    assert agent._submit_started is False


@pytest.mark.asyncio
async def test_submitted_once_set_only_after_success():
    """Failed submit must not flip _submitted_once — caller can retry."""
    from agents.sub_agents.appeal import AppealAgent

    agent = _bare_appeal_agent()

    async def failing_submit():
        return False

    agent._submit = failing_submit

    fn = _unwrap(AppealAgent.confirm_and_submit)
    result = await fn(agent, context=None, summary="")

    assert isinstance(result, str)
    assert agent._submitted_once is False
    assert agent._submit_started is False


@pytest.mark.asyncio
async def test_success_handoff_records_confirmation_in_main_history():
    from agents.sub_agents.appeal import AppealAgent

    agent = _bare_appeal_agent()
    main_agent = SimpleNamespace(conversation_history=[])
    agent._create_main_agent = lambda: main_agent

    fn = _unwrap(AppealAgent.confirm_and_submit)
    result = await fn(agent, context=None, summary="")

    assert result[0] is main_agent
    assert result[1] == AppealAgent._message_for(agent, "success")
    assert main_agent.conversation_history[-1]["role"] == "assistant"
    assert "muvaffaqiyatli qabul qilindi" in main_agent.conversation_history[-1]["content"]


def _stub_run_context(parent_agent):
    """RunContext.session.current_agent is what _do_handoff inspects."""
    session = MagicMock()
    session.current_agent = parent_agent
    return MagicMock(session=session)


@pytest.mark.asyncio
async def test_handoff_tool_rejects_when_parent_has_murojaat_id():
    """Even before the appeal sub-agent spins up, the handoff tool should
    refuse a second transfer when the previous appeal already succeeded."""
    from agents.factory import AgentFactory
    from agents.sub_agents.appeal import AppealAgent
    from config.schema import SubAgentConfig
    from tools.registry import ToolRegistry

    factory = AgentFactory(tool_registry=ToolRegistry())
    sub_config = SubAgentConfig.model_validate(
        {"type": "appeal", "fields": [{"name": "content", "prompt": "?"}]}
    )
    tool = factory._create_handoff_tool("appeal", sub_config, AppealAgent, _config())

    parent = SimpleNamespace(murojaat_id="mur-existing-1", caller_phone="+923901234567")
    fn = _unwrap(tool)
    result = await fn(_stub_run_context(parent))

    assert result == AppealAgent._DUPLICATE_REJECT_MESSAGE
