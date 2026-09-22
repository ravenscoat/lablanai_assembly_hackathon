"""Tests for sub-agent → main agent handoff state preservation.

These guard against the regression where the main agent created on
sub-agent return:
  (a) lost its call state (call_db_id, platform_client, job_context,
      language, caller history), breaking subsequent operator handoff,
  (b) triggered an LLM reply in on_enter, causing Gemini to hallucinate
      another transfer_to_<name> call and submit duplicate appeals.

See the call log analysis: after AppealAgent.confirm_and_submit the main
agent kept calling transfer_to_appeal even when the user asked for an
operator. Fix: BaseSubAgent._create_main_agent now forwards all state
and marks the new agent as is_post_handoff=True so on_enter waits for
the user.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from config.schema import TenantConfig


def _make_config():
    return TenantConfig.model_validate(
        {
            "tenant": {"id": "t1", "slug": "test", "name": "Test"},
            "transfer": {"enabled": True},
        }
    )


def _make_parent_agent(config):
    """Minimal TenantAgent stand-in with all the state a sub-agent needs."""
    from agents.tenant_agent import TenantAgent

    parent = TenantAgent.__new__(TenantAgent)
    parent.config = config
    parent.language = "uz"
    parent.call_id = "lk-call-abc"
    parent.call_db_id = "db-call-abc"
    parent.caller_phone = "+923901234567"
    parent.agent_phone = "+923781225397"
    parent.agent_identity = "agent-id-xyz"
    parent.transferred = False
    parent.murojaat_id = None
    parent.conversation_history = [
        {"role": "user", "content": "murojaat qilmoqchiman"},
        {"role": "assistant", "content": "Albatta, ma'lumotlarni so'rayman."},
    ]
    parent._platform_client = AsyncMock()
    parent._job_context = MagicMock()
    parent._caller_history = MagicMock(last_language="uz", has_history=True)
    return parent


def _make_sub_agent(cfg, parent):
    """Bare AppealAgent wired up for `_create_main_agent` tests.

    `Agent.chat_ctx` is a read-only property; bypass it by patching
    the descriptor on the class for the duration of the test.
    """
    from agents.sub_agents.appeal import AppealAgent

    sub = AppealAgent.__new__(AppealAgent)
    sub._parent_config = cfg
    sub._parent_agent = parent
    sub._platform_client = parent._platform_client
    sub._parent_job_context = parent._job_context
    return sub


class TestSubAgentReturnStatePreservation:
    """Verify BaseSubAgent._create_main_agent forwards the right kwargs
    to AgentFactory.create_agent and copies call state from the parent.

    We stub get_factory() so the test doesn't need real STT/TTS/LLM
    credentials — we only care that the sub-agent passes the correct
    arguments and writes the right state to the returned agent.
    """

    def _run(self, cfg, sub_agent):
        """Invoke _create_main_agent with a stubbed factory and return
        (returned_agent, factory_kwargs_captured)."""
        from agents.sub_agents.appeal import AppealAgent

        stub_agent = MagicMock()
        stub_agent.conversation_history = []

        captured = {}

        def fake_create_agent(config, **kwargs):
            captured.update(kwargs)
            captured["__config__"] = config
            return stub_agent

        stub_factory = MagicMock()
        stub_factory.create_agent = fake_create_agent

        # chat_ctx is a read-only property on the base Agent class; return
        # None from it without needing a real livekit activity.
        with (
            patch.object(AppealAgent, "chat_ctx", new=property(lambda self: None)),
            patch("agents.factory.get_factory", return_value=stub_factory),
        ):
            returned = sub_agent._create_main_agent()

        return returned, captured

    def test_returned_main_agent_is_marked_post_handoff(self):
        """Without is_post_handoff=True, on_enter would trigger
        generate_reply and Gemini would hallucinate another
        transfer_to_appeal call — the exact bug in the call log."""
        cfg = _make_config()
        parent = _make_parent_agent(cfg)
        sub = _make_sub_agent(cfg, parent)

        _, captured = self._run(cfg, sub)

        assert captured["is_post_handoff"] is True

    def test_forwards_platform_and_job_context(self):
        """platform_client and job_context must reach the new main agent.
        Without them, the subsequent escalate_to_human silently no-ops
        because do_forward_to_operator has no client or room."""
        cfg = _make_config()
        parent = _make_parent_agent(cfg)
        sub = _make_sub_agent(cfg, parent)

        _, captured = self._run(cfg, sub)

        assert captured["platform_client"] is parent._platform_client
        assert captured["job_context"] is parent._job_context
        assert captured["current_language"] == "uz"

    def test_copies_call_state_onto_returned_agent(self):
        """call_db_id, caller_phone, caller_history, etc. must land on
        the new agent so the eventual PATCH /calls and operator handoff
        know which call record to update."""
        cfg = _make_config()
        parent = _make_parent_agent(cfg)
        sub = _make_sub_agent(cfg, parent)

        returned, _ = self._run(cfg, sub)

        assert returned.call_db_id == "db-call-abc"
        assert returned.caller_phone == "+923901234567"
        assert returned.agent_phone == "+923781225397"
        assert returned.agent_identity == "agent-id-xyz"
        assert returned._caller_history is parent._caller_history
        # conversation_history must be copied, not shared, so the
        # defunct parent doesn't get mutated by later turns.
        assert returned.conversation_history == parent.conversation_history
        assert returned.conversation_history is not parent.conversation_history


class TestOnEnterSkipsLLMInPostHandoff:
    async def test_post_handoff_on_enter_does_not_generate_reply(self):
        """on_enter must NOT call session.generate_reply when the agent
        was created as a sub-agent return target. The sub-agent already
        spoke a closing message; triggering the LLM here makes Gemini
        hallucinate another handoff."""
        from agents.tenant_agent import TenantAgent

        cfg = _make_config()
        agent = TenantAgent.__new__(TenantAgent)
        agent.config = cfg
        agent.call_id = "lk-call-xyz"
        agent.language = "uz"
        agent._is_post_handoff = True
        agent.conversation_history = []
        agent._caller_history = None
        agent._session_state = MagicMock()
        agent._error_handler = None
        agent._silence_monitor = None
        agent._pending_post_handoff_prompt = None
        mock_session = MagicMock()
        mock_session.generate_reply = AsyncMock()
        agent._session = mock_session
        # Agent.session property reads from the activity; patch on the
        # instance for the test.
        type(agent).session = property(lambda self: self._session)

        await agent.on_enter()

        mock_session.generate_reply.assert_not_called()


class TestSubAgentSafetyNetTools:
    """Sub-agents must expose escalate_to_human and end_call so the LLM
    can route the user out of a collection flow when they change their
    mind (ask for operator, say goodbye, etc.).

    In the bug log, the user inside AppealAgent kept asking for an
    operator but the sub-agent had no way to escalate — Gemini either
    tried to continue filling the form or waited for another handoff
    that never happened.
    """

    async def _run_tool(self, method_name, *args, **kwargs):
        """Instantiate a bare AppealAgent, stub its _create_main_agent,
        and invoke the named @function_tool. Returns (result, main_agent_stub).
        """
        from agents.sub_agents.appeal import AppealAgent

        cfg = _make_config()
        sub = AppealAgent.__new__(AppealAgent)
        sub._parent_config = cfg
        sub._parent_agent = _make_parent_agent(cfg)
        sub._platform_client = sub._parent_agent._platform_client

        # Stub out _create_main_agent so we don't hit VoiceFactory
        # (no Yandex creds in CI).
        stub_main = MagicMock()
        stub_main.do_forward_to_operator = AsyncMock(return_value="OPERATOR_MSG")
        stub_main.do_end_call = AsyncMock(return_value="GOODBYE_MSG")
        sub._create_main_agent = lambda: stub_main

        # Unwrap the @function_tool decorator to get at the underlying
        # async method.
        tool = getattr(AppealAgent, method_name)
        fn = getattr(tool, "fnc", None) or getattr(tool, "_fnc", None) or tool
        ctx = MagicMock()
        result = await fn(sub, ctx, *args, **kwargs)
        return result, stub_main

    async def test_request_then_confirm_returns_main_agent_and_transfer_msg(self, monkeypatch):
        """request_escalation returns reflection (string); confirm_escalation
        returns the (new_main_agent, transfer_message) handoff tuple."""
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

        # Strict reason → first call returns reflection prompt (string).
        reflection, _ = await self._run_tool(
            "request_escalation",
            reason_for_transfer="kb_miss",
        )
        assert isinstance(reflection, str)
        assert "INTERNAL CHECK" in reflection

        # Confirm fires the parent's forward via a fresh main agent.
        # _run_tool builds a fresh sub each call, so we need parent state to
        # carry the proposed reason. Replicate that by setting it manually
        # before invoking confirm:
        from agents.sub_agents.appeal import AppealAgent

        cfg = _make_config()
        sub = AppealAgent.__new__(AppealAgent)
        sub._parent_config = cfg
        sub._parent_agent = _make_parent_agent(cfg)
        sub._parent_agent._proposed_escalation_reason = "kb_miss"
        sub._parent_agent._transfer_rejection_count = 1
        sub._platform_client = sub._parent_agent._platform_client

        stub_main = MagicMock()
        stub_main.do_forward_to_operator = AsyncMock(return_value="OPERATOR_MSG")
        sub._create_main_agent = lambda: stub_main

        tool = AppealAgent.confirm_escalation
        fn = getattr(tool, "fnc", None) or getattr(tool, "_fnc", None) or tool
        result = await fn(sub, MagicMock(), reasoning="tried to help in-flow")

        assert isinstance(result, tuple) and len(result) == 2
        new_agent, msg = result
        assert new_agent is stub_main
        assert msg == "OPERATOR_MSG"
        stub_main.do_forward_to_operator.assert_awaited_once()
        # State cleared on parent after successful confirm.
        assert sub._parent_agent._proposed_escalation_reason is None
        assert sub._parent_agent._transfer_rejection_count == 0

    async def test_end_call_returns_main_agent_and_goodbye(self):
        result, stub_main = await self._run_tool("end_call")
        new_agent, msg = result
        assert new_agent is stub_main
        assert msg == "GOODBYE_MSG"
        stub_main.do_end_call.assert_awaited_once()

    async def test_confirm_from_sub_agent_survives_do_forward_exception(self, monkeypatch):
        """If do_forward_to_operator raises, we still hand back to main
        with a usable transfer message — the user should never be left
        stuck in the sub-agent on an internal error. The fallback is the
        configured transfer_message."""
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

        from agents.sub_agents.appeal import AppealAgent

        cfg = _make_config()
        sub = AppealAgent.__new__(AppealAgent)
        sub._parent_config = cfg
        sub._parent_agent = _make_parent_agent(cfg)
        # Simulate the prior request_escalation having already proposed.
        sub._parent_agent._proposed_escalation_reason = "kb_miss"
        sub._parent_agent._transfer_rejection_count = 1
        sub._platform_client = sub._parent_agent._platform_client

        stub_main = MagicMock()
        stub_main.do_forward_to_operator = AsyncMock(side_effect=RuntimeError("backend down"))
        sub._create_main_agent = lambda: stub_main

        tool = AppealAgent.confirm_escalation
        fn = getattr(tool, "fnc", None) or getattr(tool, "_fnc", None) or tool
        result = await fn(sub, MagicMock(), reasoning="x")

        new_agent, msg = result
        assert new_agent is stub_main
        # Fallback: the configured transfer_message
        assert msg == cfg.transfer.transfer_message


class TestHandoffToolDescription:
    def test_appeal_handoff_uses_description_not_sub_agent_prompt(self):
        """Tool docstring must tell the LLM WHEN to call it, not rehash
        the sub-agent's own system prompt. With the old behavior the LLM
        saw 'You are collecting appeal data...' as the description and
        couldn't distinguish when to pick this tool vs escalate_to_human."""
        from agents.factory import AgentFactory
        from config.schema import SubAgentConfig

        factory = AgentFactory(tool_registry=MagicMock())
        cfg = _make_config()
        sub_cfg = SubAgentConfig(
            type="appeal",
            instructions="Sen ma'lumot yig'yapsan",  # sub-agent's own prompt
            description="Call ONLY for formal complaints.",
        )

        class _DummySubAgent:
            def __init__(self, **kw):  # noqa: D401 — test stub
                pass

        tool = factory._create_handoff_tool("appeal", sub_cfg, _DummySubAgent, cfg)
        # function_tool wraps the async callable — introspect its info.
        info = getattr(tool, "info", None) or getattr(tool, "_info", None)
        description = None
        if info is not None:
            description = getattr(info, "description", None) or getattr(
                info, "raw_description", None
            )
        if description is None:
            # Fall back to the underlying callable's docstring
            fn = getattr(tool, "fnc", None) or getattr(tool, "_fnc", None)
            if fn is not None:
                description = fn.__doc__

        assert description is not None
        assert "formal complaint" in description.lower()
        assert "yig'yapsan" not in description  # sub-agent prompt leakage


class TestHandoffForwardsLanguage:
    """When a Russian-speaking Paynet caller triggers transfer_to_appeal,
    the AppealAgent must be constructed with language="ru" so prompts
    resolve to Russian instead of the tenant default (which may differ)."""

    def test_do_handoff_passes_parent_language_to_sub_agent(self):
        from unittest.mock import MagicMock

        from agents.factory import AgentFactory
        from config.schema import FieldConfig, SubAgentConfig, TenantConfig

        factory = AgentFactory(tool_registry=MagicMock())
        cfg = TenantConfig.model_validate(
            {
                "tenant": {"id": "paynet-t", "slug": "paynet", "name": "Paynet"},
                "languages": {"default": "ru", "available": ["ru", "uz"]},
            }
        )
        sub_cfg = SubAgentConfig(
            type="appeal",
            fields=[FieldConfig(name="content", prompts={"uz": "U", "ru": "R"})],
            messages={"ru": {"start": "S"}},
        )

        captured = {}

        class _DummySubAgent:
            def __init__(self, **kw):
                captured.update(kw)

        tool = factory._create_handoff_tool("appeal", sub_cfg, _DummySubAgent, cfg)

        # Simulate the main agent having already switched to "uz" mid-call.
        main_agent = MagicMock()
        main_agent.chat_ctx = None
        main_agent._platform_client = None
        main_agent.call_db_id = None
        main_agent.caller_phone = None
        main_agent._job_context = None
        main_agent.language = "uz"

        ctx = MagicMock()
        ctx.session.current_agent = main_agent

        # Unwrap the function_tool wrapper.
        fn = getattr(tool, "fnc", None) or getattr(tool, "_fnc", None) or tool
        import asyncio

        asyncio.run(fn(ctx))
        assert captured.get("language") == "uz"
        assert captured.get("messages") == {"ru": {"start": "S"}}

    def test_do_handoff_resolves_transfer_message_for_caller_language(self):
        """When transfer_messages is populated, the spoken handoff message
        must match the main agent's runtime language (not tenant default)."""
        import asyncio
        from unittest.mock import MagicMock

        from agents.factory import AgentFactory
        from config.schema import FieldConfig, SubAgentConfig, TenantConfig

        factory = AgentFactory(tool_registry=MagicMock())
        cfg = TenantConfig.model_validate(
            {
                "tenant": {"id": "paynet-t", "slug": "paynet", "name": "Paynet"},
                "languages": {"default": "uz", "available": ["ru", "uz"]},
            }
        )
        sub_cfg = SubAgentConfig(
            type="appeal",
            fields=[FieldConfig(name="content", prompts={"uz": "U", "ru": "R"})],
            transfer_messages={"uz": "UZ-MSG", "ru": "RU-MSG"},
        )

        class _DummySubAgent:
            def __init__(self, **kw):
                pass

        tool = factory._create_handoff_tool("appeal", sub_cfg, _DummySubAgent, cfg)

        main_agent = MagicMock()
        main_agent.chat_ctx = None
        main_agent._platform_client = None
        main_agent.call_db_id = None
        main_agent.caller_phone = None
        main_agent._job_context = None
        main_agent.language = "ru"  # caller switched to Russian

        ctx = MagicMock()
        ctx.session.current_agent = main_agent

        fn = getattr(tool, "fnc", None) or getattr(tool, "_fnc", None) or tool
        _, msg = asyncio.run(fn(ctx))
        assert msg == "RU-MSG"
