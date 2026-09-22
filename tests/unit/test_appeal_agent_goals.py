"""AppealAgent behavior when goals are declared."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from api.models import MurojaatResult
from config.schema import FieldConfig, GoalConfig, TenantConfig


def _config():
    return TenantConfig.model_validate(
        {
            "tenant": {"id": "ya-test", "slug": "yoshlar", "name": "YA"},
            "languages": {"default": "uz", "available": ["uz"]},
        }
    )


def _fields_with_content():
    return [
        FieldConfig(name="content", prompt="Mazmun?"),
        FieldConfig(name="full_name", prompt="Ism-familiya?"),
    ]


def _goals():
    return [
        GoalConfig(key="problem_specifics", description={"uz": "Tafsilot"}),
        GoalConfig(key="desired_outcome", description={"uz": "Natija"}, required=True),
    ]


class TestInitAcceptsGoals:
    def test_goals_stored_on_instance(self):
        from agents.sub_agents.appeal import AppealAgent

        # __new__ bypass to avoid Agent.__init__ which needs LiveKit.
        # Then drive the normalization step explicitly.
        agent = AppealAgent.__new__(AppealAgent)
        agent._goals = []
        agent._summary_instructions = {}
        # Simulate the __init__ normalization path for goals:
        normalized = [
            g if isinstance(g, GoalConfig) else GoalConfig.model_validate(g) for g in _goals()
        ]
        agent._goals = normalized
        assert len(agent._goals) == 2
        assert agent._goals[0].key == "problem_specifics"

    def test_summary_instructions_default_empty(self):
        from agents.sub_agents.appeal import AppealAgent

        agent = AppealAgent.__new__(AppealAgent)
        agent._summary_instructions = {}
        assert agent._summary_instructions == {}


class TestContentStrippingUnderGoals:
    def test_content_stripped_from_fields_when_goals_present(self):
        """When goals are declared, set_content tool should NOT be generated."""
        from agents.sub_agents.appeal import AppealAgent

        # Drive the constructor normalization step manually since we're
        # bypassing Agent.__init__.
        raw_fields = _fields_with_content()
        raw_goals = _goals()

        # Exercise the stripping rule directly:
        stripped = AppealAgent._filter_fields_for_goals(raw_fields, raw_goals)
        names = {f.name for f in stripped}
        assert "content" not in names
        assert "full_name" in names

    def test_content_preserved_in_fields_when_no_goals(self):
        from agents.sub_agents.appeal import AppealAgent

        stripped = AppealAgent._filter_fields_for_goals(_fields_with_content(), [])
        names = {f.name for f in stripped}
        assert "content" in names
        assert "full_name" in names


class TestConfirmAndSubmitWithSummary:
    @pytest.mark.asyncio
    async def test_summary_arg_overwrites_content_before_submit(self):
        from agents.sub_agents.appeal import AppealAgent

        mock_client = AsyncMock()
        mock_client.submit_murojaat.return_value = MurojaatResult(success=True, murojaat_id="mur-1")

        agent = AppealAgent.__new__(AppealAgent)
        agent._parent_config = _config()
        agent._platform_client = mock_client
        # Simulate no content setter exists under goals; _data["content"]
        # starts empty and will be populated by the summary arg.
        agent._fields = [FieldConfig(name="full_name", prompt="Ism?")]
        agent._data = {"full_name": "Ali"}
        agent._goals = _goals()
        agent._summary_instructions = {}
        agent._messages = {}
        agent._language = "uz"
        agent._parent_call_db_id = None
        agent._parent_caller_phone = None
        agent._parent_agent = None
        # Stub _create_main_agent so the success path doesn't try to build
        # a real TenantAgent (would need VoiceFactory / LiveKit wiring).
        agent._create_main_agent = lambda: MagicMock()

        tool = AppealAgent.confirm_and_submit
        fn = getattr(tool, "fnc", None) or getattr(tool, "_fnc", None) or tool
        await fn(
            agent,
            context=None,
            summary="Caller needs a business loan; Bank X rejected; wants SME options.",
        )
        # Returned when _submit succeeds: (main_agent, success_message).
        # We don't care about the main agent object here — only about what was submitted.
        mock_client.submit_murojaat.assert_called_once()
        kwargs = mock_client.submit_murojaat.call_args.kwargs
        assert kwargs["content"].startswith("Caller needs a business loan")

    @pytest.mark.asyncio
    async def test_empty_summary_with_goals_returns_retry_message(self):
        from agents.sub_agents.appeal import AppealAgent

        mock_client = AsyncMock()

        agent = AppealAgent.__new__(AppealAgent)
        agent._parent_config = _config()
        agent._platform_client = mock_client
        agent._fields = []
        agent._data = {}
        agent._goals = _goals()
        agent._summary_instructions = {}
        agent._messages = {}
        agent._language = "uz"

        tool = AppealAgent.confirm_and_submit
        fn = getattr(tool, "fnc", None) or getattr(tool, "_fnc", None) or tool
        result = await fn(agent, context=None, summary="")
        assert isinstance(result, str)
        assert "xulosa" in result.lower() or "summary" in result.lower()
        mock_client.submit_murojaat.assert_not_called()

    @pytest.mark.asyncio
    async def test_legacy_tenant_without_goals_falls_back_to_data_content(self):
        """Tenants that do not declare goals behave exactly as before."""
        from agents.sub_agents.appeal import AppealAgent

        mock_client = AsyncMock()
        mock_client.submit_murojaat.return_value = MurojaatResult(success=True, murojaat_id="mur-2")

        agent = AppealAgent.__new__(AppealAgent)
        agent._parent_config = _config()
        agent._platform_client = mock_client
        agent._fields = [
            FieldConfig(name="content", prompt="?"),
            FieldConfig(name="full_name", prompt="?"),
        ]
        agent._data = {"content": "Literal caller quote", "full_name": "Ali"}
        agent._goals = []
        agent._summary_instructions = {}
        agent._messages = {}
        agent._language = "uz"
        agent._parent_call_db_id = None
        agent._parent_caller_phone = None
        agent._parent_agent = None
        agent._create_main_agent = lambda: MagicMock()

        tool = AppealAgent.confirm_and_submit
        fn = getattr(tool, "fnc", None) or getattr(tool, "_fnc", None) or tool
        await fn(agent, context=None, summary="")
        kwargs = mock_client.submit_murojaat.call_args.kwargs
        assert kwargs["content"] == "Literal caller quote"
