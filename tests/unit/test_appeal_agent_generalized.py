"""Tests for the generalized (dynamic-field, language-aware) AppealAgent."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from api.models import MurojaatResult
from config.schema import FieldConfig, TenantConfig


def _paynet_like_config():
    """Minimal TenantConfig for a 2-field Paynet-style intake."""
    return TenantConfig.model_validate(
        {
            "tenant": {"id": "paynet-test", "slug": "paynet", "name": "Paynet"},
            "languages": {"default": "ru", "available": ["ru", "uz"]},
        }
    )


def _ya_like_config():
    """Minimal TenantConfig for a 6-field Youth-Agency-style intake."""
    return TenantConfig.model_validate(
        {
            "tenant": {"id": "ya-test", "slug": "yoshlar", "name": "YA"},
            "languages": {"default": "uz", "available": ["uz"]},
        }
    )


def _paynet_fields() -> list[FieldConfig]:
    return [
        FieldConfig(
            name="content",
            prompts={"ru": "Опишите проблему.", "uz": "Muammoni ayting."},
        ),
        FieldConfig(
            name="full_name",
            prompts={"ru": "Ваше имя?", "uz": "Ismingiz?"},
        ),
    ]


def _ya_fields() -> list[FieldConfig]:
    return [
        FieldConfig(name="content", prompt="Mazmun?"),
        FieldConfig(name="full_name", prompt="Ism-familiya?"),
        FieldConfig(name="age", prompt="Yosh?", validation="14-30"),
        FieldConfig(name="region", prompt="Viloyat?"),
        FieldConfig(name="district", prompt="Tuman?"),
        FieldConfig(name="position", prompt="Lavozim?"),
    ]


class TestDynamicSetterTools:
    def test_paynet_config_registers_two_setter_tools(self):
        from agents.sub_agents.appeal import AppealAgent

        agent = AppealAgent.__new__(AppealAgent)
        # Call the dynamic-tool builder directly without going through
        # Agent.__init__ (which would require a LiveKit session).
        tools = AppealAgent._build_setter_tools(agent, _paynet_fields())
        names = {getattr(t, "name", None) or getattr(t, "__name__", "") for t in tools}
        # Each FieldConfig.name is prefixed with "set_"
        assert "set_content" in names
        assert "set_full_name" in names
        # No lingering YA-only setters
        assert "set_age" not in names
        assert "set_region" not in names
        assert len(tools) == 2

    def test_ya_config_registers_six_setter_tools(self):
        from agents.sub_agents.appeal import AppealAgent

        agent = AppealAgent.__new__(AppealAgent)
        tools = AppealAgent._build_setter_tools(agent, _ya_fields())
        names = {getattr(t, "name", None) or getattr(t, "__name__", "") for t in tools}
        assert names == {
            "set_content",
            "set_full_name",
            "set_age",
            "set_region",
            "set_district",
            "set_position",
        }

    def test_setter_tools_registered_on_fully_constructed_agent(self):
        """Regression: BaseSubAgent must forward tools= to the base Agent.
        Without this, dynamic setter tools are silently dropped and the
        LLM inside AppealAgent has no way to record collected fields."""
        from agents.sub_agents.appeal import AppealAgent

        agent = AppealAgent(
            fields=_paynet_fields(),
            parent_config=_paynet_like_config(),
        )
        # Agent.tools is the authoritative public list the LLM sees.
        # FunctionTool exposes name via .info.name or .__name__; use the
        # same pattern as the other tests in this file.
        tool_names = {
            (getattr(t.info, "name", None) if hasattr(t, "info") else None)
            or getattr(t, "__name__", None)
            or getattr(t, "name", None)
            for t in agent.tools
        }
        assert "set_content" in tool_names, f"set_content missing from {tool_names}"
        assert "set_full_name" in tool_names, f"set_full_name missing from {tool_names}"

    def test_setter_tool_schema_has_typed_value_parameter(self):
        """Regression: _bound trampoline must carry the annotation so LiveKit
        can build the JSON schema. Without this, schema generation raised
        KeyError at session start and the LLM saw zero setter tools."""
        from agents.sub_agents.appeal import AppealAgent

        agent = AppealAgent.__new__(AppealAgent)
        tools = AppealAgent._build_setter_tools(
            agent,
            _paynet_fields()
            + [
                FieldConfig(name="age", prompt="Yosh?", validation="14-30"),
            ],
        )

        # Find each tool and pull its raw callable's annotations.
        for tool in tools:
            fn = getattr(tool, "fnc", None) or getattr(tool, "_fnc", None) or tool
            annotations = getattr(fn, "__annotations__", {})
            # Tool name lives on __name__ (FunctionTool proxies dunder attrs to
            # the wrapped callable) — not on a plain .name attribute.
            tool_name = getattr(tool, "__name__", None) or getattr(tool, "name", "?")
            assert "value" in annotations, f"tool {tool_name} is missing 'value' annotation"
            if tool_name == "set_age":
                assert annotations["value"] is int
            else:
                assert annotations["value"] is str


class TestLanguageAwarePrompts:
    def test_next_field_prompt_resolves_russian(self):
        from agents.sub_agents.appeal import AppealAgent

        agent = AppealAgent.__new__(AppealAgent)
        agent._fields = _paynet_fields()
        agent._data = {}
        agent._language = "ru"
        agent._messages = {}
        # First missing field is `content`.
        assert "Опишите проблему" in agent._next_field_prompt("")

    def test_next_field_prompt_resolves_uzbek(self):
        from agents.sub_agents.appeal import AppealAgent

        agent = AppealAgent.__new__(AppealAgent)
        agent._fields = _paynet_fields()
        agent._data = {"content": "issue"}  # content already collected
        agent._language = "uz"
        agent._messages = {}
        assert "Ismingiz" in agent._next_field_prompt("content")

    def test_summary_uses_configured_message(self):
        from agents.sub_agents.appeal import AppealAgent

        agent = AppealAgent.__new__(AppealAgent)
        agent._fields = _paynet_fields()
        agent._data = {"content": "платёж", "full_name": "Иванов"}
        agent._language = "ru"
        agent._messages = {
            "ru": {"summary": "Данные: {content} / {full_name}. OK?"},
        }
        prompt = agent._next_field_prompt("full_name")
        assert prompt == "Данные: платёж / Иванов. OK?"


class TestIntRangeValidation:
    def test_validate_int_range_accepts_in_range(self):
        from agents.sub_agents.appeal import AppealAgent

        assert AppealAgent._validate_int_range("14-30", 20) is None  # no error

    def test_validate_int_range_rejects_out_of_range(self):
        from agents.sub_agents.appeal import AppealAgent

        err = AppealAgent._validate_int_range("14-30", 45)
        assert err is not None
        assert "14" in err and "30" in err

    def test_validate_int_range_empty_skips_check(self):
        from agents.sub_agents.appeal import AppealAgent

        assert AppealAgent._validate_int_range("", 999) is None


class TestSubmitSentinelDefaults:
    @pytest.mark.asyncio
    async def test_paynet_submit_fills_sentinels(self):
        """Paynet collects only content + full_name; backend still gets yosh/viloyat/tuman/lavozim."""
        from agents.sub_agents.appeal import AppealAgent

        mock_client = AsyncMock()
        mock_client.submit_murojaat.return_value = MurojaatResult(
            success=True, murojaat_id="mur-paynet-1"
        )

        agent = AppealAgent.__new__(AppealAgent)
        agent._parent_config = _paynet_like_config()
        agent._platform_client = mock_client
        agent._fields = _paynet_fields()
        agent._data = {"content": "Платёж не прошёл", "full_name": "Иванов"}
        agent._parent_call_db_id = "db-call-xyz"
        agent._parent_caller_phone = "+923901234567"
        agent._parent_agent = None

        ok = await agent._submit()
        assert ok is True
        mock_client.submit_murojaat.assert_called_once()
        kwargs = mock_client.submit_murojaat.call_args.kwargs
        assert kwargs["content"] == "Платёж не прошёл"
        assert kwargs["full_name"] == "Иванов"
        assert kwargs["age"] == 0
        assert kwargs["region"] == "-"
        assert kwargs["district"] == "-"
        assert kwargs["position"] == "Mijoz"
        assert kwargs["tenant_id"] == "paynet-test"
        assert kwargs["call_id"] == "db-call-xyz"
        assert kwargs["caller_phone"] == "+923901234567"

    @pytest.mark.asyncio
    async def test_ya_submit_passes_collected_values_not_sentinels(self):
        """Youth Agency collects all 6 fields; sentinels must not override them."""
        from agents.sub_agents.appeal import AppealAgent

        mock_client = AsyncMock()
        mock_client.submit_murojaat.return_value = MurojaatResult(
            success=True, murojaat_id="mur-ya-1"
        )

        agent = AppealAgent.__new__(AppealAgent)
        agent._parent_config = _ya_like_config()
        agent._platform_client = mock_client
        agent._fields = _ya_fields()
        agent._data = {
            "content": "Korrupsiya",
            "full_name": "Ali Valiyev",
            "age": 22,
            "region": "Toshkent",
            "district": "Chilonzor",
            "position": "Talaba",
        }
        agent._parent_call_db_id = None
        agent._parent_caller_phone = None
        agent._parent_agent = None

        ok = await agent._submit()
        assert ok is True
        kwargs = mock_client.submit_murojaat.call_args.kwargs
        assert kwargs["age"] == 22
        assert kwargs["region"] == "Toshkent"
        assert kwargs["district"] == "Chilonzor"
        assert kwargs["position"] == "Talaba"
