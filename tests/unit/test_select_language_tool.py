"""Tests for the select_language platform tool factory."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.schema import LanguagesConfig, TenantConfig, TenantIdentity


@pytest.fixture
def multilingual_config():
    return TenantConfig(
        tenant=TenantIdentity(id="t1", slug="paynet"),
        languages=LanguagesConfig(default="ru", available=["ru", "uz"], ask_on_start=True),
    )


def test_factory_returns_none_for_single_language_tenant():
    """Tool is only meaningful when >1 languages are available."""
    from tools.platform.select_language import create_select_language_tool

    config = TenantConfig(tenant=TenantIdentity(id="t1"))  # defaults: single lang
    assert create_select_language_tool(config) is None


def test_factory_returns_tool_for_multilingual(multilingual_config):
    from tools.platform.select_language import create_select_language_tool

    tool = create_select_language_tool(multilingual_config)
    assert tool is not None


async def test_tool_rejects_unknown_language(multilingual_config):
    from tools.platform.select_language import create_select_language_tool

    tool = create_select_language_tool(multilingual_config)

    ctx = MagicMock()
    ctx.session.current_agent = MagicMock()
    result = await tool.__wrapped__(ctx, lang="en")
    # Should return an error-ish string (no handoff tuple).
    assert isinstance(result, str)
    assert "en" in result.lower() or "не" in result.lower() or "yo'q" in result.lower()


async def test_tool_same_language_returns_confirmation_only(multilingual_config):
    from tools.platform.select_language import create_select_language_tool

    tool = create_select_language_tool(multilingual_config)

    ctx = MagicMock()
    ctx.session.current_agent = MagicMock()
    ctx.session.current_agent.language = "ru"
    result = await tool.__wrapped__(ctx, lang="ru")
    # No handoff if already in that language.
    assert isinstance(result, str)
    assert result  # non-empty confirmation


async def test_tool_stops_old_silence_monitor_on_handoff(multilingual_config):
    from tools.platform.select_language import create_select_language_tool

    tool = create_select_language_tool(multilingual_config)

    ctx = MagicMock()
    old_agent = MagicMock()
    old_agent.language = "ru"
    old_agent.chat_ctx = None
    old_agent._job_context = None
    old_agent._platform_client = None
    old_agent._stop_silence_monitor = MagicMock()
    ctx.session.current_agent = old_agent

    fake_new_agent = MagicMock()
    fake_factory = MagicMock()
    fake_factory.create_agent.return_value = fake_new_agent

    with (
        patch("agents.factory.get_factory", return_value=fake_factory),
        patch("tools.platform.select_language.Path.is_file", return_value=False),
    ):
        result = await tool.__wrapped__(ctx, lang="uz")

    assert isinstance(result, tuple)
    assert result[0] is fake_new_agent
    # No pre-rendered audio played (Path.is_file=False) → tool_message
    # carries both confirmation + help; _pending_post_handoff_prompt is NOT
    # set (would cause double-speaking otherwise).
    assert (
        result[1]
        == "Yaxshi, suhbatni o'zbek tilida davom ettiraman. Sizga qanday yordam bera olaman?"
    )
    old_agent._stop_silence_monitor.assert_called_once()


async def test_tool_returns_help_prompt_only_when_prerendered_confirmation_played(
    multilingual_config,
):
    from tools.platform.select_language import create_select_language_tool

    tool = create_select_language_tool(multilingual_config)

    ctx = MagicMock()
    old_agent = MagicMock()
    old_agent.language = "uz"
    old_agent.chat_ctx = None
    old_agent._job_context = None
    old_agent._platform_client = None
    old_agent._stop_silence_monitor = MagicMock()
    ctx.session.current_agent = old_agent
    multilingual_config.personality.language_switch_audio = {"ru": "dummy.wav"}

    handle = MagicMock()
    handle.wait_for_playout = AsyncMock(return_value=None)
    ctx.session.say.return_value = handle

    fake_new_agent = MagicMock()
    fake_factory = MagicMock()
    fake_factory.create_agent.return_value = fake_new_agent

    with (
        patch("agents.factory.get_factory", return_value=fake_factory),
        patch("tools.platform.select_language.Path.is_file", return_value=True),
    ):
        result = await tool.__wrapped__(ctx, lang="ru")

    assert isinstance(result, tuple)
    assert result[0] is fake_new_agent
    # Pre-rendered confirmation was played → tool_message is empty
    # (new_agent.on_enter will speak the help prompt from the pending-attr,
    # which avoids a double "how can I help?" after the confirmation audio).
    assert result[1] == ""
    assert fake_new_agent._pending_post_handoff_prompt == "Чем я могу вам помочь?"
