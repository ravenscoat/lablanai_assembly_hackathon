from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.tenant_agent import TenantAgent
from config.schema import PersonalityConfig, TenantConfig, TenantIdentity


def test_personality_greeting_allow_interruptions_defaults_true():
    p = PersonalityConfig()
    assert p.greeting_allow_interruptions is True


@pytest.mark.asyncio
async def test_greet_monolingual_uses_configured_interruption_flag():
    cfg = TenantConfig(
        tenant=TenantIdentity(id="t1", slug="test"),
        personality={"greeting_allow_interruptions": False},
    )
    agent = TenantAgent.__new__(TenantAgent)
    agent.config = cfg
    agent._error_handler = None

    handle = MagicMock()
    handle.wait_for_playout = AsyncMock()
    session = MagicMock()
    session.say = MagicMock(return_value=handle)
    agent._session = session
    type(agent).session = property(lambda self: self._session)

    await agent._greet_monolingual("salom")

    session.say.assert_called_once_with(
        "salom",
        add_to_chat_ctx=True,
        allow_interruptions=False,
    )
    handle.wait_for_playout.assert_awaited_once()


@pytest.mark.asyncio
async def test_prerendered_greeting_uses_configured_interruption_flag():
    cfg = TenantConfig.model_validate(
        {
            "tenant": {"id": "t1", "slug": "paynet"},
            "languages": {"available": ["ru", "uz"], "default": "uz"},
            "personality": {
                "greeting_allow_interruptions": False,
                "greeting_parts": {"ru": "privet", "uz": "salom"},
                "greeting_audio": {"ru": "ru.wav", "uz": "uz.wav"},
            },
        }
    )
    agent = TenantAgent.__new__(TenantAgent)
    agent.config = cfg
    agent._error_handler = None

    handle = MagicMock()
    handle.wait_for_playout = AsyncMock()
    session = MagicMock()
    session.say = MagicMock(return_value=handle)
    agent._session = session
    type(agent).session = property(lambda self: self._session)

    with patch.object(
        TenantAgent,
        "_load_wav_as_frames",
        new=AsyncMock(return_value=object()),
    ):
        await agent._play_prerendered_greeting()

    assert session.say.call_count == 2
    for call in session.say.call_args_list:
        assert call.kwargs["allow_interruptions"] is False
        assert call.kwargs["add_to_chat_ctx"] is True
