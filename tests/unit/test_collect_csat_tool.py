"""Tests for the collect_csat tool (NAV-156).

The tool's contract:
- Valid 1-5 ratings stash on agent._pending_csat and return a localized thanks.
- Out-of-range / non-int values do NOT stash and return a localized retry prompt.
- Language stash + thanks honor agent.language (uz default, ru when set).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from config.schema import TenantConfig
from tools.platform.collect_csat import create_collect_csat_tool


def _make_config():
    return TenantConfig.model_validate({"tenant": {"id": "t1", "slug": "test", "name": "Test"}})


def _unwrap(tool):
    return getattr(tool, "fnc", None) or getattr(tool, "_fnc", None) or tool


def _ctx_with_lang(language: str | None):
    agent = MagicMock()
    # MagicMock would auto-create .language as another MagicMock, which breaks
    # the `or "uz"` fallback. Set it explicitly.
    if language is None:
        del agent.language
    else:
        agent.language = language
    ctx = MagicMock()
    ctx.session.current_agent = agent
    return ctx, agent


@pytest.mark.asyncio
@pytest.mark.parametrize("rating", [1, 2, 3, 4, 5])
async def test_valid_rating_stashes_on_agent_and_returns_uzbek_thanks(rating):
    fn = _unwrap(create_collect_csat_tool(_make_config()))
    ctx, agent = _ctx_with_lang("uz")

    result = await fn(ctx, rating=rating)

    assert agent._pending_csat == rating
    assert result == "Bahoyingiz uchun rahmat!"


@pytest.mark.asyncio
async def test_valid_rating_returns_russian_thanks_when_agent_language_ru():
    fn = _unwrap(create_collect_csat_tool(_make_config()))
    ctx, agent = _ctx_with_lang("ru")

    result = await fn(ctx, rating=4)

    assert agent._pending_csat == 4
    assert result == "Спасибо за вашу оценку!"


@pytest.mark.asyncio
@pytest.mark.parametrize("rating", [0, 6, -1, 100])
async def test_out_of_range_rating_is_rejected_and_does_not_stash(rating):
    fn = _unwrap(create_collect_csat_tool(_make_config()))
    ctx, agent = _ctx_with_lang("uz")

    result = await fn(ctx, rating=rating)

    assert not hasattr(agent, "_pending_csat") or agent._pending_csat != rating
    assert "1 dan 5 gacha" in result


@pytest.mark.asyncio
async def test_out_of_range_rejection_localizes_to_russian():
    fn = _unwrap(create_collect_csat_tool(_make_config()))
    ctx, _ = _ctx_with_lang("ru")

    result = await fn(ctx, rating=7)

    assert "от 1 до 5" in result


@pytest.mark.asyncio
async def test_missing_language_defaults_to_uzbek():
    fn = _unwrap(create_collect_csat_tool(_make_config()))
    ctx, agent = _ctx_with_lang(None)

    result = await fn(ctx, rating=3)

    assert agent._pending_csat == 3
    assert result == "Bahoyingiz uchun rahmat!"
