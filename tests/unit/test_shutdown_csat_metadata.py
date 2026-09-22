"""Shutdown PATCH includes metadata.csat when collect_csat fired (NAV-156).

Mirrors the structure of test_shutdown_transfer_metadata.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.schema import TenantConfig


def _make_config():
    return TenantConfig.model_validate({"tenant": {"id": "test", "slug": "test", "name": "Test"}})


def _make_agent(*, csat=None):
    agent = MagicMock()
    agent.call_db_id = "507f1f77bcf86cd799439011"
    agent.transferred = False
    agent.call_start_time = None
    agent.conversation_history = []
    agent.agent_identity = "ai"
    agent.language = "uz"
    agent._pending_transfer_metadata = {}
    if csat is None:
        # Force the attribute to be absent so getattr(default=None) hits.
        del agent._pending_csat
    else:
        agent._pending_csat = csat
    return agent


class _FakeCtx:
    def __init__(self):
        self.room = MagicMock()
        self._cb = None

    def add_shutdown_callback(self, fn):
        self._cb = fn
        return fn


async def _run_shutdown(agent, config):
    from lifecycle.shutdown import register_shutdown_callbacks

    ctx = _FakeCtx()
    with patch(
        "lifecycle.call_tracker.update_call",
        new=AsyncMock(return_value=True),
    ) as mock_update:
        register_shutdown_callbacks(ctx, agent, config)
        await ctx._cb()
    return mock_update


@pytest.mark.asyncio
@pytest.mark.parametrize("rating", [1, 2, 3, 4, 5])
async def test_metadata_csat_is_set_when_pending_csat_present(rating):
    mock_update = await _run_shutdown(_make_agent(csat=rating), _make_config())

    kwargs = mock_update.await_args.kwargs
    assert kwargs["metadata"]["csat"] == rating


@pytest.mark.asyncio
async def test_metadata_csat_absent_when_no_pending_csat():
    mock_update = await _run_shutdown(_make_agent(csat=None), _make_config())

    kwargs = mock_update.await_args.kwargs
    assert "csat" not in kwargs["metadata"]


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_value", [0, 6, -3, "5", 3.5, None])
async def test_metadata_csat_absent_when_pending_csat_is_invalid(bad_value):
    """Defense in depth: even if something writes a bad value to
    _pending_csat directly, shutdown refuses to include it."""
    agent = _make_agent(csat=1)  # start valid so attribute exists
    agent._pending_csat = bad_value

    mock_update = await _run_shutdown(agent, _make_config())

    kwargs = mock_update.await_args.kwargs
    assert "csat" not in kwargs["metadata"]
