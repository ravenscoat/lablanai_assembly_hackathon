"""Shutdown handler nests transfer metadata under metadata.transfer (NAV-150)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.schema import TenantConfig


def _make_config():
    return TenantConfig.model_validate({"tenant": {"id": "test", "slug": "test", "name": "Test"}})


@pytest.mark.asyncio
async def test_shutdown_nests_pending_transfer_metadata_under_transfer_key():
    """When agent._pending_transfer_metadata has content, it is added to
    the PATCH body under metadata['transfer']."""
    from lifecycle.shutdown import register_shutdown_callbacks

    config = _make_config()

    agent = MagicMock()
    agent.call_db_id = "507f1f77bcf86cd799439011"
    agent.transferred = True
    agent.call_start_time = None
    agent.conversation_history = []
    agent.agent_identity = "ai"
    agent.language = "uz"
    agent._pending_transfer_metadata = {
        "caller_request": "grant pulim kelmayapti",
        "ai_attempt_summary": "kb answered 5-7 day window; caller insisted",
        "reason_for_transfer": "user_dissatisfied",
        "loop_break": False,
    }

    registered = {}

    class FakeCtx:
        def add_shutdown_callback(self, fn):
            registered["cb"] = fn

        room = MagicMock()

    ctx = FakeCtx()

    with patch(
        "lifecycle.call_tracker.update_call",
        new=AsyncMock(return_value=True),
    ) as mock_update:
        register_shutdown_callbacks(ctx, agent, config)
        assert "cb" in registered
        await registered["cb"]()

    assert mock_update.await_count == 1
    kwargs = mock_update.await_args.kwargs
    assert "metadata" in kwargs
    assert kwargs["metadata"]["transfer"] == agent._pending_transfer_metadata


@pytest.mark.asyncio
async def test_shutdown_omits_transfer_key_when_buffer_empty():
    """If no transfer was fired, metadata['transfer'] is absent."""
    from lifecycle.shutdown import register_shutdown_callbacks

    config = _make_config()

    agent = MagicMock()
    agent.call_db_id = "507f1f77bcf86cd799439011"
    agent.transferred = False
    agent.call_start_time = None
    agent.conversation_history = []
    agent.agent_identity = "ai"
    agent.language = "uz"
    agent._pending_transfer_metadata = {}

    registered = {}

    class FakeCtx:
        def add_shutdown_callback(self, fn):
            registered["cb"] = fn

        room = MagicMock()

    ctx = FakeCtx()

    with patch(
        "lifecycle.call_tracker.update_call",
        new=AsyncMock(return_value=True),
    ) as mock_update:
        register_shutdown_callbacks(ctx, agent, config)
        await registered["cb"]()

    kwargs = mock_update.await_args.kwargs
    assert "transfer" not in kwargs["metadata"]


@pytest.mark.asyncio
async def test_shutdown_omits_transfer_key_when_attr_missing():
    """Agent object without the buffer attribute at all (legacy case) — no crash,
    no transfer key."""
    from lifecycle.shutdown import register_shutdown_callbacks

    config = _make_config()

    # An agent object that has never been through the new __init__.
    class _LegacyAgent:
        pass

    agent = _LegacyAgent()
    agent.call_db_id = "507f1f77bcf86cd799439011"
    agent.transferred = False
    agent.call_start_time = None
    agent.conversation_history = []
    agent.agent_identity = "ai"
    agent.language = "uz"

    registered = {}

    class FakeCtx:
        def add_shutdown_callback(self, fn):
            registered["cb"] = fn

        room = MagicMock()

    ctx = FakeCtx()

    with patch(
        "lifecycle.call_tracker.update_call",
        new=AsyncMock(return_value=True),
    ) as mock_update:
        register_shutdown_callbacks(ctx, agent, config)
        await registered["cb"]()

    kwargs = mock_update.await_args.kwargs
    assert "transfer" not in kwargs["metadata"]
