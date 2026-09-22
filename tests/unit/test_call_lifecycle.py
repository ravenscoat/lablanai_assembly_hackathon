"""Tests for call lifecycle functions."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from lifecycle.call_tracker import create_call, get_caller_history, update_call


class TestCreateCall:
    async def test_delegates_to_platform_client(self):
        mock_client = AsyncMock()
        mock_client.create_call.return_value = "db-id-123"
        with patch("lifecycle.call_tracker.get_platform_client", return_value=mock_client):
            result = await create_call(
                tenant_id="youth-agency",
                caller_phone="+923901234567",
                agent_phone="+923781225394",
                call_sid="lk-room-abc",
            )
            assert result == "db-id-123"
            mock_client.create_call.assert_called_once_with(
                tenant_id="youth-agency",
                tenant_slug=None,
                call_sid="lk-room-abc",
                caller_phone="+923901234567",
                agent_phone="+923781225394",
                metadata=None,
            )


class TestUpdateCall:
    async def test_sends_all_fields(self):
        mock_client = AsyncMock()
        mock_client.update_call.return_value = True
        with patch("lifecycle.call_tracker.get_platform_client", return_value=mock_client):
            result = await update_call(
                call_db_id="db-id-123",
                tenant_id="youth-agency",
                status="completed",
                duration_seconds=120,
                transcript=[{"role": "user", "content": "salom"}],
                transcript_text="user: salom",
                ai_summary="User greeted agent",
                ended_at="2026-04-09T12:00:00+05:00",
                metadata={"tenant": "youth-agency"},
            )
            assert result is True
            kwargs = mock_client.update_call.call_args.kwargs
            assert kwargs["ai_summary"] == "User greeted agent"
            assert kwargs["ended_at"] == "2026-04-09T12:00:00+05:00"


class TestCallerHistory:
    async def test_fetches_history(self):
        from api.models import CallerHistory

        mock_client = AsyncMock()
        mock_client.get_caller_history.return_value = CallerHistory(
            has_history=True,
            total_calls=2,
            last_topic="daftar",
        )
        with patch("lifecycle.call_tracker.get_platform_client", return_value=mock_client):
            result = await get_caller_history(
                phone="+923901234567",
                tenant_id="youth-agency",
            )
            assert result.has_history is True
            assert result.last_topic == "daftar"


async def test_shutdown_writes_language_in_metadata(monkeypatch):
    """The call-record PATCH on shutdown includes metadata.language."""
    from unittest.mock import MagicMock

    from lifecycle.shutdown import register_shutdown_callbacks

    captured = {}

    async def fake_update_call(**kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr(
        "lifecycle.call_tracker.update_call",
        fake_update_call,
    )

    ctx = MagicMock()
    callback_holder = {}

    def _add(cb):
        callback_holder["cb"] = cb
        return cb

    ctx.add_shutdown_callback = _add
    ctx.room.name = "room-1"

    agent = MagicMock()
    agent.call_db_id = "507f1f77bcf86cd799439011"
    agent.call_start_time = 0
    agent.conversation_history = []
    agent.transferred = False
    agent.language = "ru"
    agent.agent_identity = "agent-1"
    agent.transfer_reason = ""
    agent.ai_summary = ""
    agent.murojaat_id = None

    config = MagicMock()
    config.tenant.slug = "paynet"
    config.tenant.id = "507f1f77bcf86cd799439011"

    register_shutdown_callbacks(ctx, agent, config)
    await callback_holder["cb"]()

    assert "metadata" in captured
    assert captured["metadata"].get("language") == "ru"
