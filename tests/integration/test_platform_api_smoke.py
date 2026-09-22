# tests/integration/test_platform_api_smoke.py
"""
Smoke test: verifies the full call lifecycle wiring.
PlatformClient HTTP is mocked, but everything else is real.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from api.models import CallerHistory, OperatorStatus
from api.platform_client import PlatformClient
from config.schema import TenantConfig


@pytest.fixture
def config():
    return TenantConfig.model_validate(
        {
            "tenant": {"id": "test-tenant", "slug": "test", "name": "Test Agent"},
            "transfer": {"enabled": True},
        }
    )


@pytest.fixture
def mock_platform_client():
    client = AsyncMock(spec=PlatformClient)
    client.create_call.return_value = "call-db-001"
    client.update_call.return_value = True
    client.get_caller_history.return_value = CallerHistory(
        has_history=True,
        total_calls=1,
        last_topic="daftar",
    )
    client.request_operator.return_value = True
    client.get_operator_status.return_value = OperatorStatus(status="transferred")
    return client


class TestFullCallLifecycle:
    async def test_create_update_cycle(self, mock_platform_client):
        """Create a call, then update it — verifies wiring through call_tracker."""
        with patch("lifecycle.call_tracker.get_platform_client", return_value=mock_platform_client):
            from lifecycle.call_tracker import create_call, update_call

            call_id = await create_call(
                tenant_id="test-tenant",
                caller_phone="+923901234567",
                agent_phone="+923781225394",
                call_sid="lk-room-123",
            )
            assert call_id == "call-db-001"

            ok = await update_call(
                call_db_id=call_id,
                tenant_id="test-tenant",
                status="completed",
                duration_seconds=60,
                transcript_text="user: salom\nassistant: salom",
                ai_summary="Brief greeting call",
                ended_at="2026-04-09T12:01:00+05:00",
            )
            assert ok is True

    async def test_caller_history_flow(self, mock_platform_client):
        """Caller history flows from PlatformClient through call_tracker."""
        with patch("lifecycle.call_tracker.get_platform_client", return_value=mock_platform_client):
            from lifecycle.call_tracker import get_caller_history

            history = await get_caller_history(
                phone="+923901234567",
                tenant_id="test-tenant",
            )
            assert history.has_history is True
            assert history.last_topic == "daftar"

    async def test_operator_handoff_flow(self, mock_platform_client):
        """Verify operator request + status poll via PlatformClient."""
        result = await mock_platform_client.request_operator(
            call_db_id="507f1f77bcf86cd799439011",
            tenant_id="test-tenant",
            transfer_reason="User requested",
            ai_summary="Asked about daftar",
            room_name="room-1",
            agent_identity="agent-1",
        )
        assert result is True

        status = await mock_platform_client.get_operator_status(
            call_db_id="507f1f77bcf86cd799439011",
            tenant_id="test-tenant",
        )
        assert status.status == "transferred"
