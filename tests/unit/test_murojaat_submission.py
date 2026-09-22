"""Tests for AppealAgent murojaat submission."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from api.models import MurojaatResult


class TestAppealSubmission:
    @pytest.mark.asyncio
    async def test_submit_uses_platform_client(self):
        """AppealAgent._submit() should use PlatformClient.submit_murojaat()."""
        mock_client = AsyncMock()
        mock_client.submit_murojaat.return_value = MurojaatResult(
            success=True,
            murojaat_id="mur-789",
        )

        from agents.sub_agents.appeal import AppealAgent
        from config.schema import TenantConfig

        config = TenantConfig.model_validate(
            {
                "tenant": {"id": "youth-agency", "name": "Test"},
            }
        )

        agent = AppealAgent.__new__(AppealAgent)
        agent._parent_config = config
        agent._platform_client = mock_client
        agent._fields = []  # _submit reads _data directly; fields unused here
        agent._data = {
            "content": "Korrupsiya haqida",
            "full_name": "Ali Valiyev",
            "age": 22,
            "region": "Toshkent",
            "district": "Chilonzor",
            "position": "Talaba",
        }
        agent._parent_call_db_id = None
        agent._parent_caller_phone = None
        agent._parent_agent = None

        result = await agent._submit()
        assert result is True
        mock_client.submit_murojaat.assert_called_once_with(
            tenant_id="youth-agency",
            tenant_slug="",
            content="Korrupsiya haqida",
            full_name="Ali Valiyev",
            age=22,
            region="Toshkent",
            district="Chilonzor",
            position="Talaba",
            call_id=None,
            caller_phone=None,
        )

    @pytest.mark.asyncio
    async def test_submit_fallback_when_no_client(self):
        """Without PlatformClient on the agent, fall back to singleton."""
        from agents.sub_agents.appeal import AppealAgent
        from config.schema import TenantConfig

        config = TenantConfig.model_validate(
            {
                "tenant": {"id": "test", "name": "Test"},
            }
        )
        agent = AppealAgent.__new__(AppealAgent)
        agent._parent_config = config
        agent._platform_client = None
        agent._fields = []
        agent._data = {
            "content": "Test",
            "full_name": "Test",
            "age": 20,
            "region": "Test",
            "district": "Test",
        }
        agent._parent_call_db_id = None
        agent._parent_caller_phone = None
        agent._parent_agent = None

        mock_client = AsyncMock()
        mock_client.submit_murojaat.return_value = MurojaatResult(success=True, murojaat_id="m-1")
        with patch("api.platform_client.get_platform_client", return_value=mock_client):
            result = await agent._submit()
            assert result is True
