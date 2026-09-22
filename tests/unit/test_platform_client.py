"""Tests for PlatformClient — the centralized HTTP client for Platform API."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from api.models import CallerHistory, MurojaatResult, OperatorStatus
from api.platform_client import PlatformClient, get_platform_client

# Valid MongoDB ObjectId for paths whose spec requires ^[0-9a-fA-F]{24}$.
_OID = "507f1f77bcf86cd799439011"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client():
    """Create a PlatformClient with test defaults (no env vars needed)."""
    c = PlatformClient(
        base_url="https://test.example.com",
        api_key="test-key-123",
        timeout=5.0,
    )
    yield c


# ---------------------------------------------------------------------------
# Constructor / singleton
# ---------------------------------------------------------------------------


class TestConstructor:
    def test_defaults_from_args(self, client: PlatformClient):
        assert client._base_url == "https://test.example.com"
        assert client._timeout == 5.0
        # httpx client should carry the API key header
        assert client._http.headers["X-API-Key"] == "test-key-123"

    def test_defaults_from_env(self, monkeypatch):
        monkeypatch.setenv("PLATFORM_API_URL", "https://env.example.com")
        monkeypatch.delenv("AGENT_API_KEY", raising=False)
        monkeypatch.setenv("INTERNAL_API_KEY", "env-key-456")
        c = PlatformClient()
        assert c._base_url == "https://env.example.com"
        assert c._http.headers["X-API-Key"] == "env-key-456"

    def test_prefers_agent_api_key_over_internal(self, monkeypatch):
        monkeypatch.setenv("PLATFORM_API_URL", "https://env.example.com")
        monkeypatch.setenv("AGENT_API_KEY", "agent-key-123")
        monkeypatch.setenv("INTERNAL_API_KEY", "internal-key-456")
        c = PlatformClient()
        assert c._http.headers["X-API-Key"] == "agent-key-123"

    def test_strips_trailing_api_from_base_url(self):
        """A base URL ending with /api must be normalized so we don't
        build /api/api/v1/... and 404."""
        c = PlatformClient(base_url="https://dev.example.com/api", api_key="k")
        assert c._base_url == "https://dev.example.com"

    def test_strips_trailing_slash_from_base_url(self):
        c = PlatformClient(base_url="https://dev.example.com/", api_key="k")
        assert c._base_url == "https://dev.example.com"

    def test_singleton_returns_same_instance(self, monkeypatch):
        monkeypatch.setenv("PLATFORM_API_URL", "https://singleton.test")
        monkeypatch.setenv("INTERNAL_API_KEY", "key")
        # Reset the module-level singleton
        import api.platform_client as mod

        mod._instance = None

        a = get_platform_client()
        b = get_platform_client()
        assert a is b

        # Cleanup
        mod._instance = None


# ---------------------------------------------------------------------------
# create_call
# ---------------------------------------------------------------------------


class TestCreateCall:
    @pytest.mark.asyncio
    async def test_success(self, client: PlatformClient):
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"data": {"id": "call-db-id-1"}}

        with patch.object(client._http, "post", new_callable=AsyncMock, return_value=mock_response):
            result = await client.create_call(
                tenant_id="tenant-1",
                call_sid="CS123",
                caller_phone="+923901234567",
                agent_phone="+923901111111",
            )

        assert result == "call-db-id-1"

    @pytest.mark.asyncio
    async def test_success_fallback_id(self, client: PlatformClient):
        """When response has no 'data' wrapper, fallback to top-level 'id'."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "call-db-id-2"}

        with patch.object(client._http, "post", new_callable=AsyncMock, return_value=mock_response):
            result = await client.create_call(
                tenant_id="t1",
                call_sid="CS456",
                caller_phone="+923900000000",
                agent_phone="+923901111111",
            )

        assert result == "call-db-id-2"

    @pytest.mark.asyncio
    async def test_sends_correct_body(self, client: PlatformClient):
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"data": {"id": "x"}}

        mock_post = AsyncMock(return_value=mock_response)
        with patch.object(client._http, "post", mock_post):
            await client.create_call(
                tenant_id="t1",
                call_sid="CS789",
                caller_phone="+923900000000",
                agent_phone="+923901111111",
                direction="outbound",
                metadata={"foo": "bar"},
            )

        call_args = mock_post.call_args
        url = call_args[0][0]
        body = call_args[1]["json"]

        assert url == "https://test.example.com/api/v1/internal/calls"
        assert body["tenant_id"] == "t1"
        assert body["call_sid"] == "CS789"
        assert body["caller_phone"] == "+923900000000"
        assert body["agent_phone"] == "+923901111111"
        assert body["direction"] == "outbound"
        assert body["metadata"] == {"foo": "bar"}
        assert body["status"] == "in_progress"
        assert "started_at" in body  # UTC ISO timestamp

    @pytest.mark.asyncio
    async def test_http_error_returns_none(self, client: PlatformClient):
        with patch.object(
            client._http, "post", new_callable=AsyncMock, side_effect=httpx.ConnectError("fail")
        ):
            result = await client.create_call(
                tenant_id="t1",
                call_sid="CS000",
                caller_phone="+923900000000",
                agent_phone="+923901111111",
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_non_success_status_returns_none(self, client: PlatformClient):
        mock_response = MagicMock()
        mock_response.status_code = 500
        with patch.object(client._http, "post", new_callable=AsyncMock, return_value=mock_response):
            result = await client.create_call(
                tenant_id="t1",
                call_sid="CS000",
                caller_phone="+923900000000",
                agent_phone="+923901111111",
            )
        assert result is None

    @pytest.mark.asyncio
    async def test_uses_tenant_specific_key_header(self, monkeypatch):
        monkeypatch.setenv("EXAMPLE_TENANT_AGENT_API_KEY", "example-key-xyz")
        monkeypatch.setenv("ENABLE_LEGACY_INTERNAL_API_KEY_FALLBACK", "false")
        c = PlatformClient(base_url="https://test.example.com")

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"data": {"id": "call-db-id-3"}}

        mock_post = AsyncMock(return_value=mock_response)
        with patch.object(c._http, "post", mock_post):
            result = await c.create_call(
                tenant_id=_OID,
                tenant_slug="example-tenant",
                call_sid="CS111",
                caller_phone="+92900000000",
                agent_phone="+92901111111",
            )
        assert result == "call-db-id-3"
        assert mock_post.call_args.kwargs["headers"]["X-API-Key"] == "example-key-xyz"

    @pytest.mark.asyncio
    async def test_missing_tenant_key_skips_request(self, monkeypatch):
        monkeypatch.delenv("EXAMPLE_TENANT_AGENT_API_KEY", raising=False)
        monkeypatch.delenv("AGENT_API_KEY", raising=False)
        monkeypatch.delenv("INTERNAL_API_KEY", raising=False)
        monkeypatch.setenv("ENABLE_LEGACY_INTERNAL_API_KEY_FALLBACK", "false")
        c = PlatformClient(base_url="https://test.example.com")

        mock_post = AsyncMock()
        with patch.object(c._http, "post", mock_post):
            result = await c.create_call(
                tenant_id=_OID,
                tenant_slug="example-tenant",
                call_sid="CS112",
                caller_phone="+92900000000",
                agent_phone="+92901111111",
            )
        assert result is None
        mock_post.assert_not_called()

    @pytest.mark.asyncio
    async def test_prefers_yaml_tenant_slug_for_key_resolution(self, monkeypatch):
        monkeypatch.setenv("ACME_AGENT_API_KEY", "acme-key-xyz")
        monkeypatch.setenv("ENABLE_LEGACY_INTERNAL_API_KEY_FALLBACK", "false")
        c = PlatformClient(base_url="https://test.example.com")

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"data": {"id": "call-db-id-4"}}

        mock_post = AsyncMock(return_value=mock_response)
        with patch.object(c._http, "post", mock_post):
            result = await c.create_call(
                tenant_id=_OID,
                tenant_slug="acme",
                call_sid="CS113",
                caller_phone="+92900000000",
                agent_phone="+92901111111",
            )
        assert result == "call-db-id-4"
        assert mock_post.call_args.kwargs["headers"]["X-API-Key"] == "acme-key-xyz"


# ---------------------------------------------------------------------------
# update_call
# ---------------------------------------------------------------------------


class TestUpdateCall:
    @pytest.mark.asyncio
    async def test_success(self, client: PlatformClient):
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_patch = AsyncMock(return_value=mock_response)
        with patch.object(client._http, "patch", mock_patch):
            result = await client.update_call(
                call_db_id=_OID,
                tenant_id="t1",
                status="completed",
                duration_seconds=120,
                transcript=[{"role": "user", "text": "hello"}],
                transcript_text="hello",
                ai_summary="A short call",
                transfer_reason=None,
                ended_at="2026-01-01T00:00:00Z",
                metadata={"key": "val"},
            )

        assert result is True
        call_args = mock_patch.call_args
        assert call_args[0][0] == f"https://test.example.com/api/v1/internal/calls/{_OID}"
        body = call_args[1]["json"]
        assert body["tenant_id"] == "t1"
        assert body["status"] == "completed"
        assert body["duration_seconds"] == 120
        assert body["transcript_text"] == "hello"
        assert body["ai_summary"] == "A short call"
        assert body["ended_at"] == "2026-01-01T00:00:00Z"

    @pytest.mark.asyncio
    async def test_rejects_non_objectid_call_db_id(self, client: PlatformClient):
        """Spec says path :id must match ^[0-9a-fA-F]{24}$."""
        mock_patch = AsyncMock()
        with patch.object(client._http, "patch", mock_patch):
            result = await client.update_call(
                call_db_id="not-an-objectid",
                tenant_id="t1",
            )
        assert result is False
        mock_patch.assert_not_called()

    @pytest.mark.asyncio
    async def test_requires_tenant_id(self, client: PlatformClient):
        """Spec marks tenant_id as required in PATCH body."""
        mock_patch = AsyncMock()
        with patch.object(client._http, "patch", mock_patch):
            result = await client.update_call(call_db_id=_OID, tenant_id="")
        assert result is False
        mock_patch.assert_not_called()

    @pytest.mark.asyncio
    async def test_failure_returns_false(self, client: PlatformClient):
        with patch.object(
            client._http,
            "patch",
            new_callable=AsyncMock,
            side_effect=httpx.TimeoutException("timeout"),
        ):
            result = await client.update_call(
                call_db_id=_OID,
                tenant_id="t1",
                status="completed",
                duration_seconds=0,
            )
        assert result is False


# ---------------------------------------------------------------------------
# get_caller_history
# ---------------------------------------------------------------------------


class TestGetCallerHistory:
    @pytest.mark.asyncio
    async def test_success(self, client: PlatformClient):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "has_history": True,
                "total_calls": 5,
                "last_topic": "appeal",
                "last_summary": "Discussed housing",
                "last_call_status": "completed",
            }
        }

        with patch.object(client._http, "get", new_callable=AsyncMock, return_value=mock_response):
            result = await client.get_caller_history(phone="+923901234567", tenant_id="t1")

        assert isinstance(result, CallerHistory)
        assert result.has_history is True
        assert result.total_calls == 5
        assert result.last_topic == "appeal"

    @pytest.mark.asyncio
    async def test_sends_correct_params(self, client: PlatformClient):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": {}}

        mock_get = AsyncMock(return_value=mock_response)
        with patch.object(client._http, "get", mock_get):
            await client.get_caller_history(phone="+923901234567", tenant_id="t1")

        call_args = mock_get.call_args
        assert call_args[0][0] == "https://test.example.com/api/v1/internal/caller-history"
        # Server spec uses `caller_phone`, not `phone`.
        assert call_args[1]["params"] == {
            "caller_phone": "+923901234567",
            "tenant_id": "t1",
        }

    @pytest.mark.asyncio
    async def test_error_returns_empty_dataclass(self, client: PlatformClient):
        with patch.object(
            client._http, "get", new_callable=AsyncMock, side_effect=Exception("boom")
        ):
            result = await client.get_caller_history(phone="+923901234567", tenant_id="t1")

        assert isinstance(result, CallerHistory)
        assert result.has_history is False
        assert result.total_calls == 0


# ---------------------------------------------------------------------------
# request_operator
# ---------------------------------------------------------------------------


class TestRequestOperator:
    @pytest.mark.asyncio
    async def test_success(self, client: PlatformClient):
        mock_response = MagicMock()
        mock_response.status_code = 200

        mock_post = AsyncMock(return_value=mock_response)
        with patch.object(client._http, "post", mock_post):
            result = await client.request_operator(
                call_db_id=_OID,
                tenant_id="t1",
                transfer_reason="angry customer",
                ai_summary="Customer needs help",
                room_name="room-xyz",
                agent_identity="agent-1",
            )

        assert result is True
        call_args = mock_post.call_args
        assert call_args[0][0] == f"https://test.example.com/api/v1/internal/calls/{_OID}/operator"
        body = call_args[1]["json"]
        assert body["tenant_id"] == "t1"
        assert body["transfer_reason"] == "angry customer"
        assert body["ai_summary"] == "Customer needs help"
        assert body["room_name"] == "room-xyz"
        assert body["agent_identity"] == "agent-1"
        assert body["metadata"]["handoff_mode"] == "web_livekit_only"

    @pytest.mark.asyncio
    async def test_error_returns_false(self, client: PlatformClient):
        with patch.object(
            client._http, "post", new_callable=AsyncMock, side_effect=Exception("fail")
        ):
            result = await client.request_operator(
                call_db_id=_OID,
                tenant_id="t1",
                transfer_reason="test",
                ai_summary="test",
                room_name="room-1",
                agent_identity="agent-1",
            )
        assert result is False


# ---------------------------------------------------------------------------
# get_operator_status
# ---------------------------------------------------------------------------


class TestGetOperatorStatus:
    @pytest.mark.asyncio
    async def test_success(self, client: PlatformClient):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {
                "status": "transferred",
                "operator_id": "op-1",
                "operator_name": "Aziz",
            }
        }

        with patch.object(client._http, "get", new_callable=AsyncMock, return_value=mock_response):
            result = await client.get_operator_status(call_db_id=_OID, tenant_id="t1")

        assert isinstance(result, OperatorStatus)
        assert result.status == "transferred"
        assert result.operator_id == "op-1"
        assert result.operator_name == "Aziz"

    @pytest.mark.asyncio
    async def test_error_returns_unknown(self, client: PlatformClient):
        with patch.object(
            client._http, "get", new_callable=AsyncMock, side_effect=Exception("fail")
        ):
            result = await client.get_operator_status(call_db_id=_OID, tenant_id="t1")

        assert isinstance(result, OperatorStatus)
        assert result.status == "unknown"
        assert result.operator_id is None


# ---------------------------------------------------------------------------
# submit_murojaat
# ---------------------------------------------------------------------------


class TestSubmitMurojaat:
    @pytest.mark.asyncio
    async def test_success(self, client: PlatformClient):
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "data": {"id": "murojaat-1"},
            "success": True,
        }

        mock_post = AsyncMock(return_value=mock_response)
        with patch.object(client._http, "post", mock_post):
            result = await client.submit_murojaat(
                tenant_id="t1",
                content="Need housing help",
                full_name="Alisher Karimov",
                age=25,
                region="Toshkent",
                district="Yunusobod",
                murojaat_id="m-1",
            )

        assert isinstance(result, MurojaatResult)
        assert result.success is True
        assert result.murojaat_id == "murojaat-1"
        assert result.error is None

        # Verify body format
        call_args = mock_post.call_args
        url = call_args[0][0]
        body = call_args[1]["json"]
        assert url == "https://test.example.com/api/v1/murojatlar/voice-assistant"
        assert body["tenant_id"] == "t1"
        assert body["murojaat_id"] == "m-1"
        # `sana` is an ISO-8601 UTC datetime with Z suffix (e.g.
        # "2026-04-13T10:23:45.123Z"), not just a date.
        assert body["sana"].endswith("Z")
        assert "T" in body["sana"]
        assert body["murojaat_qiluvchi"]["ism_familiya"] == "Alisher Karimov"
        assert body["murojaat_qiluvchi"]["yosh"] == 25
        assert body["murojaat_qiluvchi"]["viloyat"] == "Toshkent"
        assert body["murojaat_qiluvchi"]["tuman"] == "Yunusobod"
        assert body["murojaat"]["mazmuni"] == "Need housing help"
        assert body["metadata"]["source"] == "navai-voice-agent"

    @pytest.mark.asyncio
    async def test_error_returns_failure(self, client: PlatformClient):
        with patch.object(
            client._http, "post", new_callable=AsyncMock, side_effect=Exception("boom")
        ):
            result = await client.submit_murojaat(
                tenant_id="t1",
                content="test",
                full_name="Test",
                age=20,
                region="Toshkent",
                district="Chilonzor",
            )

        assert isinstance(result, MurojaatResult)
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_non_success_status(self, client: PlatformClient):
        mock_response = MagicMock()
        mock_response.status_code = 422
        mock_response.text = "Validation failed"

        with patch.object(client._http, "post", new_callable=AsyncMock, return_value=mock_response):
            result = await client.submit_murojaat(
                tenant_id="t1",
                content="test",
                full_name="Test",
                age=20,
                region="Toshkent",
                district="Chilonzor",
            )

        assert result.success is False
        assert result.error is not None


# ---------------------------------------------------------------------------
# check_health
# ---------------------------------------------------------------------------


class TestCheckHealth:
    @pytest.mark.asyncio
    async def test_healthy_when_api_returns_json(self, client: PlatformClient):
        """Any JSON response means the API is reachable, even a 401."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {"content-type": "application/json; charset=utf-8"}

        with patch.object(client._http, "get", new_callable=AsyncMock, return_value=mock_response):
            assert await client.check_health() is True

    @pytest.mark.asyncio
    async def test_unhealthy_when_frontend_html_served(self, client: PlatformClient):
        """`/health` behind a SPA catch-all returns 200 text/html — not healthy."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html; charset=utf-8"}

        with patch.object(client._http, "get", new_callable=AsyncMock, return_value=mock_response):
            assert await client.check_health() is False

    @pytest.mark.asyncio
    async def test_connection_error(self, client: PlatformClient):
        with patch.object(
            client._http, "get", new_callable=AsyncMock, side_effect=httpx.ConnectError("down")
        ):
            assert await client.check_health() is False


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


class TestClose:
    @pytest.mark.asyncio
    async def test_close(self, client: PlatformClient):
        with patch.object(client._http, "aclose", new_callable=AsyncMock) as mock_close:
            await client.close()
            mock_close.assert_awaited_once()


# ---------------------------------------------------------------------------
# get_caller_history — last_language field
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_caller_history_parses_last_language(monkeypatch):
    """Server includes metadata.language on the latest call — we surface it."""
    from api.models import CallerHistory
    from api.platform_client import PlatformClient

    class _FakeResp:
        status_code = 200

        def json(self):
            return {
                "data": {
                    "previous_calls": 3,
                    "last_topic": "Кэшбэк",
                    "last_summary": "Клиент спрашивал про подключение кэшбэка.",
                    "last_language": "ru",
                }
            }

    monkeypatch.setenv("ENABLE_LEGACY_INTERNAL_API_KEY_FALLBACK", "true")
    monkeypatch.setenv("AGENT_API_KEY", "generic-test-key")
    client = PlatformClient()

    async def _fake_get(url, params=None, headers=None):
        return _FakeResp()

    monkeypatch.setattr(client._http, "get", _fake_get)

    history = await client.get_caller_history(
        phone="+923901234567",
        tenant_id="507f1f77bcf86cd799439011",
    )
    assert isinstance(history, CallerHistory)
    assert history.last_language == "ru"


@pytest.mark.asyncio
async def test_get_caller_history_defaults_last_language_when_absent(monkeypatch):
    """Missing field -> empty string, not a crash."""
    from api.platform_client import PlatformClient

    class _FakeResp:
        status_code = 200

        def json(self):
            return {"data": {"previous_calls": 1, "last_topic": "x"}}

    monkeypatch.setenv("ENABLE_LEGACY_INTERNAL_API_KEY_FALLBACK", "true")
    monkeypatch.setenv("AGENT_API_KEY", "generic-test-key")
    client = PlatformClient()

    async def _fake_get(url, params=None, headers=None):
        return _FakeResp()

    monkeypatch.setattr(client._http, "get", _fake_get)

    history = await client.get_caller_history(
        phone="+923901234567",
        tenant_id="507f1f77bcf86cd799439011",
    )
    assert history.last_language == ""
