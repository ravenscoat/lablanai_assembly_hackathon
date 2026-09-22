from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.loader import ConfigLoader


@pytest.mark.asyncio
async def test_fetch_from_api_uses_default_key_when_legacy_enabled(monkeypatch):
    # Pre-tenant config lookup happens BEFORE the slug is known, so the generic
    # resolver has no per-tenant key to use and falls back to the default/legacy
    # key (only when ENABLE_LEGACY_INTERNAL_API_KEY_FALLBACK is on).
    monkeypatch.setenv("INTERNAL_API_KEY", "internal-key-abc")
    monkeypatch.setenv("ENABLE_LEGACY_INTERNAL_API_KEY_FALLBACK", "true")

    loader = ConfigLoader()

    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_get = AsyncMock(return_value=mock_response)

    mock_client = MagicMock()
    mock_client.get = mock_get

    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)

    with patch("config.loader.httpx.AsyncClient", return_value=mock_ctx):
        result = await loader._fetch_from_api("+92781225396")

    assert result is None
    assert mock_get.call_args.kwargs["headers"]["X-API-Key"] == "internal-key-abc"


@pytest.mark.asyncio
async def test_fetch_from_api_missing_key_fails_closed(monkeypatch):
    monkeypatch.delenv("AGENT_API_KEY", raising=False)
    monkeypatch.delenv("INTERNAL_API_KEY", raising=False)
    monkeypatch.setenv("ENABLE_LEGACY_INTERNAL_API_KEY_FALLBACK", "false")

    loader = ConfigLoader()

    with patch("config.loader.httpx.AsyncClient") as mock_client_cls:
        result = await loader._fetch_from_api("+92781225396")

    assert result is None
    mock_client_cls.assert_not_called()
