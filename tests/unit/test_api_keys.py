from __future__ import annotations

from config.api_keys import (
    resolve_tenant_api_key,
    resolve_tenant_api_key_for_phone,
    slug_to_env_var,
)


def test_slug_to_env_var():
    assert slug_to_env_var("example-tenant") == "EXAMPLE_TENANT_AGENT_API_KEY"
    assert slug_to_env_var("acme") == "ACME_AGENT_API_KEY"
    assert slug_to_env_var("Acme Corp 1") == "ACME_CORP_1_AGENT_API_KEY"
    assert slug_to_env_var("") == ""


def test_resolve_tenant_api_key_by_slug(monkeypatch):
    monkeypatch.setenv("EXAMPLE_TENANT_AGENT_API_KEY", "example-key")
    monkeypatch.setenv("ACME_AGENT_API_KEY", "acme-key")
    monkeypatch.setenv("ENABLE_LEGACY_INTERNAL_API_KEY_FALLBACK", "false")

    assert resolve_tenant_api_key("example-tenant") == "example-key"
    assert resolve_tenant_api_key("acme") == "acme-key"


def test_resolve_tenant_api_key_for_phone_uses_default(monkeypatch):
    # The data-driven resolver has no static phone->key table; phone lookup
    # falls back to the (optional) legacy/default key.
    monkeypatch.setenv("ENABLE_LEGACY_INTERNAL_API_KEY_FALLBACK", "true")
    monkeypatch.setenv("INTERNAL_API_KEY", "internal-key")

    assert resolve_tenant_api_key_for_phone("+92000000000") == "internal-key"


def test_resolve_tenant_api_key_missing_fails_closed(monkeypatch):
    monkeypatch.delenv("EXAMPLE_TENANT_AGENT_API_KEY", raising=False)
    monkeypatch.delenv("AGENT_API_KEY", raising=False)
    monkeypatch.delenv("INTERNAL_API_KEY", raising=False)
    monkeypatch.setenv("ENABLE_LEGACY_INTERNAL_API_KEY_FALLBACK", "false")

    assert resolve_tenant_api_key("example-tenant") == ""
    assert resolve_tenant_api_key_for_phone("+92000000000") == ""
